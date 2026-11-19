"""
Sign and CV controllers of the simulator (the controllers of VSL-S and CV speed pacing; CAZP, VSL-L, the zipper-merge
sign and CV early merge are implemented in policy.py).

Every controller sees measurements only: the occupancy of the merge (bottleneck) area, segment-detector speeds
(application-area speed v_app, estimated queue tail) and, for CV strategies, what the selected CVs sense.

    NC           no control
    VSL_FIXZONE  VSL-S: Q10(b v0) of a PI feedback on the occupancy of the merge area (Carlson et al., 2011), with the
                 credibility rule, minimum holding time and step limit; one sign (two gantries at the same position)
    CV_MTFC      CV speed pacing: the same feedback speed is sent to selected CVs in all lanes, leader rule

Attributes shared by every controller: signs (km/h), g_ctrl (km/h), b, v_req (m/s), on, uses_sign, uses_pacers, log,
n_sign_changes, n_mask_changes, n_masked, x_e, x_tail, queued, app_area, max_sign_step_kmh, min_sign_kmh,
t_last_change, binding ({vid: "vreq"|"vmin"|"leader"}).
"""
from __future__ import annotations
import numpy as np

from config import KMH, FIXZONE, STRATEGIES, ceil10

BIND_CODE = {"sign": 0, "vreq": 1, "band": 2, "vmin": 3, "leader": 4}


def q10(v_kmh: float) -> float:
    return 10.0 * round(v_kmh / 10.0)


class PIReg:
    """Velocity-form PI regulator of the bottleneck occupancy -> b in [b_min, 1], with activation persistence:
    `on` after n_on consecutive periods with b < 0.999, off after one full period at b = 1 (off-delay)."""

    def __init__(self, KP, KI, b_min, occ_set, n_on=2):
        self.KP, self.KI, self.bmin, self.o_set, self.n_on = KP, KI, b_min, occ_set, max(1, int(n_on))
        self.b, self.o_prev = 1.0, None
        self.on_cnt, self.off_cnt, self.on = 0, 2, False

    def step(self, occ: float, freeze_down: bool = False) -> float:
        if self.o_prev is None:
            self.o_prev = occ
        dP = self.KP * (self.o_prev - occ)
        dI = self.KI * (self.o_set - occ)
        if freeze_down and dI < 0:          # conditional integration (anti-windup): freeze the I term only
            dI = 0.0
        self.b = float(np.clip(self.b + dP + dI, self.bmin, 1.0))
        self.o_prev = occ
        if self.b < 0.999:
            self.on_cnt, self.off_cnt = self.on_cnt + 1, 0
        else:
            self.on_cnt, self.off_cnt = 0, self.off_cnt + 1
        if self.on_cnt >= self.n_on:
            self.on = True
        elif self.off_cnt >= 2:             # one regulator period at b = 1 elapsed -> off (avoids dumping pacers)
            self.on = False
        return self.b


class Controller:
    uses_sign = False
    uses_pacers = False
    cred_floor = False                      # credibility rule on the displayed value (VSL-S)

    def __init__(self, cfg):
        self.cfg = cfg
        g, c = cfg.geo, cfg.ctrl
        self.v0 = g.v_base_kmh
        self.gx = [float(x) for x in g.gantries]
        self.n_g = len(self.gx)
        self.fixed = g.fixed
        self.signs = [self.v0] * self.n_g               # km/h, upstream -> downstream
        self.g_ctrl = self.v0                           # controlled value (km/h) shown on the application gantries
        self.reg = PIReg(c.K_P, c.K_I, c.b_min, c.occ_set, c.n_on)
        self.b = 1.0
        self.v_req = g.v_base               # m/s
        self.on = False
        self.log = {}
        self.n_sign_changes = 0
        self.max_sign_step_kmh = 0.0        # largest per-update change of the controlled value
        self.min_sign_kmh = self.v0         # lowest value ever displayed
        self.t_last_change = -1e9
        self.binding = {}                   # vid -> constraint that bound the last advisory
        self.sign_log = []                  # records of model-based sign updates (none for these controllers)
        self.M_last = {}                    # last measurement dict seen (v_app for the credibility rule)
        # tail-following deployment state
        self.x_tail, self.queued = None, False
        self.x_e = g.release_x              # control end
        self.mask = np.zeros(self.n_g, dtype=bool)      # True: gantry at / inside the queue -> shows v0
        self._t_flip = np.full(self.n_g, -1e9)
        self.n_mask_changes = 0
        self.app_idx, self.stair_idx = [], None
        self._layout()

    # hooks called by the simulation loop ---------------------------------
    def set_tail(self, t, x_tail, queued):
        """5-s clock, before exec_targets: control end, mask (with dwell), application set."""
        g, c = self.cfg.geo, self.cfg.ctrl
        self.queued = bool(queued) and x_tail is not None
        self.x_tail = x_tail
        if self.fixed:
            return
        self.x_e = min(float(x_tail) - c.tail_hys_m, g.release_x) if self.queued else g.release_x
        want = np.array([x >= self.x_e - 1e-9 for x in self.gx], dtype=bool)
        flip = (want != self.mask) & (t - self._t_flip >= c.tail_dwell_s)
        if flip.any():
            self.mask[flip] = want[flip]
            self._t_flip[flip] = t
            if self.uses_sign:
                self.n_mask_changes += int(flip.sum())
            self._layout()
            self._render()

    def regulate(self, t, occ, M=None):  # every reg_period(cfg); M (with v_app) for the credibility rule
        pass

    def exec_targets(self, t, M, sensed):  # every T_exec -> {vid: target m/s}; sets self.binding
        self.binding = {}
        return {}

    def pacers_on(self):
        return False

    # derived state ---------------------------------------------------------
    @property
    def n_masked(self) -> int:
        return int(self.mask.sum())

    @property
    def app_area(self):
        """(x0, x1) of the current application area (m) or None when the queue covers the whole array."""
        g = self.cfg.geo
        if self.fixed:
            return (self.gx[0], g.release_x)
        if not self.app_idx:
            return None
        return (self.gx[self.app_idx[0]], min(self.gx[self.app_idx[-1]] + g.d_gantry, g.release_x))

    # helpers ---------------------------------------------------------------
    def _regulate(self, occ, freeze=False, M=None):
        if M is not None:
            self.M_last = M
        self.b = self.reg.step(occ, freeze_down=freeze)
        self.v_req = self.b * self.cfg.geo.v_base
        self.on = self.reg.on

    def _layout(self):
        """Application gantries = the n_app most downstream unmasked ones; staircase = the next upstream one."""
        if self.fixed:
            self.app_idx, self.stair_idx = list(range(self.n_g)), None
            return
        free = [j for j in range(self.n_g) if not self.mask[j]]
        n_app = self.cfg.geo.n_app
        self.app_idx = free[-n_app:]
        rest = free[:-n_app]
        self.stair_idx = rest[-1] if rest else None

    def _render(self):
        """Display from g_ctrl and the layout (no counting: hardware mask flips are counted in set_tail)."""
        g, c = self.cfg.geo, self.cfg.ctrl
        n, v0 = self.n_g, self.v0
        new = [v0] * n
        if self.fixed:                                   # downstream gantry g_ctrl, +10 km/h per gantry upstream
            for j in range(n):
                new[j] = min(v0, self.g_ctrl + 10.0 * (n - 1 - j))
        else:
            for j in self.app_idx:
                new[j] = self.g_ctrl
            if self.stair_idx is not None:
                new[self.stair_idx] = min(v0, self.g_ctrl + g.staircase_kmh)
        for j in range(n - 2, -1, -1):                   # consecutive UNMASKED gantries differ by <= gantry_diff
            if not self.mask[j] and not self.mask[j + 1]:
                new[j] = min(new[j], new[j + 1] + c.gantry_diff_kmh)
        self.min_sign_kmh = min(self.min_sign_kmh, min(new))
        self.signs = new

    def _set_signs(self, v_ctrl_kmh):
        """Controlled value -> application gantries (per-update step_max, quantised); counts n_sign_changes."""
        c = self.cfg.ctrl
        v = max(c.vs_min_kmh, min(self.v0, v_ctrl_kmh))
        v = float(np.clip(v, self.g_ctrl - c.step_max_kmh, self.g_ctrl + c.step_max_kmh))
        v = q10(v)
        if v != self.g_ctrl:
            self.n_sign_changes += 1
            self.max_sign_step_kmh = max(self.max_sign_step_kmh, abs(v - self.g_ctrl))
            self.g_ctrl = v
        self._render()

    def _cred_floor_kmh(self, M):
        """ceil10(v_app - cred) or None (no credibility rule / no application-area speed)."""
        if not self.cred_floor:
            return None
        M = M if M is not None else self.M_last
        v_app = M.get("v_app") if M else None
        return None if v_app is None else ceil10(v_app / KMH - self.cfg.ctrl.cred_kmh)

    def _apply_sign_target(self, t, target_kmh, M=None) -> bool:
        """Display logic of the sign: credibility rule, quantisation, dead band, minimum holding time of the
        controlled value. Returns True when the display changed."""
        c = self.cfg.ctrl
        f = self._cred_floor_kmh(M)
        if f is not None:
            target_kmh = max(target_kmh, f)
        target_kmh = min(target_kmh, self.v0)
        if abs(target_kmh - self.g_ctrl) >= 5.0 + c.deadband_kmh and t - self.t_last_change >= c.hold:
            self._set_signs(q10(target_kmh))
            self.t_last_change = t
            return True
        return False

    def signs_ms(self):
        return [v * KMH for v in self.signs]


# ------------------------------------------------------------------ baselines
class NoControl(Controller):
    pass


class SignMTFC(Controller):
    """VSL-S: local mainstream traffic flow control (Carlson et al., 2011), Q10(b v0) on the application area with
    the credibility rule and the minimum holding time."""
    uses_sign = True
    cred_floor = True

    def regulate(self, t, occ, M=None):
        self._regulate(occ, M=M)
        self._apply_sign_target(t, q10(self.b * self.v0), M)


class PacerBase(Controller):
    uses_pacers = True

    def pacers_on(self):
        return self.on

    def _limit(self, v):
        return max(v, self.cfg.cv.v_min_kmh * KMH)

    def _leader_v(self, s):
        """v* = max(min(mean follower speed, leader speed), v_min); None without such information."""
        cand = [x for x in (s.get("v_f"), s.get("v_l")) if x is not None]
        return self._limit(min(cand)) if cand else None

    def _leader_cap(self, v, s):
        """Leader rule applied to a target: (v, capped)."""
        cap = self._leader_v(s)
        if cap is not None and cap < v:
            return cap, True
        return v, False


class CVMTFC(PacerBase):
    """CV speed pacing: the feedback speed max(v_req, v_min) is sent to the selected CVs in all lanes (no sign),
    limited by the leader rule."""

    def regulate(self, t, occ, M=None):
        self._regulate(occ, M=M)

    def exec_targets(self, t, M, sensed):
        out, self.binding = {}, {}
        vmin = self.cfg.cv.v_min_kmh * KMH
        for vid, s in sensed.items():
            v, bind = (self.v_req, "vreq") if self.v_req >= vmin else (vmin, "vmin")
            v, capped = self._leader_cap(v, s)
            out[vid], self.binding[vid] = v, ("leader" if capped else bind)
        return out


def make_controller(cfg) -> Controller:
    """Controller for cfg.ctrl.strategy. Strategies of FIXZONE set cfg.geo.deployment = "fixed" in place, so that the
    pacer manager and the metrics built afterwards see the same layout."""
    s = cfg.ctrl.strategy
    if s not in STRATEGIES:
        raise ValueError(f"unknown strategy {s!r}; valid: {sorted(STRATEGIES)}")
    if s in FIXZONE and cfg.geo.deployment != "fixed":
        cfg.geo.deployment = "fixed"
    return dict(NC=NoControl, VSL_FIXZONE=SignMTFC, CV_MTFC=CVMTFC)[s](cfg)


def reg_period(cfg):
    """Period of the feedback (s)."""
    return cfg.ctrl.mtfc_period

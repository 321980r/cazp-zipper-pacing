"""
Connected-vehicle pacers (fast actuator). In the paper this module moves the CVs of the CV speed pacing benchmark
in the fixed deployment; the exclusion of CVs near the lane end is switched off (runs.py).

* Zone (deployment "tail"): Z(t) = [x_e - L_app, x_e - m_sel] with the control end x_e = x_tail - tail_hys_m
  (queue detected) or release_x, clipped to [x_min_pacer, release_x - m_sel]; identical extent to the sign's
  application area, so every active pacer is sign-covered by construction. Deployment "fixed" keeps the
  zone [zone_start, release_x - 150].
* Selection: CVs inside the zone are selected up to n_max with a greedy (lane, cell) coverage rule: first one
  CV per uncovered cell (most downstream first), then the remaining room by position. Cells are counted
  relative to the zone start, n_cells = n_up * ceil(L_app / seg_len). CVs in the dropping lane close to the
  drop are not selected (they must merge). Whether a CV driver follows the advisories is decided once per
  vehicle at demand generation (`comp_set`).
* Release (tail): x >= x_tail - m_rel, or v < v_req - release_dv (a queue the estimator has not seen), or
  x >= release_x, or control off (fixed: release_x / off only). Release resets maxSpeed (50 m/s).
* Execution: advisory -> dead time -> ceiling. The ADVISORY changes by at most rate_kmh_per_5s per T_exec; each
  new advisory is queued and applied as the vehicle's maxSpeed after the driver's dead time theta.
* Sensing: adjacent-lane speeds within +-sense_range (noise from rng), follower / leader speeds in the own
  lane, and the compliance flag `comp`.
* Leakage: vehicles overtaking each compliant pacer in the OTHER lanes are counted (1-s checks); the
  moving-observer prediction sum_j k_j (v_j - v_p) is logged per pacer-minute for the mechanism test (always).
"""
from __future__ import annotations
import numpy as np
from core.demand import COL_CV
from core.controllers import BIND_CODE

GREEN = (46, 139, 87, 255)
GREY_OUT = (190, 150, 60, 255)


def _rgb(s):
    return tuple(int(float(c) * 255) for c in s.split(",")) + (255,)


def _idx(vid: str) -> int:
    try:
        return int(vid[1:])
    except ValueError:
        return -1


class PacerManager:
    def __init__(self, cfg, rng, k, comp_set, len_up: float):
        self.cfg, self.rng, self.k = cfg, rng, k          # rng = sensing-noise stream only
        self.comp_set = set(comp_set)
        g = cfg.geo
        self.fixed = g.fixed
        self.release_x = g.release_x
        self.seg = g.seg_len
        self.L_app, self.m_sel, self.m_rel, self.x_min = g.L_app, g.m_sel, g.m_rel, g.x_min_pacer
        self.hys = cfg.ctrl.tail_hys_m
        self.release_dv = cfg.ctrl.release_dv_kmh / 3.6
        self.x_e = g.release_x
        self.zone = (g.zone_start, g.release_x - 150.0) if self.fixed else self._tail_zone(self.x_e)
        self.zone_clipped = False         # tail zone hit x_min_pacer at the last select
        self.x_tail = None                # last tail estimate given to select (m) or None
        self.drop_lanes = set(range(g.n_up - g.n_down))
        self.x_drop = float(len_up)                        # from the loaded network
        self.active = {}          # vid -> dict(comp, lane, x, v, v_sel, adv, queue, theta, bind, ceil, t0, n_ot, band_s, win)
        self.done = set()
        self.was_pacer = {}       # vid -> dict(comp, t_sel, x_sel, lane_sel, theta, t_rel, n_ot, band_s, x_tail_at_sel)
        self.total_overtakes = 0
        self.pacer_seconds = 0.0          # compliant pacer time
        self.pacer_seconds_all = 0.0
        self.band_s = 0.0                 # compliant time with bind == "band"
        self.win_overtakes = 0
        self.win_overtakes_band = 0
        self.win_pacer_s = 0.0
        self.win_band_s = 0.0
        self.mech_log = []        # per pacer-minute samples (always)
        self.r_win = []           # (t, r) window overtake rates returned by pop_window
        self.snap = []            # (t, idx, x, lane, v, ceil, target, bind_code, x_tail) per set_targets, compliant pacers
        self._prev = None         # (x dict, lane dict) at the last 1-s check
        self._t = 0.0

    # ------------------------------------------------------------ zone / cells
    def _tail_zone(self, x_e):
        z0 = max(self.x_min, x_e - self.L_app)
        z1 = min(x_e - self.m_sel, self.release_x - self.m_sel)
        return (z0, max(z0, z1))

    def _cell(self, lane, x):
        return (int(lane), int((x - self.zone[0]) // self.seg))

    def _n_seg(self):
        span = self.L_app if not self.fixed else self.release_x - self.zone[0]
        return int(np.ceil(span / self.seg - 1e-9))

    def coverage(self):
        n_seg = self._n_seg()
        cells = {self._cell(p["lane"], p["x"]) for p in self.active.values() if p["comp"]}
        return {c for c in cells if 0 <= c[1] < n_seg}

    def n_cells(self):
        return self._n_seg() * self.cfg.geo.n_up

    def phi(self):
        return min(1.0, len(self.coverage()) / max(1, self.n_cells()))

    def _theta(self):
        """Execution dead time of this pacer's driver (s)."""
        tau = self.cfg.cv.tau_a
        return 0.0 if tau <= 0 else float(self.rng.lognormal(np.log(tau), 0.5))

    # ------------------------------------------------------------ selection
    def select(self, t, V, on: bool, x_tail=None, v_req=None):
        """Every T_exec: move the zone, release, select. x_tail = estimated queue tail (m) or None,
        v_req = regulator request (m/s) or None (release rule v < v_req - release_dv skipped)."""
        self._t = t
        self.x_tail = None if x_tail is None else float(x_tail)
        if not self.fixed:
            self.x_e = self.release_x if self.x_tail is None else min(self.x_tail - self.hys, self.release_x)
            self.zone = self._tail_zone(self.x_e)
            self.zone_clipped = self.x_e - self.L_app < self.x_min
        # release pacers that left the zone (or all, if control is off)
        for vid in list(self.active):
            i = V.idx.get(vid)
            rel = (not on) or i is None or V.x[i] >= self.release_x
            if not rel and not self.fixed:
                rel = ((self.x_tail is not None and V.x[i] >= self.x_tail - self.m_rel)
                       or (v_req is not None and V.v[i] < v_req - self.release_dv))
            if rel:
                self.release(vid)
        if not on:
            return
        for vid, p in self.active.items():
            i = V.idx[vid]
            p["x"], p["lane"], p["v"] = float(V.x[i]), int(V.lane[i]), float(V.v[i])
        room = self.cfg.cv.n_max - len(self.active)
        if room <= 0 or self.zone[1] <= self.zone[0]:
            return
        cov = self.coverage()
        cands = []
        for i in np.where(V.is_cv & (V.x > self.zone[0]) & (V.x < self.zone[1]))[0]:
            vid = V.ids[i]
            if vid in self.active or vid in self.done:
                continue
            if V.lane[i] in self.drop_lanes and self.x_drop - V.x[i] < self.cfg.cv.avoid_drop_lane_dist:
                continue
            cands.append((-float(V.x[i]), vid, int(i), self._cell(V.lane[i], V.x[i])))
        cands.sort()
        chosen, rest = [], []
        for item in cands:                          # pass 1: one candidate per uncovered cell (greedy)
            if len(chosen) >= room:
                break
            if item[3] not in cov:
                chosen.append(item)
                cov.add(item[3])
            else:
                rest.append(item)
        chosen += rest[:max(0, room - len(chosen))]  # pass 2: fill by position
        xt = np.nan if self.x_tail is None else self.x_tail
        for _, vid, i, _cell in chosen:
            comp = vid in self.comp_set
            self.active[vid] = dict(comp=comp, lane=int(V.lane[i]), x=float(V.x[i]), v=float(V.v[i]),
                                    v_sel=float(V.v[i]), adv=None, queue=[], theta=self._theta(), bind=None,
                                    ceil=None, t0=t, n_ot=0, band_s=0.0,
                                    win=dict(t0=t, pred=0.0, adj=0, lco=0, dv=[], v_adj=[], v_p=[], n_ot=0))
            self.was_pacer[vid] = dict(comp=comp, t_sel=t, x_sel=float(V.x[i]), lane_sel=int(V.lane[i]),
                                       theta=self.active[vid]["theta"], t_rel=np.nan, n_ot=0, band_s=0.0,
                                       x_tail_at_sel=xt)
            try:
                self.k.vehicle.setColor(vid, GREEN if comp else GREY_OUT)
            except Exception:
                pass

    def release(self, vid):
        """Release a pacer (left the zone, control off, or arrived) and write its episode."""
        p = self.active.pop(vid, None)
        self.done.add(vid)
        if p is None:
            return
        ep = self.was_pacer.setdefault(vid, dict(comp=p["comp"], t_sel=p["t0"], x_sel=p["x"], lane_sel=p["lane"],
                                                 theta=p["theta"], x_tail_at_sel=np.nan))
        ep.update(t_rel=self._t, n_ot=p["n_ot"], band_s=p["band_s"])
        try:
            self.k.vehicle.setMaxSpeed(vid, 50.0)
            self.k.vehicle.setColor(vid, _rgb(COL_CV))
        except Exception:
            pass

    def release_all(self):
        for vid in list(self.active):
            self.release(vid)

    # ------------------------------------------------------------ execution
    def set_targets(self, t, targets: dict, binding: dict):
        """Every T_exec: rate-limit the ADVISORY, queue it behind the dead time, log a snapshot."""
        self._t = t
        lim = self.cfg.cv.rate_kmh_per_5s / 3.6 * (self.cfg.ctrl.T_exec / 5.0)
        xt = np.nan if self.x_tail is None else self.x_tail
        for vid, v in targets.items():
            p = self.active.get(vid)
            if p is None or not p["comp"]:
                continue
            p["bind"] = binding.get(vid)
            if p["adv"] is None:
                p["adv"] = max(float(v), p["v_sel"])        # first advisory: never below the driven speed
            else:
                p["adv"] += float(np.clip(float(v) - p["adv"], -lim, lim))
            p["queue"].append((t + p["theta"], p["adv"]))
            self.snap.append((t, _idx(vid), p["x"], p["lane"], p["v"],
                              np.nan if p["ceil"] is None else p["ceil"], float(v),
                              BIND_CODE.get(p["bind"], -1), xt))

    def per_step(self, t, dt):
        self._t = t
        for vid, p in self.active.items():
            self.pacer_seconds_all += dt
            if not p["comp"]:
                continue
            self.pacer_seconds += dt
            self.win_pacer_s += dt
            if p["bind"] == "band":
                self.band_s += dt
                self.win_band_s += dt
                p["band_s"] += dt
            while p["queue"] and p["queue"][0][0] <= t + 1e-9:   # dead time elapsed -> apply directly
                _, ceil = p["queue"].pop(0)
                if ceil != p["ceil"]:
                    p["ceil"] = ceil
                    try:
                        self.k.vehicle.setMaxSpeed(vid, max(ceil, 1.0))
                    except Exception:
                        pass

    # ------------------------------------------------------------ sensing
    def sense(self, V, vid):
        i = V.idx.get(vid)
        if i is None:
            return None
        xp, lp, vp = float(V.x[i]), int(V.lane[i]), float(V.v[i])
        R = self.cfg.cv.sense_range
        m = (np.abs(V.x - xp) < R) & V.on_up
        m[i] = False
        other = m & (V.lane != lp)
        v_adj = float(V.v[other].mean()) if other.any() else None
        if v_adj is not None and self.cfg.cv.noise_kmh > 0:
            v_adj += float(self.rng.normal(0, self.cfg.cv.noise_kmh / 3.6))
        same = V.on_up & (V.lane == lp)
        beh = same & (V.x < xp) & (V.x > xp - 100.0)
        ahead = same & (V.x > xp) & (V.x < xp + 150.0)
        v_f = float(V.v[beh].mean()) if beh.any() else None
        v_l = None
        if ahead.any():
            j = np.where(ahead)[0][np.argmin(V.x[ahead])]
            v_l = float(V.v[j])
        return dict(x=xp, lane=lp, v=vp, v_adj=v_adj, v_f=v_f, v_l=v_l, comp=vid in self.comp_set)

    # ------------------------------------------------------------ leakage
    def leakage_check(self, t, V):
        """Called every 1 s: count vehicles overtaking compliant pacers in OTHER lanes; log mechanism samples."""
        self._t = t
        xnow = {V.ids[i]: float(V.x[i]) for i in range(len(V.ids))}
        lnow = {V.ids[i]: int(V.lane[i]) for i in range(len(V.ids))}
        if self._prev is not None:
            px, pl = self._prev
            for vid, p in self.active.items():
                if not p["comp"] or vid not in px or vid not in xnow:
                    continue
                xp0, xp1, lp0, lp1 = px[vid], xnow[vid], pl[vid], lnow[vid]
                i = V.idx[vid]
                win = (np.abs(V.x - xp1) < 120.0) & V.on_up
                fwd = bwd = lco_pass = 0
                for j in np.where(win)[0]:
                    u = V.ids[j]
                    if u == vid or u not in px or lnow[u] == lp1:      # other lanes only
                        continue
                    if px[u] < xp0 and xnow[u] > xp1:
                        fwd += 1
                        if pl.get(u) == lp0:                            # left the pacer lane to pass
                            lco_pass += 1
                    elif px[u] > xp0 and xnow[u] < xp1:
                        bwd += 1
                # lane-change-outs from the pacer lane (within 200 m behind)
                lco = 0
                for u, x0 in px.items():
                    if u == vid or pl.get(u) != lp0 or not (xp0 - 200.0 < x0 < xp0):
                        continue
                    if u in lnow and lnow[u] != lp0 and xnow.get(u, 1e9) < xp1 + 50:
                        lco += 1
                p["n_ot"] += fwd
                self.total_overtakes += fwd
                self.win_overtakes += fwd
                if p["bind"] == "band":
                    self.win_overtakes_band += fwd
                # moving-observer prediction over other lanes within +-250 m
                w = p["win"]
                pred, dvs, vadj = 0.0, [], []
                for ln in range(self.cfg.geo.n_up):
                    if ln == lp1:
                        continue
                    mj = V.on_up & (V.lane == ln) & (np.abs(V.x - xp1) < 250.0)
                    if mj.any():
                        kj = mj.sum() / 500.0
                        vj = float(V.v[mj].mean())
                        pred += kj * (vj - V.v[i])
                        dvs.append(vj - V.v[i])
                        vadj.append(vj)
                w["pred"] += pred           # veh per second, integrated over 1 s
                w["adj"] += fwd - bwd - lco_pass
                w["lco"] += lco
                w["n_ot"] += fwd
                w["v_p"].append(float(V.v[i]))
                if dvs:
                    w["dv"].append(float(np.mean(dvs)))
                    w["v_adj"].append(float(np.mean(vadj)))
                if t - w["t0"] >= 60.0:
                    v_p = float(np.mean(w["v_p"])) if w["v_p"] else float(V.v[i])
                    self.mech_log.append(dict(
                        t=t, pred=w["pred"], obs_net=w["adj"], lco=w["lco"],
                        dv=float(np.mean(w["dv"])) if w["dv"] else np.nan, n_ot=w["n_ot"],
                        v_adj=float(np.mean(w["v_adj"])) if w["v_adj"] else np.nan, v_p=v_p,
                        x=xp1, lane=lp1, queued=int(v_p < 40.0 / 3.6)))
                    p["win"] = dict(t0=t, pred=0.0, adj=0, lco=0, dv=[], v_adj=[], v_p=[], n_ot=0)
        self._prev = (xnow, lnow)

    def pop_window(self):
        """Overtakes per compliant BAND-LIMITED pacer-minute since the last call (band adaptation; not used in the paper).
        None if less than 30 s of band-limited pacer time was observed; the window counters are reset."""
        r = None
        if self.win_band_s >= 30.0:
            r = self.win_overtakes_band / (self.win_band_s / 60.0)
            self.r_win.append((self._t, r))
        self.win_overtakes, self.win_overtakes_band = 0, 0
        self.win_pacer_s, self.win_band_s = 0.0, 0.0
        return r

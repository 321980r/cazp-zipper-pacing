"""
Performance and diagnostic measures computed from the simulation state.

Efficiency : TTS, delay (from the scheduled departure = total; from network entry = network delay),
             insertion wait, peak discharge, free-capacity estimate, queue length (fill-ratio rule and
             40 km/h rule), N_pending, breakdown time / flag, speed penalty of un-queued restrictions
Stability  : stop-and-go counts with hysteresis, acceleration noise, in-queue / upstream speed variance,
             CVS (coefficient of variation of 30-s lane speeds in 5-min windows at the approach stations)
Safety     : hard braking onsets (all / near the tail), TET (1.5 s and 2 s) / TIT (gap incl. minGap), DRAC
             exposure, speed drop at the tail, collisions (windowed to x_lo <= x < x_drop, warm <= t <= warm+peak)
Mechanism  : lane changes by reason (from lanechange-output), overtakes of pacers, pacer burden (matched)
Control    : sign changes / shares / floors, usage saturation, measured compliance gain
Deployment : online tail vs grid tail (RMSE), pacer time upstream of the tail, sign coverage, mask flips,
             tail beyond the gantry array, zone clipping, gantries used
Recording  : 30-s time series, per-lane time-space grids, five loop detectors per lane, per-vehicle table,
             gantry crossings, event lists, optional trajectory window.

`Metrics` is pure Python/numpy (no SUMO calls) so it can be unit-tested with synthetic vehicle arrays:
V needs ids, idx, num, x, v, lane, acc, on_up, is_cv, length. Hooks: depart/arrive, step (every step),
sample (1 s), tick (5 s, after the control blocks), record (30 s), result (end).
"""
from __future__ import annotations
import numpy as np

HB = -3.0                 # hard-braking threshold (m/s^2)
TTC_STAR = 1.5            # TTC threshold (s) of TET / TIT
TTC_2 = 2.0               # secondary TET threshold (s)
DRAC_STAR = 3.35          # DRAC threshold (m/s^2)
Q_SPEED = 40 / 3.6        # 40 km/h queue rule and in-queue speed variance: below 40 km/h
K_CRIT = 30.0             # critical density (veh/km/lane) of the grid queue rule: 1.5 k_crit = 45 veh/km/lane is
#                           an equilibrium speed of 40-55 km/h (tau 1.0-1.4 s), consistent with the 0.5 v_ff speed
#                           rule; it is not derived from the occupancy set-point (a pre-breakdown occupancy
#                           of 0.08-0.10 would put the density threshold inside dense free flow)
V_STOP, V_GO = 5 / 3.6, 30 / 3.6   # stop-and-go hysteresis thresholds
STOP_STEPS = 4            # v < V_STOP for 4 consecutive steps (2 s at 0.5 s) = one stop
FILL = 0.70               # queue tail: >= 70 % of the cells between tail and head congested
HEAD_CELLS = 3            # head must lie in the last 3 cells before the drop (merge area)
Q_FULL = 1000.0           # "queue >= 1 km" of the breakdown definitions
BREAKDOWN_S = 900.0       # breakdown flag: t_b exists and >= 15 min of queue >= Q_FULL
TAIL_UP, TAIL_DOWN = 300.0, 100.0   # "near the tail": [tail - TAIL_UP, tail + TAIL_DOWN]
CVS_BINS, CVS_MIN = 10, 5           # 5-min windows of 30-s bins; >= 5 valid bins per window
LC_REASON = {"strategic": 1, "cooperative": 2, "speedGain": 3, "keepRight": 4, "sublane": 5, "traci": 6}
DET_NAMES = ["inflow", "app_mid", "release", "merge_entry", "discharge"]
TS_FIELDS = ("t", "b", "v_req", "g1", "g2", "vs_tilde", "u", "u_star", "w", "phi", "n_pacers", "n_pacers_comp",
             "occ", "v_zone", "v_app", "r_win", "sat_both", "event", "queue_m", "queue40_m", "n_pending",
             "n_in_zone", "a_mean_kmh", "tail_x", "queued", "n_masked", "v_adj_hat", "vs_floor", "g_ctrl")


def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _nanmean(a):
    a = np.asarray(a, dtype=float)
    return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float("nan")


def _share(m):
    return float(np.mean(m)) if len(m) else float("nan")


def lc_reason_code(reason: str) -> int:
    for key, code in LC_REASON.items():
        if key in reason:
            return code
    return 0


def lc_class(reason: str) -> str:
    """'mandatory' (strategic), 'discretionary' (speedGain / keepRight / cooperative) or 'other'."""
    if "strategic" in reason:
        return "mandatory"
    if any(k in reason for k in ("speedGain", "keepRight", "cooperative")):
        return "discretionary"
    return "other"


def det_positions(g, x_drop, x_out):
    """The five reporting loops: inflow, application-area station (x_drop - d_acc - L_app/2; midpoint of the
    two gantries for the fixed deployment), release gantry, merge entry, discharge."""
    if g.fixed:
        app = 0.5 * (g.gantries[0] + g.gantries[1])
    else:
        app = g.x_drop - g.d_acc - 0.5 * g.L_app
    return [float(g.inflow_x), float(app), float(g.release_x), float(x_drop - g.d_merge), float(x_out)]


def cross_positions(g):
    """Gantry-crossing loops: 200 m upstream of the array, 150 m after the application station, 150 m after the
    most downstream gantry (fixed deployment: G1 - 200, G1 + 150, G2 + 150)."""
    gx = g.gantries
    if g.fixed:
        return [gx[0] - 200.0, gx[0] + 150.0, gx[1] + 150.0]
    return [gx[0] - 200.0, g.x_drop - g.d_acc - 0.5 * g.L_app + 150.0, gx[-1] + 150.0]


# ---------------------------------------------------------------- queue rules (pure functions)
def queue_from_grid(v, k, x_drop, dx, v_ff, k_crit, fill=FILL, head_cells=HEAD_CELLS):
    """Queue rule on a (nt, nx) speed grid v (m/s) and pooled density grid k (veh/km/lane).

    A cell is congested if v < 0.5 v_ff or k > 1.5 k_crit. The head is the last congested cell within the
    last `head_cells` cells before the drop; the tail is the upstream-most congested cell such that at least
    `fill` of the cells between tail and head are congested (no distance cap).
    Returns (queue_m, tail_cell, head_cell) with tail/head = -1 when there is no queue."""
    nt, nx = v.shape
    xd = min(int(np.ceil(x_drop / dx)), nx)                  # cells 0..xd-1 are upstream (last one partial)
    with np.errstate(invalid="ignore"):
        cong = (np.isfinite(v) & (v < 0.5 * v_ff)) | (np.nan_to_num(k) > 1.5 * k_crit)
    queue = np.zeros(nt)
    tail = np.full(nt, -1, dtype=int)
    head = np.full(nt, -1, dtype=int)
    lo = max(0, xd - head_cells)
    for tb in range(nt):
        row = cong[tb, :xd]
        hs = np.where(row[lo:xd])[0]
        if not len(hs):
            continue
        h = lo + int(hs.max())
        n_c, tl = 0, h
        for c in range(h, -1, -1):
            if row[c]:
                n_c += 1
                if n_c >= fill * (h - c + 1):
                    tl = c
        head[tb], tail[tb] = h, tl
        queue[tb] = x_drop - tl * dx
    return queue, tail, head


def queue40_from_grid(v, x_drop, dx, cap_m=3000.0):
    """40 km/h rule: distance from the drop to the upstream-most cell (within cap_m) with v < 40 km/h."""
    nt, nx = v.shape
    xd = min(int(x_drop // dx), nx)
    x0 = max(0, xd - int(cap_m // dx))
    q = np.zeros(nt)
    with np.errstate(invalid="ignore"):
        slow = np.isfinite(v) & (v < Q_SPEED)
    for tb in range(nt):
        idx = np.where(slow[tb, x0:xd])[0]
        if len(idx):
            q[tb] = (xd - x0 - idx.min()) * dx
    return q


def breakdown_time(queue_m, dtb, q_on=300.0, q_full=Q_FULL, horizon_s=600.0):
    """Breakdown detector: first bin with queue >= q_on after which queue >= q_full is reached within horizon_s."""
    nb = int(horizon_s // dtb)
    for tb in range(len(queue_m)):
        if queue_m[tb] >= q_on and (queue_m[tb:tb + nb + 1] >= q_full).any():
            return float(tb * dtb)
    return None


def cvs_of(v_tm, tb0, tb1, stations=(1, 2, 3), n_bins=CVS_BINS, n_min=CVS_MIN):
    """Coefficient of variation of the 30-s lane speeds v_tm (n_det, n_lane, nt) in windows of n_bins bins
    inside [tb0, tb1) at the given stations, mean over lanes / windows / stations (NaN without a window)."""
    out = []
    for w0 in range(tb0, tb1 - n_bins + 1, n_bins):
        for d in stations:
            for ln in range(v_tm.shape[1]):
                x = np.asarray(v_tm[d, ln, w0:w0 + n_bins], dtype=float)
                x = x[np.isfinite(x)]
                if len(x) >= n_min and x.mean() > 0:
                    out.append(float(x.std() / x.mean()))
    return float(np.mean(out)) if out else float("nan")


# ---------------------------------------------------------------- metrics
class Metrics:
    def __init__(self, cfg, geom: dict, sched: dict, vtype: dict, min_gap: dict, comp_set=()):
        """geom: dict(x_end, x_drop, x_out, len_up, len_int, len_down, n_up, n_down, ends) from the loaded network;
        ends = x of every lane end (lane count of a grid cell = n_down + lanes ending downstream of it)."""
        self.cfg, self.geom = cfg, geom
        g, s, d = cfg.geo, cfg.sim, cfg.dem
        self.x_end, self.x_drop, self.x_out = geom["x_end"], geom["x_drop"], geom["x_out"]
        self.n_up, self.n_down = int(geom["n_up"]), int(geom["n_down"])
        self.drop = self.n_up - self.n_down
        self.v0, self.vb = g.v_base_kmh, g.v_base
        self.sched, self.vtype, self.min_gap = sched, vtype, dict(min_gap)
        self.dx, self.dtb = float(s.ts_dx), float(s.ts_dt)
        self.nx = int(np.ceil(self.x_end / self.dx))
        self.nt = int(np.ceil((d.t_end + s.t_clear_max) / self.dtb)) + 1
        self.t0_win, self.t1_win = d.warm, d.warm + d.peak            # safety / stability window (time)
        self.x_lo = max(g.zone_start - 500.0, g.x_min_pacer)          # ... and space: x_lo <= x < x_drop
        self.eval0, self.eval1 = d.warm + 600.0, d.warm + d.peak       # evaluation window (throughput, queue)
        # per-vehicle state (index = vehicle number)
        n = self.n_veh = len(sched)
        nan = np.full(n, np.nan)
        self.t_sched = np.array([sched.get(f"v{i}", np.nan) for i in range(n)], dtype=float)
        self.sched_sorted = np.sort(self.t_sched)
        self.t_dep, self.t_in, self.t_arr = nan.copy(), nan.copy(), nan.copy()
        self.f, self.c, self.tau = nan.copy(), nan.copy(), nan.copy()
        self.is_cv = np.array([vtype.get(f"v{i}", "").startswith("CV") for i in range(n)], dtype=bool)
        self.is_trk = np.array([vtype.get(f"v{i}", "") == "TRK" for i in range(n)], dtype=bool)
        self.comp = np.array([f"v{i}" in comp_set for i in range(n)], dtype=bool)
        self.n_stops = np.zeros(n, dtype=int)
        self.n_hb = np.zeros(n, dtype=int)
        self.acc_sum, self.acc_sq, self.acc_n = np.zeros(n), np.zeros(n), np.zeros(n)
        self.max_drac = np.zeros(n)
        self.pen = np.zeros(n)                # speed penalty (s): un-queued travel time lost to a reduced anchor
        self.low_cnt = np.zeros(n, dtype=int)
        self.armed = np.ones(n, dtype=bool)
        self.prev_x = nan.copy()
        self.acc_prev = np.zeros(n)
        self.n_dep = 0
        # grids
        nt, nx, nl = self.nt, self.nx, self.n_up
        self.vsum, self.cnt = np.zeros((nt, nx)), np.zeros((nt, nx))
        self.vsum_l, self.cnt_l = np.zeros((nt, nx, nl)), np.zeros((nt, nx, nl))
        self.steps_in_bin = np.zeros(nt)
        # detectors and gantry crossings
        self.gx = [float(x) for x in g.gantries]
        self.x_det = det_positions(g, self.x_drop, self.x_out)
        self.det_cnt = np.zeros((5, nl, nt))
        self.det_vsum = np.zeros((5, nl, nt))
        self.det_occ = np.zeros((5, nl, nt))
        self.x_cross = cross_positions(g)
        self.cross = []                       # (t, loop, lane, v, v_disp_kmh, c, is_cv, is_pacer, idx)
        # deployment bookkeeping (5-s ticks): t, x_tail, x_e, app0, app1, queued, zone_clipped, sign_on
        self.ticks = []
        self.g_used = np.zeros(len(self.gx), dtype=bool)
        # events / safety
        self.ev_hb, self.ev_tet, self.ev_drac, self.ev_coll = [], [], [], []
        self.tet = self.tet2 = self.tit = self.t_drac = 0.0
        self.svar = []                        # (t, var_upstream, var_in_queue)
        self.occ_bins, self.occ_n = np.zeros(nt), np.zeros(nt)
        # time series
        self.ts = {k: [] for k in TS_FIELDS}
        self._n_signlog = 0
        # trajectory window; x in uint16 units of 0.1 m -> at most 6.5 km
        self.traj_on = bool(getattr(s, "log_traj", False))
        self.traj_t0, self.traj_t1 = d.warm + 600.0, d.warm + 1320.0
        self.traj_x0, self.traj_x1 = max(g.zone_start - 200.0, self.x_drop - 6200.0), self.x_drop + 300.0
        self.traj_rows = []                   # (idx, t_units, x_units, lane, v_units, pacer)

    # ------------------------------------------------------------ vehicle life cycle
    def depart(self, vid, t, st: dict):
        i = int(vid[1:])
        self.t_dep[i] = t
        self.f[i], self.c[i], self.tau[i] = st.get("f", np.nan), st.get("c", np.nan), st.get("tau", np.nan)
        self.n_dep += 1

    def arrive(self, vid, t):
        i = int(vid[1:])
        self.t_arr[i] = t
        self.prev_x[i] = np.nan

    def n_pending(self, t) -> int:
        return int(np.searchsorted(self.sched_sorted, t, side="right")) - self.n_dep

    def in_window(self, t) -> bool:
        return self.t0_win <= t <= self.t1_win

    def _bin(self, t) -> int:
        """30-s bin that ends at (or contains) t."""
        return int(np.clip(int((t - 1e-6) // self.dtb), 0, self.nt - 1))

    # ------------------------------------------------------------ every step
    def step(self, t, V, leaders: dict, dr_state: dict, active: dict):
        tb = min(int(t // self.dtb), self.nt - 1)
        self.steps_in_bin[tb] += 1
        if not len(V.ids):
            return
        ii, x, v, ln, a = V.num, V.x, V.v, V.lane, V.acc
        new = np.isnan(self.t_in[ii])
        if new.any():
            self.t_in[ii[new]] = t
        # grids
        xb = np.clip((x // self.dx).astype(int), 0, self.nx - 1)
        lb = np.clip(ln, 0, self.n_up - 1)
        np.add.at(self.vsum[tb], xb, v)
        np.add.at(self.cnt[tb], xb, 1)
        np.add.at(self.vsum_l[tb], (xb, lb), v)
        np.add.at(self.cnt_l[tb], (xb, lb), 1)
        # windowed per-vehicle measures
        win = (x >= self.x_lo) & (x < self.x_drop) if self.in_window(t) else np.zeros(len(x), dtype=bool)
        hb = (a <= HB) & (self.acc_prev[ii] > HB) & win
        for j in np.where(hb)[0]:
            ld = leaders.get(V.ids[j])
            follows = int(bool(ld) and ld[0] in active and active[ld[0]]["comp"])
            self.ev_hb.append((t, x[j], ln[j], v[j], follows))
            self.n_hb[ii[j]] += 1
        self.acc_prev[ii] = a
        if win.any():
            w = ii[win]
            self.acc_sum[w] += a[win]
            self.acc_sq[w] += a[win] ** 2
            self.acc_n[w] += 1
        low = v < V_STOP
        lc = np.where(low, self.low_cnt[ii] + 1, 0)
        self.low_cnt[ii] = lc
        new_stop = win & low & (lc == STOP_STEPS) & self.armed[ii]
        if new_stop.any():
            self.n_stops[ii[new_stop]] += 1
            self.armed[ii[new_stop]] = False
        self.armed[ii[v > V_GO]] = True
        # crossings: detectors, gantry loops
        px = self.prev_x[ii]
        has = ~np.isnan(px)
        for j, X in enumerate(self.x_det):
            m = has & (px < X) & (x >= X)
            if m.any():
                np.add.at(self.det_cnt[j], (lb[m], tb), 1)
                np.add.at(self.det_vsum[j], (lb[m], tb), v[m])
                np.add.at(self.det_occ[j], (lb[m], tb), V.length[m] / np.maximum(v[m], 0.5))
        for j, X in enumerate(self.x_cross):
            m = has & (px < X) & (x >= X)
            for q in np.where(m)[0]:
                vid = V.ids[q]
                st = dr_state.get(vid, {})
                self.cross.append((t, j, int(ln[q]), float(v[q]), st.get("v_disp", np.nan) * 3.6,
                                   st.get("c", np.nan), int(V.is_cv[q]),
                                   int(vid in active and active[vid]["comp"]), int(ii[q])))
        self.prev_x[ii] = x

    def occupancy(self, t, occ):
        tb = min(int(t // self.dtb), self.nt - 1)
        self.occ_bins[tb] += occ
        self.occ_n[tb] += 1

    def collision(self, t, x, lane, collider, victim):
        self.ev_coll.append((t, x, lane, collider, victim))

    # ------------------------------------------------------------ every second
    def sample(self, t, V, leaders: dict, dr_state=None, app=None, est=None, dt_s: float = 1.0):
        """TET / TET2 / TIT / DRAC with gap = leader gap + minGap, windowed; every 5 s the speed
        variances; speed penalty: for every vehicle inside the application area `app` whose segment is
        un-queued (est.cong), dt_s * (1 - a / v_base) with a = its driver anchor (dr_state[vid]['a'])."""
        if self.in_window(t) and len(V.ids):
            xw = (V.x >= self.x_lo) & (V.x < self.x_drop)
            for i in np.where(xw)[0]:
                vid = V.ids[i]
                ld = leaders.get(vid)
                if not ld or ld[0] == "":
                    continue
                j = V.idx.get(ld[0])
                if j is None:
                    continue
                dv = V.v[i] - V.v[j]
                gap = ld[1] + self.min_gap.get(self.vtype.get(vid, ""), 2.0)
                if dv > 0.1 and gap > 0:
                    ttc = gap / dv
                    if ttc < TTC_2:
                        self.tet2 += dt_s
                    if ttc < TTC_STAR:
                        self.tet += dt_s
                        self.tit += (TTC_STAR - ttc) * dt_s
                        self.ev_tet.append((t, V.x[i], V.lane[i]))
                    drac = dv * dv / (2.0 * gap)
                    n = V.num[i]
                    if drac > self.max_drac[n]:
                        self.max_drac[n] = drac
                    if drac > DRAC_STAR:
                        self.t_drac += dt_s
                        self.ev_drac.append((t, V.x[i], V.lane[i]))
        if int(round(t)) % 5 == 0 and len(V.ids):
            m = V.x < self.x_drop
            if m.sum() > 5:
                vv = V.v[m]
                q = vv < Q_SPEED
                self.svar.append((t, float(np.var(vv)), float(np.var(vv[q])) if q.sum() > 5 else np.nan))
        if app is not None and dr_state and est is not None and len(V.ids):
            m = (V.x >= app[0]) & (V.x < app[1])
            if m.any():
                si = ((V.x[m] - est.x0[0]) // est.seg).astype(int)
                ok = (si >= 0) & (si < est.n)
                free = np.ones(int(m.sum()), dtype=bool)
                free[ok] = ~np.asarray(est.cong)[si[ok]]
                for j in np.where(m)[0][free]:
                    a = dr_state.get(V.ids[j], {}).get("a")
                    if a is not None and a < self.vb:
                        self.pen[V.num[j]] += dt_s * (1.0 - a / self.vb)

    # ------------------------------------------------------------ every 5 s (after the control blocks)
    def tick(self, t, ctrl, pm):
        """Deployment bookkeeping: online tail, control end, application area, zone clipping, sign state."""
        a = getattr(ctrl, "app_area", None)
        xt = getattr(ctrl, "x_tail", None)
        self.ticks.append((t, np.nan if xt is None else float(xt), float(getattr(ctrl, "x_e", np.nan)),
                           np.nan if a is None else float(a[0]), np.nan if a is None else float(a[1]),
                           int(bool(getattr(ctrl, "queued", False))),
                           int(bool(getattr(pm, "zone_clipped", False))),
                           int(float(getattr(ctrl, "g_ctrl", self.v0)) < self.v0 - 1e-6)))
        s = np.asarray(ctrl.signs, dtype=float)
        if len(s) == len(self.g_used):
            self.g_used |= s < self.v0 - 1e-6

    # ------------------------------------------------------------ every 2 s inside the showcase window
    def traj(self, t, V, active: dict):
        if not self.traj_on or not (self.traj_t0 <= t <= self.traj_t1) or not len(V.ids):
            return
        inx = (V.x >= self.traj_x0) & (V.x <= self.traj_x1)
        pac = np.zeros(len(V.ids), dtype=bool)
        for vid, p in active.items():
            if p["comp"] and vid in V.idx:
                pac[V.idx[vid]] = True
        near = np.zeros(len(V.ids), dtype=bool)
        for i in np.where(pac)[0]:
            near |= np.abs(V.x - V.x[i]) < 150.0
        keep = inx & (V.is_cv | near | (V.num % 3 == 0))
        for i in np.where(keep)[0]:
            self.traj_rows.append((int(V.num[i]), int(round((t - self.traj_t0) * 2)),
                                   int(round((V.x[i] - self.traj_x0) * 10)), int(V.lane[i]),
                                   int(round(V.v[i] * 36.0)), int(pac[i])))

    # ------------------------------------------------------------ every 30 s
    def record(self, t, ctrl, pm, dr, V, occ, v_zone, v_app, r_last):
        ts = self.ts
        u = getattr(ctrl, "u", np.nan)
        u = np.nan if u is None else u
        uses_sign = bool(ctrl.uses_sign)
        g_ctrl = float(ctrl.g_ctrl)
        signs = list(ctrl.signs)
        stair = getattr(ctrl, "stair_idx", None)
        if getattr(ctrl, "fixed", False):
            g1 = signs[0]
        else:
            g1 = signs[stair] if stair is not None and stair < len(signs) else np.nan
        v_floor = getattr(ctrl, "v_floor_kmh", np.nan) if uses_sign else np.nan
        v_adj = getattr(ctrl, "v_adj_hat", np.nan)
        v_adj = np.nan if v_adj is None else v_adj
        x_tail = getattr(ctrl, "x_tail", None)
        app = getattr(ctrl, "app_area", None)
        ts["t"].append(t)
        ts["b"].append(ctrl.b)
        ts["v_req"].append(ctrl.v_req * 3.6)
        ts["g1"].append(g1)
        ts["g2"].append(g_ctrl)
        ts["g_ctrl"].append(g_ctrl)
        ts["vs_tilde"].append(getattr(ctrl, "vs_tilde", np.nan) * 3.6)
        ts["u"].append(u)
        ts["u_star"].append(getattr(ctrl, "u_star", np.nan))
        ts["w"].append(getattr(ctrl, "w", np.nan) * 3.6)
        ts["phi"].append(pm.phi() if ctrl.uses_pacers else np.nan)
        ts["n_pacers"].append(len(pm.active))
        ts["n_pacers_comp"].append(sum(1 for p in pm.active.values() if p["comp"]))
        ts["occ"].append(occ)
        ts["v_zone"].append(np.nan if v_zone is None else v_zone * 3.6)
        ts["v_app"].append(np.nan if v_app is None else v_app * 3.6)
        ts["r_win"].append(np.nan if r_last is None else r_last)
        ts["sat_both"].append(float(np.isfinite(u) and u > 1.0 and g_ctrl <= v_floor + 1e-6)
                              if hasattr(ctrl, "u") and uses_sign else np.nan)
        n_log = len(ctrl.sign_log)
        ts["event"].append(float(sum(int(r[8]) for r in ctrl.sign_log[self._n_signlog:n_log])) if n_log else 0.0)
        self._n_signlog = n_log
        tb = self._bin(t)
        ts["queue_m"].append(tb)                       # bin index -> replaced by the grid value in result()
        ts["queue40_m"].append(tb)
        ts["n_pending"].append(self.n_pending(t))
        ts["tail_x"].append(np.nan if x_tail is None else float(x_tail))
        ts["queued"].append(float(bool(getattr(ctrl, "queued", False))))
        ts["n_masked"].append(float(getattr(ctrl, "n_masked", 0)))
        ts["v_adj_hat"].append(v_adj * 3.6 if np.isfinite(v_adj) else np.nan)
        ts["vs_floor"].append(v_floor)
        zone = getattr(pm, "zone", None)
        if len(V.ids) and zone is not None:
            ts["n_in_zone"].append(int(((V.x >= zone[0]) & (V.x < zone[1])).sum()))
        else:
            ts["n_in_zone"].append(0)
        if len(V.ids) and app is not None:
            m = (~V.is_cv) & (V.x >= app[0]) & (V.x < app[1])
            am = dr.mean_anchor([V.ids[i] for i in np.where(m)[0]])
            ts["a_mean_kmh"].append(np.nan if am is None else am * 3.6)
        else:
            ts["a_mean_kmh"].append(np.nan)

    # ------------------------------------------------------------ end of run
    def result(self, t_final, pm, ctrl, lc_rows, counters: dict) -> dict:
        """Assemble summary, ts, grid, det, flow, veh, cross, events, pacers, sign_log, mech, traj.
        counters: n_estop, n_tele, ot_band, band_s (band-window totals accumulated by the simulation)."""
        cfg, g, d, c = self.cfg, self.cfg.geo, self.cfg.dem, self.cfg.ctrl
        v0, vb = self.v0, self.vb
        nt = min(self.nt, int(t_final // self.dtb) + 1)
        n = max(self.n_veh, 1)
        dur = self.eval1 - self.eval0
        tb0, tb1 = int(self.eval0 // self.dtb), min(int(self.eval1 // self.dtb), nt)
        # ---- grids and densities
        steps = np.maximum(self.steps_in_bin[:nt], 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            vmean = self.vsum[:nt] / self.cnt[:nt]
            v_lane = self.vsum_l[:nt] / self.cnt_l[:nt]
        k_lane = self.cnt_l[:nt] / steps[:, None, None] / (self.dx / 1000.0)
        xc = (np.arange(self.nx) + 0.5) * self.dx
        ends = self.geom.get("ends") or [self.x_drop] * self.drop       # x of every lane end
        n_lanes_x = self.n_down + sum((xc < e + self.dx).astype(int) for e in ends)
        k_pool = self.cnt[:nt] / steps[:, None] / (self.dx / 1000.0) / n_lanes_x[None, :]
        xd = min(int(np.ceil(self.x_drop / self.dx)), self.nx)
        nwarm = max(1, min(nt, int(d.warm // self.dtb)))
        up_warm = vmean[:nwarm, :xd]
        v_ff = float(np.nanmedian(up_warm)) if np.isfinite(up_warm).any() else vb * cfg.drv.mu_f
        v_ff = max(v_ff, 0.5 * vb)
        k_crit = K_CRIT
        queue_m, tail, head = queue_from_grid(vmean, k_pool, self.x_drop, self.dx, v_ff, k_crit)
        queue40 = queue40_from_grid(vmean, self.x_drop, self.dx)
        t_bin_end = np.minimum((np.arange(nt) + 1) * self.dtb, t_final)
        dep_sorted = np.sort(self.t_dep[np.isfinite(self.t_dep)])
        n_pend = (np.searchsorted(self.sched_sorted, t_bin_end, side="right")
                  - np.searchsorted(dep_sorted, t_bin_end, side="right")).astype(float)
        n_q = np.zeros(nt)
        for tb in range(nt):
            if head[tb] >= 0:
                n_q[tb] = self.cnt[tb, tail[tb]:head[tb] + 1].sum() / steps[tb]
        queue_vehh = float(((n_q + n_pend) * self.dtb).sum() / 3600.0)
        t_b = breakdown_time(queue_m, self.dtb)
        breakdown = bool(t_b is not None and (queue_m >= Q_FULL).sum() * self.dtb >= BREAKDOWN_S)
        tail_grid = np.where(queue_m > 0, self.x_drop - queue_m, np.nan)     # grid tail position per bin
        # ---- per-vehicle
        done = np.isfinite(self.t_arr)
        n_done, n_trunc = int(done.sum()), int((~done).sum())
        ta = np.where(done, self.t_arr, t_final)
        tt = ta - self.t_sched
        f = np.where(np.isfinite(self.f), self.f, 1.0)
        vmax = np.where(self.is_trk, 25.0, 50.0)
        ff = self.x_end / np.minimum(f * vb, vmax)
        delay = np.maximum(tt - ff, 0.0)
        t_in = np.where(np.isfinite(self.t_in), self.t_in, np.where(np.isfinite(self.t_dep), self.t_dep, ta))
        delay_net = np.maximum(ta - t_in - ff, 0.0)
        ins_wait = np.where(np.isfinite(self.t_dep), self.t_dep, t_final) - self.t_sched
        peak_dep = (self.t_sched >= d.warm) & (self.t_sched < d.warm + d.peak)
        trunc = n_trunc > 0
        tts_lb, dly_lb, dmean_lb = float(tt.sum() / 3600), float(delay.sum() / 3600), float(delay.mean()) if n else np.nan
        with np.errstate(invalid="ignore"):
            acc_std = np.sqrt(np.maximum(self.acc_sq / np.maximum(self.acc_n, 1)
                                         - (self.acc_sum / np.maximum(self.acc_n, 1)) ** 2, 0.0))
        acc_std = np.where(self.acc_n >= 20, acc_std, np.nan)
        # ---- pacers
        wp = pm.was_pacer
        was = np.zeros(n, dtype=bool)
        t_sel = np.full(n, np.nan)
        ep = []
        for vid, e in wp.items():
            i = int(vid[1:])
            if 0 <= i < n:
                was[i] = True
                t_sel[i] = e.get("t_sel", np.nan)
            ep.append((i, e.get("t_sel", np.nan), e.get("x_sel", np.nan), e.get("lane_sel", -1),
                       e.get("t_rel", np.nan), int(e.get("comp", False)), e.get("theta", np.nan),
                       e.get("n_ot", 0), e.get("band_s", 0.0), e.get("x_tail_at_sel", np.nan)))
        episodes = np.array(ep, dtype=np.float32).reshape(-1, 10)
        comp_p = was & self.comp
        n_pc, n_pn = int(comp_p.sum()), int((was & ~self.comp).sum())
        extra = self._matched_burden(delay, comp_p, was)
        placebo = (float(delay[comp_p].mean() - delay[was & ~self.comp].mean())
                   if n_pc and n_pn else np.nan)
        duty_min = pm.pacer_seconds / 60.0 / n_pc if n_pc else np.nan
        snap = np.array(pm.snap, dtype=float).reshape(-1, 9)
        # ---- detectors and flows
        with np.errstate(invalid="ignore", divide="ignore"):
            v_tm = self.det_vsum[:, :, :nt] / self.det_cnt[:, :, :nt]
        det_cnt = self.det_cnt[:, :, :nt]
        occ_t = np.minimum(self.det_occ[:, :, :nt] / self.dtb, 1.0)
        out_cnt, in_cnt, me_cnt = det_cnt[4].sum(0), det_cnt[0].sum(0), det_cnt[3].sum(0)
        from core.demand import rate_vph
        rate = np.mean([rate_vph(cfg, t) for t in np.arange(self.eval0, self.eval1, self.dtb)]) if dur > 0 else np.nan
        q_free = self._free_capacity(out_cnt, queue_m, nwarm, nt)
        # ---- time series bookkeeping
        ts = {k: np.array(v, dtype=float) for k, v in self.ts.items()}
        if len(ts["t"]):
            ib = np.clip(ts["queue_m"].astype(int), 0, nt - 1)
            ts["queue_m"], ts["queue40_m"] = queue_m[ib], queue40[ib]
        ts = {k: _f32(v) for k, v in ts.items()}
        tw = (ts["t"] >= d.warm) & (ts["t"] <= d.warm + d.peak) if len(ts["t"]) else np.zeros(0, dtype=bool)
        g2w, bw = ts["g2"][tw], ts["b"][tw]
        active = bw < 0.999
        # ---- deployment verification
        dep = self._deployment(snap, queue_m, tail_grid, nt, bool(ctrl.uses_sign), bool(ctrl.uses_pacers))
        # ---- lane changes
        lc = self._lane_changes(lc_rows, snap)
        # ---- safety / collisions
        coll = np.array(self.ev_coll, dtype=np.float32).reshape(-1, 5)
        n_coll = int(len(coll))
        n_coll_zone = int(((coll[:, 1] >= self.x_lo) & (coll[:, 1] < self.x_drop)).sum()) if n_coll else 0
        sv = np.array(self.svar, dtype=float).reshape(-1, 3)
        hb = np.array(self.ev_hb, dtype=float).reshape(-1, 5)
        if len(hb):
            tg = tail_grid[np.clip(((hb[:, 0] - 1e-6) // self.dtb).astype(int), 0, nt - 1)]
            n_hb_tail = int((np.isfinite(tg) & (hb[:, 1] >= tg - TAIL_UP) & (hb[:, 1] <= tg + TAIL_DOWN)).sum())
        else:
            n_hb_tail = 0
        # ---- compliance gain from the gantry crossings
        cross = np.array(self.cross, dtype=float).reshape(-1, 9)
        g_meas = self._gain(cross, queue_m, nt)
        c_eff = g_meas / cfg.drv.mu_f if np.isfinite(g_meas) else np.nan
        sign_log = np.array(ctrl.sign_log, dtype=np.float32).reshape(-1, 13)
        uses_sign = bool(ctrl.uses_sign)
        model_based = hasattr(ctrl, "u")      # controllers with a usage signal (none here)
        pm_min = pm.pacer_seconds / 60.0
        summ = dict(
            n_veh=int(self.n_veh), n_done=n_done, n_trunc=n_trunc, t_clear_s=float(t_final - d.t_end), wall_s=np.nan,
            valid_insertion=bool(np.nanmax(ins_wait) <= 5.0) if n else True,
            valid_safety=n_coll == 0,
            TTS_vehh=np.nan if trunc else tts_lb, delay_vehh=np.nan if trunc else dly_lb,
            delay_mean_s=np.nan if trunc else dmean_lb,
            TTS_vehh_lb=tts_lb, delay_vehh_lb=dly_lb, delay_mean_s_lb=dmean_lb,
            network_delay_mean_s=float(delay_net[done].mean()) if n_done else np.nan,
            delay_peak_dep_mean_s=float(delay[peak_dep & done].mean()) if (peak_dep & done).any() else np.nan,
            insertion_wait_mean_s=_nanmean(ins_wait), insertion_wait_max_s=float(np.nanmax(ins_wait)) if n else np.nan,
            inflow_ratio_peak=float(in_cnt[tb0:tb1].sum() / dur * 3600.0 / rate) if dur > 0 and rate > 0 else np.nan,
            delivered_ratio_peak=float(((self.t_dep >= self.eval0) & (self.t_dep < self.eval1)).sum()
                                       / max(((self.t_sched >= self.eval0) & (self.t_sched < self.eval1)).sum(), 1)),
            discharge_vph=float(out_cnt[tb0:tb1].sum() / dur * 3600.0) if dur > 0 else np.nan,
            q_merge_entry_vph=float(me_cnt[tb0:tb1].sum() / dur * 3600.0) if dur > 0 else np.nan,
            q_free_cap_vph=q_free,
            queue_mean_m=float(queue_m[tb0:tb1].mean()) if tb1 > tb0 else np.nan, queue_max_m=float(queue_m.max()),
            queue40_mean_m=float(queue40[tb0:tb1].mean()) if tb1 > tb0 else np.nan, queue40_max_m=float(queue40.max()),
            queue_vehh=queue_vehh, n_pending_max=int(n_pend.max()) if nt else 0,
            tail_beyond_G1_share=_share((queue_m[tb0:tb1] > 0) & (self.x_drop - queue_m[tb0:tb1] < self.gx[0])),
            tail_at_entry_share=_share(queue_m[tb0:tb1] >= self.x_drop - 1e-6),
            t_b_s=np.nan if t_b is None else float(t_b), breakdown_flag=breakdown,
            tail_x_mean_m=dep["tail_mean"], tail_x_max_m=dep["tail_max"], tail_est_rmse_m=dep["rmse"],
            pacer_up_share=dep["pacer_up"], pacer_in_app_share=dep["pacer_in_app"],
            sign_cov_share=dep["sign_cov"], n_mask_changes=int(getattr(ctrl, "n_mask_changes", 0)),
            tail_beyond_array_share=dep["beyond"], zone_clipped_share=dep["clipped"],
            n_gantries_used=int(self.g_used.sum()),
            speed_penalty_s=float(self.pen.sum() / n),
            stops_per_veh=float(self.n_stops.sum() / n), share_veh_ge2_stops=float((self.n_stops >= 2).mean()),
            acc_noise_ms2=_nanmean(acc_std),
            speed_var_inqueue=_nanmean(sv[(sv[:, 0] >= d.warm) & (sv[:, 0] <= d.warm + d.peak), 2]) if len(sv) else np.nan,
            speed_var_upstream=_nanmean(sv[(sv[:, 0] >= d.warm) & (sv[:, 0] <= d.t_end), 1]) if len(sv) else np.nan,
            cvs_approach=cvs_of(v_tm, tb0, tb1) if tb1 > tb0 else np.nan,
            dv_tail_max=self._dv_tail(vmean, tail, tb0, tb1),
            TET_s_per_veh=self.tet / n, TET2_s_per_veh=self.tet2 / n, TIT_s_per_veh=self.tit / n,
            DRAC_exposure_s_per_veh=self.t_drac / n,
            hard_brake_per_kveh=float(self.n_hb.sum() / n * 1000.0), hard_brake_n=int(self.n_hb.sum()),
            hb_tail_per_kveh=float(n_hb_tail / n * 1000.0),
            n_collisions=n_coll, n_collisions_zone=n_coll_zone,
            n_emergency_stops=int(counters.get("n_estop", 0)), n_teleports=int(counters.get("n_tele", 0)),
            lc_all_per_kveh=lc["all"] / n * 1000.0, lc_mandatory_per_kveh=lc["mandatory"] / n * 1000.0,
            lc_discretionary_per_kveh=lc["discretionary"] / n * 1000.0,
            lc_into_drop_lane_per_kveh=lc["into_drop"] / n * 1000.0,
            lc_pacer_lane_per_pmin=(lc["pacer_lane"] / pm_min) if pm_min >= 1.0 else np.nan,
            lc_unsafe_gap_share=lc["unsafe_share"],
            overtakes_per_pmin=(pm.total_overtakes / pm_min) if pm.pacer_seconds > 60 else np.nan,
            overtakes_band_per_pmin=(counters.get("ot_band", 0) / (counters.get("band_s", 0.0) / 60.0))
            if counters.get("band_s", 0.0) >= 60 else np.nan,
            pacer_seconds_comp=float(pm.pacer_seconds), pacer_seconds_all=float(pm.pacer_seconds_all),
            n_pacers_comp=n_pc, n_pacers_noncomp=n_pn,
            pacer_extra_delay_s=np.nan if trunc else extra, pacer_extra_delay_s_lb=extra,
            pacer_burden_comp_vs_noncomp_s=placebo,
            pacer_burden_per_duty_min=(extra / duty_min) if (np.isfinite(extra) and duty_min and duty_min > 0) else np.nan,
            sign_changes=int(ctrl.n_sign_changes), max_sign_step_kmh=float(ctrl.max_sign_step_kmh),
            min_sign_kmh=float(ctrl.min_sign_kmh), max_sign_reduction_kmh=float(v0 - ctrl.min_sign_kmh),
            sign_below_base_share=_share(g2w < v0 - 1e-6) if uses_sign else np.nan,
            sign_at_min_share=_share(g2w <= c.vs_min_kmh + 1e-6) if uses_sign else np.nan,
            sat_both_share=_nanmean(ts["sat_both"][tw]) if model_based and uses_sign and tw.any() else np.nan,
            n_sign_updates=int(len(sign_log)), n_event_updates=int(sign_log[:, 8].sum()) if len(sign_log) else 0,
            D_req_active_kmh=float((v0 - ts["v_req"][tw][active]).mean()) if active.any() else np.nan,
            active_share=_share(active) if len(active) else np.nan,
            c_eff=c_eff, g_meas=g_meas,
            w_final_kmh=float(ctrl.w * 3.6) if hasattr(ctrl, "w") else np.nan,
            band_updates=int(getattr(ctrl, "n_band_updates", 0)),
        )
        summ = {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in summ.items()}
        veh = dict(
            idx=np.arange(self.n_veh, dtype=np.int32), t_sched=_f32(self.t_sched), t_dep=_f32(self.t_dep),
            t_in=_f32(self.t_in), t_arr=_f32(self.t_arr), f=_f32(self.f), c=_f32(self.c), tau=_f32(self.tau),
            is_cv=self.is_cv.astype(np.uint8), is_trk=self.is_trk.astype(np.uint8), was_pacer=was.astype(np.uint8),
            comp=self.comp.astype(np.uint8), t_sel=_f32(t_sel), n_stops=self.n_stops.astype(np.int16),
            acc_std=_f32(acc_std), delay_s=_f32(np.where(done, delay, np.nan)), tt_s=_f32(np.where(done, tt, np.nan)),
            ins_wait_s=_f32(np.where(np.isfinite(self.t_dep), ins_wait, np.nan)), max_drac=_f32(self.max_drac),
            n_hb=self.n_hb.astype(np.int16))
        grid = dict(v=_f32(vmean), v_lane=_f32(v_lane), k_lane=_f32(k_lane), queue_m=_f32(queue_m),
                    queue40_m=_f32(queue40), n_pending=_f32(n_pend), t_b=t_b, dx=self.dx, dt=self.dtb,
                    x_drop=float(self.x_drop), gantries=list(self.gx), release=float(g.release_x),
                    zone_start=float(g.zone_start), x_end=float(self.x_end), v_ff=v_ff, k_crit=k_crit,
                    deployment=str(g.deployment))
        det = dict(x=[float(x) for x in self.x_det], name=list(DET_NAMES), count=_f32(det_cnt), v_tm=_f32(v_tm),
                   occ_t=_f32(occ_t))
        flow = dict(t=np.arange(nt) * self.dtb, out=out_cnt * 3600.0 / self.dtb, inn=in_cnt * 3600.0 / self.dtb,
                    occ=self.occ_bins[:nt] / np.maximum(self.occ_n[:nt], 1))
        cross_d = dict(t=_f32(cross[:, 0]), loop=cross[:, 1].astype(np.uint8), lane=cross[:, 2].astype(np.int8),
                       v=_f32(cross[:, 3]), v_disp_kmh=_f32(cross[:, 4]), c=_f32(cross[:, 5]),
                       is_cv=cross[:, 6].astype(np.uint8), is_pacer=cross[:, 7].astype(np.uint8),
                       idx=cross[:, 8].astype(np.int32))
        events = dict(hb=np.array(self.ev_hb, dtype=np.float32).reshape(-1, 5),
                      tet=np.array(self.ev_tet, dtype=np.float32).reshape(-1, 3),
                      drac=np.array(self.ev_drac, dtype=np.float32).reshape(-1, 3),
                      lc=lc["events"], coll=coll)
        pacers = dict(episodes=episodes, snap=_f32(snap))
        out = dict(summary=summ, ts=ts, grid=grid, det=det, flow=flow, veh=veh, cross=cross_d, events=events,
                   pacers=pacers, sign_log=sign_log, mech=list(pm.mech_log))
        if self.traj_on:
            tr = np.array(self.traj_rows, dtype=np.int64).reshape(-1, 6)
            out["traj"] = dict(t0=self.traj_t0, t1=self.traj_t1, x0=self.traj_x0, x1=self.traj_x1,
                               idx=tr[:, 0].astype(np.uint16), t=tr[:, 1].astype(np.uint16),
                               x=np.clip(tr[:, 2], 0, 65535).astype(np.uint16), lane=tr[:, 3].astype(np.uint8),
                               v=np.clip(tr[:, 4], 0, 65535).astype(np.uint16), pacer=tr[:, 5].astype(np.uint8))
        return out

    # ------------------------------------------------------------ summary helpers
    def _deployment(self, snap, queue_m, tail_grid, nt, uses_sign, uses_pacers):
        """Deployment verification fields from the 5-s ticks and the pacer snapshots.
        tail_mean/max: online queue extent x_drop - x_tail (m; mean over the evaluation window with 0 when
        un-queued, max over the run); rmse: online vs grid tail where both exist; pacer_up: compliant
        pacer-seconds upstream of the grid tail; pacer_in_app: ... inside the application area; sign_cov:
        share of sign-on time (peak) with the application area entirely upstream of the grid tail; beyond:
        evaluation-window ticks with a queue and the control end at / upstream of the whole array; clipped:
        evaluation-window ticks with the pacer zone clipped at x_min_pacer."""
        nan = float("nan")
        out = dict(tail_mean=nan, tail_max=nan, rmse=nan, pacer_up=nan, pacer_in_app=nan, sign_cov=nan,
                   beyond=nan, clipped=nan)
        tk = np.array(self.ticks, dtype=float).reshape(-1, 8)
        d = self.cfg.dem
        if len(tk):
            t, xt, xe, a0, a1, queued, clipped, sign_on = tk.T
            tb = np.clip(((t - 1e-6) // self.dtb).astype(int), 0, nt - 1)
            ev = (t >= self.eval0) & (t <= self.eval1)
            ext = np.where(np.isfinite(xt), self.x_drop - xt, 0.0)
            out["tail_mean"] = float(ext[ev].mean()) if ev.any() else nan
            out["tail_max"] = float(ext.max())
            both = np.isfinite(xt) & np.isfinite(tail_grid[tb])
            if both.any():
                out["rmse"] = float(np.sqrt(np.mean((xt[both] - tail_grid[tb][both]) ** 2)))
            if ev.any():
                out["beyond"] = float(np.mean((queued[ev] > 0) & (np.isnan(a0[ev]) | (xe[ev] <= self.gx[0] + 1e-6))))
                out["clipped"] = float(np.mean(clipped[ev] > 0)) if uses_pacers else nan
            on = (sign_on > 0) & (t >= d.warm) & (t <= d.warm + d.peak)
            if uses_sign and on.any():
                tg = tail_grid[tb][on]
                out["sign_cov"] = float(np.mean(np.isnan(tg) | (np.isfinite(a1[on]) & (a1[on] <= tg + 1e-6))))
        if len(snap):
            ts_, xs = snap[:, 0], snap[:, 2]
            tb = np.clip(((ts_ - 1e-6) // self.dtb).astype(int), 0, nt - 1)
            tg = tail_grid[tb]
            out["pacer_up"] = float(np.mean(np.isnan(tg) | (xs < tg)))
            if len(tk):
                j = np.clip(np.searchsorted(tk[:, 0], ts_, side="right") - 1, 0, len(tk) - 1)
                a0, a1 = tk[j, 3], tk[j, 4]
                out["pacer_in_app"] = float(np.mean(np.isfinite(a0) & (xs >= a0) & (xs < a1)))
        return out

    def _free_capacity(self, out_cnt, queue_m, nwarm, nt, n_bins=10):
        """Free-capacity estimate: maximum 5-min (n_bins) trailing mean discharge flow over windows after the
        warm-up in which the queue stayed below Q_FULL (peak-flow estimator of q_pre); NaN without a window."""
        best = float("nan")
        for e in range(max(nwarm, n_bins), nt + 1):
            if (queue_m[e - n_bins:e] < Q_FULL).all():
                q = out_cnt[e - n_bins:e].sum() / (n_bins * self.dtb) * 3600.0
                best = q if not np.isfinite(best) or q > best else best
        return best

    def _dv_tail(self, vmean, tail, tb0, tb1):
        """Largest speed drop across the tail (km/h) in the evaluation window: mean speed of the 3 cells
        upstream of the tail cell minus the mean of the tail cell and the 2 cells behind it."""
        best = float("nan")
        for tb in range(tb0, min(tb1, len(tail))):
            tl = tail[tb]
            if tl >= 3:
                up, inq = vmean[tb, tl - 3:tl], vmean[tb, tl:tl + 3]
                if np.isfinite(up).any() and np.isfinite(inq).any():
                    dv = (np.nanmean(up) - np.nanmean(inq)) * 3.6
                    best = dv if not np.isfinite(best) or dv > best else best
        return best

    def _matched_burden(self, delay, comp_p, was):
        """Mean over compliant pacers of delay_p - mean delay of never-selected vehicles scheduled
        within +-60 s (all lanes, HV or CV)."""
        ctrl_m = ~was & np.isfinite(self.t_sched)
        if not comp_p.any() or not ctrl_m.any():
            return np.nan
        order = np.argsort(self.t_sched[ctrl_m])
        ts_c = self.t_sched[ctrl_m][order]
        cum = np.concatenate([[0.0], np.cumsum(delay[ctrl_m][order])])
        diffs = []
        for i in np.where(comp_p)[0]:
            a = np.searchsorted(ts_c, self.t_sched[i] - 60.0, side="left")
            b = np.searchsorted(ts_c, self.t_sched[i] + 60.0, side="right")
            if b > a:
                diffs.append(delay[i] - (cum[b] - cum[a]) / (b - a))
        return float(np.mean(diffs)) if diffs else np.nan

    def _lane_changes(self, lc_rows, snap):
        """Classify lanechange-output rows (t, x, from_lane, to_lane, reason, unsafe, on_up)."""
        out = dict(all=0, mandatory=0, discretionary=0, into_drop=0, pacer_lane=0, unsafe_share=np.nan,
                   events=np.zeros((0, 5), dtype=np.float32))
        rows = [r for r in lc_rows if r[6] and self.x_lo < r[1] < self.x_drop]
        if not rows:
            return out
        if len(snap):
            snap = snap[np.argsort(snap[:, 0])]
        ev, unsafe = [], 0
        for t, x, fr, to, reason, uns, _ in rows:
            cls = lc_class(reason)
            out["all"] += 1
            unsafe += int(uns)
            if cls == "mandatory":
                out["mandatory"] += 1
            elif cls == "discretionary":
                out["discretionary"] += 1
                if len(snap):
                    a = np.searchsorted(snap[:, 0], t - 2.5)
                    b = np.searchsorted(snap[:, 0], t + 2.5)
                    s = snap[a:b]
                    if len(s) and ((np.abs(s[:, 2] - x) <= 200.0) & ((s[:, 3] == fr) | (s[:, 3] == to))).any():
                        out["pacer_lane"] += 1
            if to < self.drop:
                out["into_drop"] += 1
            ev.append((t, x, fr, to, lc_reason_code(reason)))
        out["unsafe_share"] = unsafe / out["all"]
        out["events"] = np.array(ev, dtype=np.float32).reshape(-1, 5)
        return out

    def _gain(self, cross, queue_m, nt):
        """Measured sign gain g: OLS slope of the speed 150 m after the most downstream gantry (km/h) on the
        displayed value, over non-pacer crossings while the loop is outside the queue; NaN with < 2 posted
        values (>= 20 crossings each, >= 20 km/h spread)."""
        if not len(cross):
            return np.nan
        m = (cross[:, 1] == 2) & (cross[:, 7] == 0) & np.isfinite(cross[:, 4])
        if not m.any():
            return np.nan
        t, v, vd = cross[m, 0], cross[m, 3] * 3.6, cross[m, 4]
        tb = np.clip((t // self.dtb).astype(int), 0, nt - 1)
        tail_x = self.x_drop - queue_m[tb]
        free = (queue_m[tb] <= 0) | (tail_x > self.x_cross[2] + 100.0)
        v, vd = v[free], vd[free]
        lv, nl = np.unique(np.round(vd), return_counts=True)
        lv = lv[nl >= 20]                                   # posted values with enough un-queued crossings
        if len(lv) < 2 or lv.max() - lv.min() < 20.0:       # need a >= 20 km/h spread of posted values
            return np.nan
        keep = np.isin(np.round(vd), lv)
        return float(np.polyfit(vd[keep], v[keep], 1)[0])

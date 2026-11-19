"""
One simulation run for one RunConfig. Returns the result dict (cfg, summary, ts, grid, det, flow, veh, cross, events,
pacers, sign_log, mech, [traj], meta, emis); runs.compact reduces it to the stored run record.

Call protocol: the tail estimator is fed at the 1-s clock; the feedback runs every reg_period (with the current
measurements M, so that VSL-S sees v_app); every T_exec: estimate -> ctrl.set_tail -> v_app, then for CV speed pacing
select -> sense -> exec_targets -> set_targets; the overtaking counts of the CVs are collected every T_band. The
optional on_step hook (policy.Policy) is called at every simulation step. Seeds: one SeedSequence split into demand,
drivers, CV compliance and sensing noise.
"""
from __future__ import annotations
import os
import shutil
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as C                                   # noqa: E402
import sumo_env                                      # noqa: E402
from core.demand import write_routes                 # noqa: E402
from core.driver_response import DriverResponse      # noqa: E402
from core.pacers import PacerManager                 # noqa: E402
from core.controllers import make_controller, reg_period   # noqa: E402
from core.metrics import Metrics                     # noqa: E402
from core.tail import TailEstimator                  # noqa: E402
from network.build_network import ensure_network     # noqa: E402

VAR_LANE_ID, VAR_LANEPOS, VAR_SPEED, VAR_ACC, VAR_LEADER = 0x51, 0x56, 0x40, 0x72, 0x68
VTYPES = ["HV_0", "HV_1", "HV_2", "CV_0", "CV_1", "CV_2", "TRK"]


class VehArrays:
    """Per-step vehicle arrays. num = vehicle number (int(vid[1:])); lane = global lane index
    (0 = rightmost upstream lane, downstream lanes offset by n_up - n_down)."""
    __slots__ = ("ids", "idx", "num", "x", "v", "lane", "acc", "is_cv", "on_up", "length")


def _lane_map(k, chain, drop):
    """Mainline geometry of the loaded network; chain = ((edge, junction at its end), ...) upstream -> downstream.
    Returns lane id -> (x offset, global lane index, on_up), x of the final lane drop (start of its junction), the
    internal length of that junction, the length of the last edge, and the x of every lane end (one entry per ending
    lane). on_up = upstream of the final lane drop. Lanes upstream of it keep their own numbers (lanes continue
    straight on through an intermediate lane end); the final junction and the last edge are offset by drop, as in
    the single-drop model."""
    ids = k.lane.getIDList()
    lane_map, ends = {}, []
    x, x_drop, len_int_drop = 0.0, None, 0.0
    n_edges = len(chain)
    for j, (edge, node) in enumerate(chain):
        down = j == n_edges - 1
        lanes = [lid for lid in ids if lid.startswith(edge + "_")]
        for lid in lanes:
            ln = int(lid.rsplit("_", 1)[1])
            lane_map[lid] = (x, ln + drop if down else ln, not down)
        length = k.lane.getLength(edge + "_0")
        x += length
        if down:
            break
        final = j == n_edges - 2
        ints = [lid for lid in ids if lid.startswith(":" + node + "_")]
        len_int = max([k.lane.getLength(lid) for lid in ints] + [0.0])
        for lid in ints:
            ln = int(lid.rsplit("_", 1)[1])
            lane_map[lid] = (x, ln + drop if final else ln, not final)
        n_next = sum(1 for lid in ids if lid.startswith(chain[j + 1][0] + "_"))
        ends += [x] * (len(lanes) - n_next)
        if final:
            x_drop, len_int_drop = x, len_int
        x += len_int
    return lane_map, x_drop, len_int_drop, length, ends


def _parse_lc(path, lane_map):
    """lanechange-output -> rows (t, x, from_lane, to_lane, reason, unsafe, from_on_up)."""
    rows = []
    if not os.path.isfile(path):
        return rows

    def num(el, a):
        v = el.get(a)
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    try:
        for _, el in ET.iterparse(path):
            if el.tag != "change":
                continue
            mf, mt = lane_map.get(el.get("from", "")), lane_map.get(el.get("to", ""))
            if mf is not None and mt is not None:
                with np.errstate(invalid="ignore"):
                    unsafe = bool(num(el, "leaderGap") < num(el, "leaderSecureGap")
                                  or num(el, "followerGap") < num(el, "followerSecureGap"))
                rows.append((num(el, "time"), mf[0] + num(el, "pos"), mf[1], mt[1], el.get("reason", ""),
                             int(unsafe), bool(mf[2])))
            el.clear()
    except ET.ParseError as e:                      # truncated file after a crash: keep what was read
        print(f"[sim] lanechange-output parse error: {e}")
    return rows


def _parse_trip(path):
    """tripinfo-output with the emissions device -> per-vehicle CO2, fuel, NOx (mg, as written by SUMO)."""
    ids, co2, fuel, nox = [], [], [], []
    if os.path.isfile(path):
        try:
            for _, el in ET.iterparse(path):
                if el.tag != "tripinfo":
                    continue
                e = el.find("emissions")
                if e is not None:
                    try:
                        ids.append(int(el.get("id")[1:]))
                        co2.append(float(e.get("CO2_abs")))
                        fuel.append(float(e.get("fuel_abs")))
                        nox.append(float(e.get("NOx_abs")))
                    except (TypeError, ValueError):
                        pass
                el.clear()
        except ET.ParseError as ex:
            print(f"[sim] tripinfo parse error: {ex}")
    return dict(idx=np.array(ids, np.int32), co2_mg=np.array(co2, np.float32), fuel_mg=np.array(fuel, np.float32),
                nox_mg=np.array(nox, np.float32))


def seg_speed(est, area):
    """Mean 30-s trailing segment speed (m/s) over the estimator segments fully inside area = (x0, x1);
    None when area is None or no segment fits (v_app of the application area, v_zone of the pacer zone)."""
    if area is None:
        return None
    m = (est.x0 >= area[0] - 1e-6) & (est.x1 <= area[1] + 1e-6)
    return float(est.seg_v[m].mean()) if m.any() else None


def run(cfg, verbose: bool = False, gui: bool = False, gui_delay: int = 50, on_step=None, extra_args=None,
        gui_settings=None) -> dict:
    sumo_env.setup(C.USER_SUMO_HOME, prefer_local_tools=gui)
    k, is_lib = sumo_env.import_kernel(prefer_libsumo=not gui)
    ss = np.random.SeedSequence(cfg.sim.seed)
    rng_dem, rng_drv, rng_pac, rng_noise = [np.random.default_rng(s) for s in ss.spawn(4)]
    ctrl = make_controller(cfg)                      # first: FIXZONE strategies set geo.deployment in place
    g, c, d = cfg.geo, cfg.ctrl, cfg.dem
    dt = cfg.sim.step
    tmp = tempfile.mkdtemp(prefix="cazp_")
    lc_path = os.path.join(tmp, "lc.xml")
    trip_path = os.path.join(tmp, "trip.xml")
    emis = {}
    started = False
    t_wall = time.time()
    try:
        routes = write_routes(cfg, os.path.join(tmp, "routes.rou.xml"), rng_dem, rng_pac)
        sched = {vid: t for vid, t, _, _ in routes}
        vtype = {vid: typ for vid, _, typ, _ in routes}
        comp_set = {vid for vid, _, _, comp in routes if comp}
        typ_cv = np.array([typ.startswith("CV") for _, _, typ, _ in routes], dtype=bool)
        typ_len = np.array([12.0 if typ == "TRK" else 4.8 for _, _, typ, _ in routes])
        net = ensure_network(g)
        binname = "sumo-gui" if gui else "sumo"
        cmd = [sumo_env.binary(binname, C.USER_SUMO_HOME) if not is_lib else "sumo",
               "-n", net, "-r", os.path.join(tmp, "routes.rou.xml"),
               "--step-length", str(dt), "--seed", str(cfg.sim.seed),
               "--time-to-teleport", "-1", "--collision.action", "warn", "--collision.mingap-factor", "0",
               "--lanechange-output", lc_path,
               "--device.emissions.probability", "1", "--tripinfo-output", trip_path,
               "--no-step-log", "true", "--no-warnings", "true", "--duration-log.disable", "true"]
        if not gui:
            cmd += ["--extrapolate-departpos", "true"]
        if cfg.drv.lc_duration > 0:
            cmd += ["--lanechange.duration", str(cfg.drv.lc_duration)]
        if gui:
            cmd += ["--delay", str(gui_delay), "--start", "true"]
            if gui_settings:
                cmd += ["--gui-settings-file", gui_settings]
        if extra_args:
            cmd += list(extra_args)
        if getattr(cfg.sim, "overtake_right", False):
            cmd += ["--lanechange.overtake-right", "true"]
        if is_lib:
            k.start(cmd)
        else:
            k.start(cmd, label=os.path.basename(tmp))
        started = True

        # ---- geometry from the loaded network (sanity checks)
        drop = g.n_up - g.n_down
        lane_map, len_up, len_int, len_down, ends = _lane_map(k, g.chain, drop)
        assert len(ends) == drop, f"{len(ends)} lane ends in the network, {drop} expected"
        assert abs(len_up - g.x_drop) < 10, f"final lane drop at {len_up} != x_drop {g.x_drop}"
        assert abs(len_down - g.L_down) < 10, f"network E_down length {len_down} != L_down {g.L_down}"
        v_lane = k.lane.getMaxSpeed("E_up_0")
        assert abs(v_lane - g.v_base) < 0.1, f"lane speed {v_lane} != v_base {g.v_base}"
        assert g.gantries[0] >= g.x_min_pacer - 1e-6 and g.release_x < len_up, \
            f"control layout {g.gantries[0]:.0f}..{g.release_x:.0f} outside the network (0..{len_up:.0f})"
        assert g.segments[0][0] >= 0.0 and g.segments[-1][1] <= g.x_drop + 1e-6, "segment detectors off the road"
        x_drop = len_up
        x_end = len_up + len_int + len_down
        x_out = len_up + len_int + g.outflow_x
        area = (g.merge_area[0], min(g.merge_area[1], len_up))
        w_area = area[1] - area[0]
        n_area = g.n_down + sum(min(max(e - area[0], 0.0), w_area) for e in ends) / w_area   # mean lane count
        min_gap = {}
        for tid in VTYPES:
            try:
                min_gap[tid] = k.vehicletype.getMinGap(tid)
            except Exception:
                min_gap[tid] = 2.0
        geom = dict(x_end=x_end, x_drop=x_drop, x_out=x_out, len_up=len_up, len_int=len_int, len_down=len_down,
                    n_up=g.n_up, n_down=g.n_down, ends=list(ends))

        dr = DriverResponse(cfg, rng_drv, k)
        pm = PacerManager(cfg, rng_noise, k, comp_set, len_up)
        met = Metrics(cfg, geom, sched, vtype, min_gap, comp_set)
        est = TailEstimator(cfg)
        T_reg = reg_period(cfg)
        nxt = dict(reg=T_reg, exec=c.T_exec, band=c.T_band, sec=1.0, rec=cfg.sim.ts_dt, traj=0.0)
        occ_acc, occ_n = 0.0, 0
        last_occ, v_zone, v_app = 0.0, g.v_base, None
        r_last = None
        M = dict(sensed={}, phi=0.0, v_zone=g.v_base, v_app=None, x_tail=None, queued=False, f_bar=cfg.drv.mu_f)
        t_end = d.t_end
        seen_coll = set()
        counters = dict(n_estop=0, n_tele=0, ot_band=0, band_s=0.0)
        t = 0.0

        while True:
            k.simulationStep()
            t = k.simulation.getTime()
            for vid in k.simulation.getDepartedIDList():
                k.vehicle.subscribe(vid, [VAR_LANE_ID, VAR_LANEPOS, VAR_SPEED, VAR_ACC])
                k.vehicle.subscribeLeader(vid, 150.0)
                dr.on_depart(vid)
                met.depart(vid, t, dr.state[vid])
            for vid in k.simulation.getArrivedIDList():
                dr.on_arrive(vid)
                met.arrive(vid, t)
                if vid in pm.active:
                    pm.release(vid)
            # ---- safety counters
            for col in k.simulation.getCollisions():
                pair = (col.collider, col.victim)
                if pair not in seen_coll:
                    seen_coll.add(pair)
                    m = lane_map.get(col.lane)
                    met.collision(t, (m[0] + col.pos) if m else np.nan, m[1] if m else -1,
                                  int(col.collider[1:]) if col.collider[1:].isdigit() else -1,
                                  int(col.victim[1:]) if col.victim[1:].isdigit() else -1)
            counters["n_estop"] += k.simulation.getEmergencyStoppingVehiclesNumber()
            counters["n_tele"] += k.simulation.getStartingTeleportNumber()
            res = k.vehicle.getAllSubscriptionResults()

            # ---- vehicle arrays
            V = VehArrays()
            ids, nums, xs, vs, ln, ac, up = [], [], [], [], [], [], []
            leaders = {}
            for vid, r in res.items():
                m = lane_map.get(r.get(VAR_LANE_ID))
                if m is None:
                    continue
                ids.append(vid)
                nums.append(int(vid[1:]))
                xs.append(m[0] + r[VAR_LANEPOS])
                vs.append(r[VAR_SPEED])
                ln.append(m[1])
                ac.append(r[VAR_ACC])
                up.append(m[2])
                if VAR_LEADER in r and r[VAR_LEADER]:
                    leaders[vid] = r[VAR_LEADER]
            V.ids = ids
            V.idx = {v: i for i, v in enumerate(ids)}
            V.num = np.array(nums, dtype=int)
            V.x, V.v = np.array(xs), np.array(vs)
            V.lane, V.acc = np.array(ln, dtype=int), np.array(ac)
            V.on_up = np.array(up, dtype=bool)
            V.is_cv = typ_cv[V.num] if len(ids) else np.zeros(0, dtype=bool)
            V.length = typ_len[V.num] if len(ids) else np.zeros(0)

            # ---- drivers react to what they see at the gantries
            dr.step(t, dt, dict(zip(ids, xs)), ctrl.signs_ms())

            # ---- merge-area space occupancy + per-step metrics
            if len(ids):
                ma = V.on_up & (V.x >= area[0]) & (V.x <= area[1])
                occ = float(V.length[ma].sum()) / ((area[1] - area[0]) * n_area)
            else:
                occ = 0.0
            occ_acc += occ
            occ_n += 1
            met.occupancy(t, occ)
            met.step(t, V, leaders, dr.state, pm.active)
            if met.traj_on and t >= nxt["traj"] - 1e-9:
                met.traj(t, V, pm.active)
                nxt["traj"] = t + 2.0

            # ---- pacers: per-step execution of the advisories
            if ctrl.uses_pacers:
                pm.per_step(t, dt)

            # ---- 1-s clock: segment detectors (tail estimator), surrogate safety samples, overtakes of CVs
            if t >= nxt["sec"] - 1e-9:
                est.update_1s(t, V.x, V.v, V.on_up)
                met.sample(t, V, leaders, dr.state, ctrl.app_area, est)
                if ctrl.uses_pacers:
                    pm.leakage_check(t, V)
                nxt["sec"] += 1.0

            # ---- feedback (30 s); M carries v_app for the credibility rule of VSL-S
            if t >= nxt["reg"] - 1e-9:
                last_occ = occ_acc / max(occ_n, 1)
                occ_acc, occ_n = 0.0, 0
                ctrl.regulate(t, last_occ, M)
                nxt["reg"] += T_reg

            # ---- 5-s block: tail -> application area -> CV advisories
            if t >= nxt["exec"] - 1e-9:
                e = est.estimate(t)
                ctrl.set_tail(t, e["x_tail"], e["queued"])
                v_app = seg_speed(est, ctrl.app_area)
                sensed = {}
                if ctrl.uses_pacers:
                    pm.select(t, V, ctrl.pacers_on(), e["x_tail"], ctrl.v_req)
                    for vid, p in pm.active.items():
                        if p["comp"]:
                            s = pm.sense(V, vid)
                            if s is not None:
                                sensed[vid] = s
                vz = seg_speed(est, pm.zone)
                v_zone = g.v_base if vz is None else vz
                M.update(sensed=sensed, phi=pm.phi() if ctrl.uses_pacers else 0.0, v_zone=v_zone, v_app=v_app,
                         x_tail=e["x_tail"], queued=e["queued"], f_bar=dr.mean_f())
                if ctrl.uses_pacers:
                    targets = ctrl.exec_targets(t, M, sensed) if ctrl.pacers_on() else {}
                    pm.set_targets(t, targets, ctrl.binding if ctrl.pacers_on() else {})
                met.tick(t, ctrl, pm)
                nxt["exec"] += c.T_exec

            # ---- overtaking counts of the CVs (300-s windows)
            if ctrl.uses_pacers and t >= nxt["band"] - 1e-9:
                counters["ot_band"] += pm.win_overtakes_band
                counters["band_s"] += pm.win_band_s
                r = pm.pop_window()
                r_last = r if r is not None else r_last
                nxt["band"] += c.T_band

            if t >= nxt["rec"] - 1e-9:
                met.record(t, ctrl, pm, dr, V, last_occ, v_zone, v_app, r_last)
                nxt["rec"] += cfg.sim.ts_dt

            if on_step is not None:
                on_step(t, k, ctrl, pm, V)

            if t >= t_end and k.simulation.getMinExpectedNumber() == 0:
                break
            if t >= t_end + cfg.sim.t_clear_max:
                break

        counters["ot_band"] += pm.win_overtakes_band
        counters["band_s"] += pm.win_band_s
        pm.release_all()
        k.close()
        started = False
        lc_rows = _parse_lc(lc_path, lane_map)          # written on close, parsed before rmtree
        emis = _parse_trip(trip_path)
    finally:
        if started:
            try:
                k.close()
            except Exception:
                pass
        shutil.rmtree(tmp, ignore_errors=True)

    out = met.result(t, pm, ctrl, lc_rows, counters)
    out["emis"] = emis
    summ = out["summary"]
    summ["wall_s"] = time.time() - t_wall
    if not summ["valid_insertion"]:
        print(f"[sim] WARNING insertion wait max {summ['insertion_wait_max_s']:.1f} s > 5 s "
              f"({c.strategy}, level {d.level:.2f}, seed {cfg.sim.seed}): demand not delivered")
    if summ["n_trunc"] > 0:
        print(f"[sim] WARNING {summ['n_trunc']} vehicle(s) not finished at t={t:.0f} s "
              f"({c.strategy}, seed {cfg.sim.seed}); TTS/delay set to NaN (lower bounds in *_lb)")
    if summ["n_collisions"] > 0:
        print(f"[sim] WARNING {summ['n_collisions']} collision(s) ({c.strategy}, seed {cfg.sim.seed})")
    if verbose:
        print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in summ.items()})
    out["cfg"] = cfg.to_dict()
    out["meta"] = dict(model_version=C.MODEL_VERSION, len_up=len_up, len_int=len_int, len_down=len_down,
                       x_end=x_end, x_out=x_out, dtb=met.dtb, dx=met.dx, n_up=g.n_up, n_down=g.n_down,
                       warm=d.warm, peak=d.peak, cool=d.cool, t_end=t_end, t_final=t,
                       eval0=met.eval0, eval1=met.eval1)
    return out


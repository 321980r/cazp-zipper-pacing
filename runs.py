"""
Run specification, execution and storage of the paired simulation runs.

A run is specified by a dict (S): closure configuration ("layout": seq, base or l43), peak demand, warm-up and
cool-down flows, random seed, CV penetration, the policy parameters (CAZP, VSL-L, zipper-merge sign, CV early merge;
policy.Policy) and the parameters of the simulator controller (VSL-S, CV speed pacing; sim/core/controllers.py). The
run key is an MD5 hash of these fields; a run is stored once as results/runs/<key>.pkl (CAZP_RUNS overrides the
folder) and is skipped when the file exists, so that interrupted blocks resume. Paired comparisons use the same seed:
the seed fixes the demand, the vehicle attributes and the CV and acceptance assignment.

Experiment blocks are defined in experiments.py.
"""
from __future__ import annotations
import ctypes
import hashlib
import json
import math
import os
import pickle
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "sim")
RES = os.path.join(HERE, "results")
RUNS = os.environ.get("CAZP_RUNS", os.path.join(RES, "runs"))
LOGS = os.path.join(HERE, "logs")
sys.path.insert(0, SIM)
sys.path.insert(0, HERE)
from closures import CLOSURES          # noqa: E402

X_DROP = 5000.0                      # final lane end of every configuration (m)
BED = dict(cf_model="EIDM", accel_scale=0.7, lc_duration=0.0, action_step=0.5, lcCooperative=1.0)
NC = dict(pol="NC")                  # no control
EM = dict(pol="EM", em=True)         # CV early merge
ZMSD = dict(pol="NC", zms=0.5, zms_dyn=2)   # zipper-merge sign shown during backups, followed by 50 % of the drivers
L3 = (3400, 3600, 3800)              # peak demands of configurations I and II (veh/h)
E = "e"                              # tag of every run specification


def design() -> dict:
    """Parameters fixed before the experiments: CAZP design, VSL-S and CV speed pacing settings, design-study values
    used by the tuning (results/design_parameters.json)."""
    return json.load(open(os.path.join(RES, "design_parameters.json"), encoding="utf-8"))


def S(block, arm, layout, dem, seed, pol, ctrl=None, warm=3000, cool=2000, pen=0.10, tag=E, drv=None, sim=None):
    """Run specification. pol: policy parameters; ctrl: simulator controller; drv / sim: modified test bed."""
    s = dict(block=block, arm=arm, layout=layout, dem=int(dem), warm=int(warm), cool=int(cool), seed=int(seed),
             pen=float(pen), pol=dict(pol), ctrl=dict(ctrl or {}), tag=tag)
    if drv:
        s["drv"] = dict(drv)
    if sim:
        s["sim"] = dict(sim)
    return s


def key(s):
    """Run key: block and arm names are labels and do not enter the key."""
    k = dict(layout=s["layout"], dem=s["dem"], warm=s["warm"], cool=s["cool"], seed=s["seed"], pen=s["pen"],
             pol=s["pol"], ctrl=s["ctrl"], tag=s["tag"])
    for f in ("drv", "sim"):
        if s.get(f):
            k[f] = s[f]
    return hashlib.md5(json.dumps(k, sort_keys=True).encode()).hexdigest()[:16]


def path(s):
    return os.path.join(RUNS, key(s) + ".pkl")


# ------------------------------------------------------------------ one run
def build(s):
    """RunConfig of a specification: geometry of the closure configuration, test-bed drivers (Table 1), demand profile
    (warm-up 15 min, peak 25 min, cool-down 15 min) and the simulator controller."""
    import config as C
    lay = CLOSURES[s["layout"]]
    cfg = C.RunConfig()
    ct = s["ctrl"]
    cfg.ctrl.strategy = ct.get("strategy", "NC")
    g = cfg.geo
    g.n_up, g.n_down, g.x_drop, g.L_down, g.chain = lay.n_up, lay.n_down, lay.x_drop, lay.L_down, lay.chain
    g.deployment, g.seg_range = "fixed", 4000.0
    g.d_g1_fixed = g.d_g2_fixed = 1600.0            # VSL-S: one sign 1.6 km upstream of the final lane end ...
    g.d_release_fixed = 600.0                       # ... governing 1.0 km; end of the limit 600 m upstream
    for k, v in BED.items():
        setattr(cfg.drv, k, v)
    for k, v in s.get("drv", {}).items():
        setattr(cfg.drv, k, v)
    cfg.sim.overtake_right = s.get("sim", {}).get("overtake_right", True)
    cfg.dem.truck_share = float(s.get("sim", {}).get("truck_share", 0.0))
    cfg.dem.cv_share = s["pen"]
    cfg.dem.q_ref, cfg.dem.level = 1000.0, s["dem"] / 1000.0
    cfg.dem.warm, cfg.dem.peak, cfg.dem.cool = 900.0, 1500.0, 900.0
    cfg.dem.warm_frac, cfg.dem.cool_frac = s["warm"] / 1000.0, s["cool"] / 1000.0
    cfg.sim.seed, cfg.sim.t_clear_max = s["seed"], 2400.0
    if cfg.ctrl.strategy == "VSL_FIXZONE":            # VSL-S
        assert float(ct["c"]) == 1.0, ct            # every driver responds to the sign (c is part of the run keys)
        cfg.drv.c_mean, cfg.drv.kappa, cfg.drv.tau_med = 1.0, math.inf, 8.0
        cfg.drv.comp = ct.get("comp")               # compliance level (block compliance); None: the posted limit
        cfg.ctrl.occ_set = ct["occ"]
    elif cfg.ctrl.strategy == "CV_MTFC":              # CV speed pacing
        cfg.ctrl.occ_set, cfg.cv.v_min_kmh, cfg.cv.avoid_drop_lane_dist = ct["occ"], ct["vmin"], 0.0
        if "nmax" in ct:
            cfg.cv.n_max = int(ct["nmax"])
        if "pa" in ct:
            cfg.cv.p_a = float(ct["pa"])
    return cfg


def compact(r, pol, s, wall):
    """Run record: indicators of Section 4.3, outflow series, lane speed and density profiles 25-40 min (last 2.5 km),
    lane changes by origin, destination and position, the speed grid, per-vehicle delays and the policy log."""
    import numpy as np
    sm, fl, g, veh = r["summary"], r["flow"], r["grid"], r["veh"]
    t, out = np.asarray(fl["t"], float), np.asarray(fl["out"], float)
    dtb = float(r["meta"]["dtb"])
    k5 = max(1, int(round(300 / dtb)))
    roll = np.convolve(out, np.ones(k5) / k5, mode="valid")
    w = (t[:len(roll)] >= 1200) & (t[:len(roll)] <= 2100)
    vl, kl = np.asarray(g["v_lane"], float), np.asarray(g["k_lane"], float)
    la = int(np.argmin(vl.shape))
    vl, kl = np.moveaxis(vl, la, 0), np.moveaxis(kl, la, 0)          # (lane, t, x)
    dx, gdt = float(g["dx"]), float(g["dt"])
    i0, i1 = int(1500 / gdt), int(2400 / gdt)
    j0 = int((X_DROP - 2500) // dx)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # cells without vehicles (lane ended)
        vprof = np.array([[np.nanmean(vl[ln, i0:i1, j0 + j]) * 3.6 for j in range(25)] for ln in range(4)], np.float32)
        kprof = np.array([[np.nanmean(kl[ln, i0:i1, j0 + j]) for j in range(25)] for ln in range(4)], np.float32)
    lc = np.asarray(r["events"]["lc"], float).reshape(-1, 5)
    m = (lc[:, 0] >= 900) & (lc[:, 0] < 2400) & (lc[:, 1] <= X_DROP) & (lc[:, 1] > X_DROP - 2500)
    lch = np.zeros((4, 4, 25), np.int16)
    for a, b, x in zip(lc[m, 2].astype(int), lc[m, 3].astype(int), lc[m, 1]):
        if 0 <= a < 4 and 0 <= b < 4:
            lch[a, b, min(24, int((X_DROP - x) // 100))] += 1
    idx = np.asarray(veh["idx"]).astype(np.int32)
    gv = np.asarray(g["v"], float) * 3.6
    em = r.get("emis") or {}
    ne = len(em.get("co2_mg", []))
    emis = dict(co2_g=float(np.mean(em["co2_mg"])) / 1000 if ne else float("nan"),
                fuel_g=float(np.mean(em["fuel_mg"])) / 1000 if ne else float("nan"),
                nox_g=float(np.mean(em["nox_mg"])) / 1000 if ne else float("nan"), n_emis=ne)
    return dict(key=key(s), spec=s, wall=wall,
                delay=float(sm["delay_mean_s_lb"]),
                q_queue=float(np.nanmean(out[(t >= 1800) & (t < 2400)])),
                q_peak=float(np.nanmean(out[(t >= 900) & (t < 2400)])),
                q_max5=float(np.nanmax(roll)), q_min5=float(np.nanmin(roll[w])) if w.any() else float("nan"),
                queue_max=float(sm["queue_max_m"]), queue_mean=float(sm.get("queue_mean_m", np.nan)),
                hb=float(sm["hard_brake_per_kveh"]), tet=float(sm.get("TET_s_per_veh", np.nan)),
                stops=float(sm.get("stops_per_veh", np.nan)), n_trunc=int(sm.get("n_trunc") or 0),
                valid=bool(sm.get("valid_insertion", True)), coll=int(sm.get("n_collisions") or 0),
                svar=float(sm.get("speed_var_upstream", float("nan"))), **emis,
                sm={k_: v_ for k_, v_ in sm.items() if isinstance(v_, (int, float, bool, str)) or v_ is None},
                t=t.astype(np.float32), out=out.astype(np.float32),
                veh=dict(idx=idx, t_dep=np.asarray(veh["t_dep"], np.float32), delay=np.asarray(veh["delay_s"], np.float32),
                         n_hb=np.asarray(veh["n_hb"], np.int16),
                         held=np.array([int(i) in pol.held for i in idx], bool)
                         | (np.asarray(veh["was_pacer"]).astype(bool) if "was_pacer" in veh else False),
                         lane1000=np.array([pol.lane1000.get(int(i), -1) for i in idx], np.int8)),
                log=dict(pol.log), n_pacers=len(pol.held), vprof=vprof, kprof=kprof, lch=lch,
                grid_v=gv.astype(np.float16), grid_dx=dx, grid_dt=gdt)


def one(s):
    """Simulate one specification and store its run record."""
    import config as C
    lay = CLOSURES[s["layout"]]
    C.NET_DIR = lay.net
    from core import simulation as Sim
    from policy import Policy
    cfg = build(s)
    pol = Policy(s["pol"], s["seed"], lay, cfg.sim.step)
    t0 = time.time()
    r = Sim.run(cfg, on_step=pol)
    res = compact(r, pol, s, time.time() - t0)
    p = path(s)
    tmp = p + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, p)
    return key(s)


# ------------------------------------------------------------------ orchestration
class _MS(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def free_gb():
    """Free physical memory (GB) on Windows; elsewhere a large value (no throttling)."""
    try:
        ms = _MS()
        ms.dwLength = ctypes.sizeof(_MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / 1e9
    except Exception:
        return 99.0


def log(msg):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + msg
    print(line, flush=True)
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "run.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(cur, done, total, rate, failed, final=False):
    eta = time.strftime("%H:%M", time.localtime(time.time() + (total - done) / max(rate, 1e-9) * 3600)) if rate > 0 else "?"
    txt = [f"updated {time.strftime('%Y-%m-%d %H:%M:%S')}", ""]
    txt.append("finished" if final else f"block {cur}: {done}/{total} runs, {rate:.0f} runs/h, block done at {eta}")
    txt.append(f"failed runs: {len(failed)}")
    os.makedirs(LOGS, exist_ok=True)
    with open(os.path.join(LOGS, "status.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(txt) + "\n")


def run_specs(specs, workers, cur):
    """Simulate the specifications whose run record does not exist yet, in parallel; a failed run is retried once."""
    os.makedirs(RUNS, exist_ok=True)
    uniq = {}
    for s in specs:
        uniq.setdefault(key(s), s)
    todo = [s for s in uniq.values() if not os.path.isfile(path(s))]
    total, done, failed, tries = len(todo), 0, [], {}
    log(f"block {cur}: {len(uniq)} distinct runs, {total} to simulate, {workers} workers")
    t0, last = time.time(), 0.0
    pending = list(todo)
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as ex:
        running = {}
        while pending or running:
            while pending and len(running) < workers and free_gb() > float(os.environ.get("FREE_GB_MIN", "1.3")):
                s = pending.pop(0)
                running[ex.submit(one, s)] = s
            if not running:
                time.sleep(10)
                continue
            fin, _ = wait(list(running), timeout=30, return_when=FIRST_COMPLETED)
            for f in fin:
                s = running.pop(f)
                try:
                    f.result()
                    done += 1
                except Exception as e:                      # noqa: BLE001
                    k_ = key(s)
                    tries[k_] = tries.get(k_, 0) + 1
                    log(f"FAILED {s['block']} {s['arm']} {s['layout']} {s['dem']} {s['seed']}: {repr(e)[:300]}")
                    if tries[k_] < 2:
                        pending.append(s)
                    else:
                        failed.append(s)
            if time.time() - last > 60 or not (pending or running):
                last = time.time()
                rate = done / max(time.time() - t0, 1.0) * 3600
                status(cur, done, total, rate, failed)
                if done and done % 50 == 0:
                    log(f"block {cur}: {done}/{total}, {rate:.0f} runs/h")
    log(f"block {cur} finished: {done}/{total} done, {len(failed)} failed")
    return failed


def load_block(specs):
    """Stored run records of the specifications (missing runs are skipped); r["spec"] is the specification given."""
    out = []
    for s in specs:
        p = path(s)
        if os.path.isfile(p):
            r = pickle.load(open(p, "rb"))
            r["spec"] = s
            out.append(r)
    return out

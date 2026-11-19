"""
Experiment blocks of the paper. Closure configurations: I = seq (successive closure, main test bed), II = base (lanes 2
and 3 end together), III = l43 (four-to-three closure).

    python experiments.py BLOCK [BLOCK ...] [--workers=N]

Tuning on the training seeds (Sections 3.8 and 4.1):
    demand_III         no control in III at 4800-6000 veh/h, seeds 101-104 -> results/demand_III.json
    tuning_gains       CAZP and VSL-L at the gains 0.1-1.0, I-III, three demands, seeds 101-108
    tuning_benchmarks  VSL-S and CV speed pacing on their grids, I-III, two highest demands, seeds 101-108
    select             parameters of every strategy and configuration -> results/tuning/tuned.json

Rule (Eq. 10): the value with the smallest mean paired delay change against no control on the training seeds.

Test blocks (test seeds), with the section and the figures of the paper:
    main               every strategy, I-III, seeds 1-30 (I: 1-40 for NC and CAZP)  5.1-5.3, 5.7; Figs. 2-4, 7c-d
    zipper_variants    zipper-merge sign followed by 25, 75 or 100% of the drivers  4.1
    ablation_I         ablation in I at 3800 veh/h, seeds 1-20                      5.4; Fig. 5a-c
    ablation_II        ablation in II at 3800 veh/h, seeds 1-20                     5.7
    gain               gains around the calibrated gain in I, seeds 1-20            5.4; Fig. 5d-f
    pacers             cap on the number of pacers and of CV speed pacing CVs (I), 1-16  5.5
    acceptance         acceptance of CAZP and CV speed pacing (I), seeds 1-16       5.5
    compliance         compliance levels with the speed limits (VSL-S, VSL-L, CAZP + VSL-S), I-III, highest demand,
                       seeds 1-16                                                   4.1, 5.8
    penetration       penetration x demand, acceptance x penetration (I), 1-12     5.5; Fig. 6
    robustness         modified test beds at 3800 veh/h (I), seeds 1-16             5.6; Fig. 7a-b
    thresholds         cell length and speed threshold of CAZP (I), seeds 1-16      4.2
    gain_transfer      CAZP with the gain of I in II and III, seeds 1-30            5.7
    acceptance_II_III  25% acceptance in II and III, highest demand, seeds 1-16     5.7
    bound_I            bounds of 10 and 30 km/h in I at 3800 veh/h, seeds 1-16      5.7
    bound_II_III       bounds of 10 and 30 km/h in II and III, highest demand, 1-16 5.7
"""
from __future__ import annotations
import json
import os
import pickle
import sys

import runs as R

GAINS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
CONFIGS = ("seq", "base", "l43")
NUMBER = {"seq": "I", "base": "II", "l43": "III"}
MAIN = "seq"                                   # main test bed: configuration I
CRED = 20.0     # km/h: bound of the holds of CAZP and VSL-L (the same margin as the credibility rule of VSL-S,
                #       sim/config.py, cred_kmh)
TUNING = os.path.join(R.RES, "tuning")
TRAIN_Q = list(range(101, 105))
TRAIN = list(range(101, 109))
GRID = dict(VSL1=[dict(occ=o) for o in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18)],
            PACE=[dict(occ=o, vmin=v) for o in (0.06, 0.08, 0.10, 0.14) for v in (40.0, 60.0, 80.0)])
TEST, TEST40 = list(range(1, 31)), list(range(1, 41))
S16, S20, S12 = list(range(1, 17)), list(range(1, 21)), list(range(1, 13))
HM_DEM = (3200, 3400, 3600, 3800, 4000)
ROBUSTNESS_ROWS = {"bed_F": dict(sim=dict(overtake_right=False)), "lcStrategic_1.0": dict(drv=dict(lcStrategic=1.0)),
                   "lcAssertive_1.5": dict(drv=dict(lcAssertive=1.5)), "lcAssertive_2": dict(drv=dict(lcAssertive=2.0)),
                   "lcAssertive_3": dict(drv=dict(lcAssertive=3.0)),
                   "lcCooperative_0.75": dict(drv=dict(lcCooperative=0.75)),
                   "lcCooperative_0.5": dict(drv=dict(lcCooperative=0.5)),
                   "tau_1.3": dict(drv=dict(tau_scale=1.3)), "trucks_10": dict(sim=dict(truck_share=0.10)),
                   "keepRight_0": dict(drv=dict(lcKeepRight=0.0))}


def jsave(obj, name):
    p = os.path.join(R.RES, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8", newline="\n"), indent=1)


def demand_III():
    p = os.path.join(R.RES, "demand_III.json")
    if not os.path.isfile(p):
        raise RuntimeError("results/demand_III.json missing: run block demand_III first")
    return json.load(open(p, encoding="utf-8"))


def demands(lay):
    """Peak demands, warm-up and cool-down flows of a configuration (veh/h)."""
    if lay == "l43":
        q = demand_III()
        return tuple(q["demand"]), q["warm"], q["cool"]
    return R.L3, 3000, 2000


def spec(block, arm, lay, dem, sd, pol, ctrl=None, **kw):
    _, w, c = demands(lay)
    return R.S(block, arm, lay, dem, sd, pol, ctrl, warm=w, cool=c, **kw)


def cazp(a):
    """CAZP with coupling gain a, the design of results/design_parameters.json and the bound CRED of the hold."""
    return dict(R.design()["cazp"], alpha=float(a), cred=CRED)


def vsll(a):
    """VSL-L with gain a; the lane limits are at most CRED below the lane into which the closing lane merges."""
    return dict(pol="NC", vsll=float(a), cred=CRED)


def t_arm(b, p):
    return b + "_" + "_".join(f"{k}{v:g}" for k, v in sorted(p.items()))


# ------------------------------------------------------------------ tuning on training seeds
def block_demand_III():
    return [R.S("demand_III", "NC", "l43", dem, sd, R.NC, warm=4500, cool=3000) for dem in (4800, 5100, 5400, 5700, 6000)
            for sd in TRAIN_Q]


def select_demand_III():
    """Demands of configuration III: the peak demands (3400, 3600, 3800 veh/h; warm-up 3000, cool-down 2000) scaled by
    the ratio of the queue discharge rate of III without control (mean outflow 30-40 min, demand levels that keep a
    queue) to that of II."""
    import numpy as np
    rs = R.load_block(block_demand_III())
    qd_main = float(np.mean([R.design()["queue_discharge_II"][str(d)] for d in R.L3]))
    by = {}
    for r in rs:
        by.setdefault(r["spec"]["dem"], []).append(r)
    rows = {d: dict(q_queue=float(np.mean([r["q_queue"] for r in v])), q_max5=float(np.mean([r["q_max5"] for r in v])),
                    queue_max=float(np.mean([r["queue_max"] for r in v])), delay=float(np.mean([r["delay"] for r in v])),
                    n=len(v)) for d, v in sorted(by.items())}
    sat = [d for d, v in rows.items() if d > v["q_queue"] + 200 and v["queue_max"] > 500]
    qd = float(np.mean([rows[d]["q_queue"] for d in sat]))
    r = qd / qd_main
    out = dict(demand=[int(100 * round(d * r / 100)) for d in R.L3], warm=int(100 * round(3000 * r / 100)),
               cool=int(100 * round(2000 * r / 100)), q_discharge=qd, q_discharge_II=qd_main, ratio=r,
               saturated_levels=sat, calibration=rows,
               rule="demands of II (3400, 3600, 3800 veh/h; warm-up 3000, cool-down 2000) times the ratio of the queue "
                    "discharge rates of III and II without control; III on training seeds 101-104")
    jsave(out, "demand_III.json")
    R.log(f"configuration III: discharge {qd:.0f} veh/h (II: {qd_main:.0f}) -> demand {out['demand']}, "
          f"warm {out['warm']}, cool {out['cool']}")
    return out


def training(block, lay, arms_, dems=None):
    """No control and the given arms (name, policy, simulator controller) at the given demands of a configuration (all
    three by default), training seeds 101-108."""
    all_dems, w, c = demands(lay)
    out = []
    for dem in dems or all_dems:
        for sd in TRAIN:
            out.append(R.S(block, "NC", lay, dem, sd, R.NC, warm=w, cool=c))
            out += [R.S(block, a, lay, dem, sd, pol, ctrl, warm=w, cool=c) for a, pol, ctrl in arms_]
    return out


def block_tuning_gains():
    """CAZP and VSL-L at every gain, both with the bound CRED, at the three peak demands of every configuration."""
    out = []
    for lay in CONFIGS:
        out += training("tuning_gains", lay, [(f"CAZP_a{a:g}", cazp(a), None) for a in GAINS]
                        + [(f"VSLL_a{a:g}", vsll(a), None) for a in GAINS])
    return out


def block_tuning_benchmarks():
    """VSL-S and CV speed pacing on their grids at the two highest peak demands of every configuration."""
    tu = R.design()
    out = []
    for lay in CONFIGS:
        out += training("tuning_benchmarks", lay,
                        [(t_arm("VSL1", p), R.NC, dict(tu["vsl_s"], **p)) for p in GRID["VSL1"]]
                        + [(t_arm("PACE", p), R.NC, dict(tu["cv_speed_pacing"], **p)) for p in GRID["PACE"]],
                        dems=demands(lay)[0][1:])
    return out


def J_table(specs):
    """Mean paired delay change against no control (s/veh) per configuration and arm, over complete arms only."""
    import numpy as np
    D = {}
    for s in specs:
        p = R.path(s)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                D[(s["layout"], s["arm"], s["dem"], s["seed"])] = pickle.load(f)["delay"]
    cells = {}
    for s in specs:
        cells.setdefault((s["layout"], s["arm"]), set()).add((s["dem"], s["seed"]))
    out = {}
    for lay in CONFIGS:
        J = {}
        for (ly, a), cs in sorted(cells.items()):
            if ly != lay or a == "NC":
                continue
            d = [D[(lay, a, dm, sd)] - D[(lay, "NC", dm, sd)] for dm, sd in sorted(cs)
                 if (lay, a, dm, sd) in D and (lay, "NC", dm, sd) in D]
            if len(d) == len(cs):
                J[a] = float(np.mean(d))
        out[lay] = J
    return out


def select_parameters():
    """Tuned parameters per configuration and the mean paired delay change of every candidate value (J)
    -> results/tuning/tuned.json."""
    cg, cb = J_table(block_tuning_gains()), J_table(block_tuning_benchmarks())
    out = dict(rule="smallest mean paired delay change vs NC on training seeds 101-108; CAZP and VSL-L at the three "
                    "demands, VSL-S and CV speed pacing at the two highest", cred_kmh=CRED, main=MAIN)
    for lay in CONFIGS:
        J = dict(CAZP={k: v for k, v in cg[lay].items() if k.startswith("CAZP_a")},
                 VSLL={k: v for k, v in cg[lay].items() if k.startswith("VSLL_a")},
                 VSL1={k: v for k, v in cb[lay].items() if k.startswith("VSL1_")},
                 PACE={k: v for k, v in cb[lay].items() if k.startswith("PACE_")})
        if min(len(J["CAZP"]), len(J["VSLL"])) < len(GAINS) or len(J["VSL1"]) < len(GRID["VSL1"]) \
                or len(J["PACE"]) < len(GRID["PACE"]):
            raise RuntimeError(f"tuning blocks incomplete for {lay}")
        best = {b: min(J[b], key=J[b].get) for b in J}
        out[lay] = dict(CAZP=float(best["CAZP"].split("_a")[1]), VSLL=float(best["VSLL"].split("_a")[1]),
                        VSL1=next(p for p in GRID["VSL1"] if t_arm("VSL1", p) == best["VSL1"]),
                        PACE=next(p for p in GRID["PACE"] if t_arm("PACE", p) == best["PACE"]), J=J)
    jsave(out, os.path.join("tuning", "tuned.json"))
    return out


def tuned():
    p = os.path.join(TUNING, "tuned.json")
    if not os.path.isfile(p):
        raise RuntimeError("results/tuning/tuned.json missing: run 'select' after the tuning blocks")
    return json.load(open(p, encoding="utf-8"))


# ------------------------------------------------------------------ test programme
def fixed_arms():
    """Strategies without tuned parameters."""
    return {"NC": (R.NC, None), "EM": (R.EM, None), "ZMSD": (R.ZMSD, None)}


def arms(lay):
    """Every strategy of the comparison with the parameters tuned for configuration lay."""
    t, tu = tuned()[lay], R.design()
    c = cazp(t["CAZP"])
    v1 = dict(tu["vsl_s"], occ=t["VSL1"]["occ"])
    pc = dict(tu["cv_speed_pacing"], occ=t["PACE"]["occ"], vmin=t["PACE"]["vmin"])
    vl = vsll(t["VSLL"])
    out = fixed_arms()
    out.update({"CAZP": (c, None), "VSL1": (R.NC, v1), "VSLL": (vl, None), "PACE": (R.NC, pc),
                "VSL1_ZMSD": (R.ZMSD, v1), "VSLL_ZMSD": (dict(vl, zms=0.5, zms_dyn=2), None), "CAZP_VSL1": (c, v1)})
    return out


def block_main():
    out = []
    for lay in CONFIGS:
        A = arms(lay)
        for dem in demands(lay)[0]:
            for a, (pol, ctrl) in A.items():
                seeds = TEST40 if (lay == MAIN and a in ("NC", "CAZP")) else TEST
                out += [spec("main", a, lay, dem, sd, pol, ctrl) for sd in seeds]
    return out


def block_zipper_variants():
    return [spec("zipper_variants", f"ZMSD_c{int(round(c * 100))}", MAIN, dem, sd, dict(R.ZMSD, zms=c))
            for dem in demands(MAIN)[0] for sd in S20 for c in (0.25, 0.75, 1.0)]


def ablation_variants(C, formation):
    var = {"no_coverage": dict(C, reach=False, lateral=False, depth=1), "depth_1": dict(C, depth=1),
           "no_formation": dict(C, form=False), "tie_only": dict(C, hold=False), "hold_only": dict(C, tie=False),
           "outer_only": dict(C, lanes="outer"), "inner_only": dict(C, lanes="inner"),
           "fixed_zone": dict(C, zone_mode="fixed"), "display": dict(C, T=5.0, q_kmh=5.0, tau=3.0, err=0.15)}
    if not formation:
        del var["no_formation"]
    return var


def block_ablation_I():
    """NC and CAZP of the ablation come from block main; lane hopping off (NC_nohop, CAZP_nohop) checks the lane-change
    model."""
    C = arms(MAIN)["CAZP"][0]
    var = ablation_variants(C, R.CLOSURES[MAIN].same_end)
    var.update({"NC_nohop": dict(R.NC, nohop=True), "CAZP_nohop": dict(C, nohop=True)})
    return [spec("ablation_I", a, MAIN, 3800, sd, pol) for sd in S20 for a, pol in var.items()]


def block_ablation_II():
    """In II the alignment rule is active; NC and CAZP come from block main."""
    C = arms("base")["CAZP"][0]
    return [spec("ablation_II", a, "base", 3800, sd, pol) for sd in S20
            for a, pol in ablation_variants(C, True).items()]


def gain_values():
    """Gains of block gain: calibrated value +-0.1 and +-0.2 (within 0.1-1.0)."""
    a0 = tuned()[MAIN]["CAZP"]
    g = {round(a0 + d, 1) for d in (-0.2, -0.1, 0.1, 0.2)}
    return sorted(x for x in g if 0.1 - 1e-9 <= x <= 1.0 + 1e-9 and abs(x - a0) > 1e-9)


def block_gain():
    C = arms(MAIN)["CAZP"][0]
    return [spec("gain", f"fix{a:g}", MAIN, dem, sd, dict(C, alpha=float(a))) for dem in demands(MAIN)[0] for sd in S20
            for a in gain_values()]


def block_pacers():
    A = arms(MAIN)
    C, pc = A["CAZP"][0], A["PACE"][1]
    out = []
    for sd in S16:
        for dem in demands(MAIN)[0]:
            out += [spec("pacers", f"CAZP_nmax{n}", MAIN, dem, sd, dict(C, n_max=n)) for n in (2, 4, 8, 12)]
        out += [spec("pacers", f"PACE_nmax{n}", MAIN, 3800, sd, R.NC, dict(pc, nmax=n)) for n in (2, 4, 8, 12)]
    return out


def block_acceptance():
    """Acceptance probability of the CV drivers of CAZP and of CV speed pacing (0.75 in block main)."""
    A = arms(MAIN)
    C, pc = A["CAZP"][0], A["PACE"][1]
    out = []
    for sd in S16:
        out += [spec("acceptance", f"CAZP_c{c}", MAIN, 3800, sd, dict(C, p_a=c)) for c in (0.25, 0.5, 1.0)]
        out += [spec("acceptance", f"PACE_c{c}", MAIN, 3800, sd, R.NC, dict(pc, pa=c)) for c in (0.25, 0.5, 1.0)]
    return out


LEVELS = ("VH", "H", "M", "L")   # compliance levels of Hellinga and Mandelzys (2011), sim/core/driver_response.comply


def block_compliance():
    """VSL-S, VSL-L and CAZP + VSL-S with the compliance levels Very High, High, Moderate and Low at the highest demand
    of every configuration; in block main, drivers treat a posted limit as the static limit."""
    out = []
    for lay in CONFIGS:
        A, dem = arms(lay), demands(lay)[0][-1]
        c, v1, vl = A["CAZP"][0], A["VSL1"][1], A["VSLL"][0]
        for lv in LEVELS:
            out += [spec("compliance", f"VSL1_{lv}", lay, dem, sd, R.NC, dict(v1, comp=lv)) for sd in S16]
            out += [spec("compliance", f"VSLL_{lv}", lay, dem, sd, dict(vl, vsll_comp=lv)) for sd in S16]
            out += [spec("compliance", f"CAZP_VSL1_{lv}", lay, dem, sd, c, dict(v1, comp=lv)) for sd in S16]
    return out


def block_penetration():
    C = arms(MAIN)["CAZP"][0]
    out = []
    for dem in HM_DEM:
        for sd in S12:
            out.append(spec("penetration", "NC", MAIN, dem, sd, R.NC))
            out += [spec("penetration", f"CAZP_pen{p}", MAIN, dem, sd, C, pen=p) for p in (0.025, 0.05, 0.10, 0.20, 0.40)]
    out += [spec("penetration", f"CAZP_acc{a}_pen{p}", MAIN, 3600, sd, C if a == 0.75 else dict(C, p_a=a), pen=p)
            for sd in S12 for a in (0.25, 0.5, 0.75, 1.0) for p in (0.05, 0.10, 0.20)]
    return out


def block_robustness():
    C = arms(MAIN)["CAZP"][0]
    return [spec("robustness", f"{a}|{row}", MAIN, 3800, sd, pol, **o) for sd in S16 for row, o in ROBUSTNESS_ROWS.items()
            for a, pol in (("NC", R.NC), ("CAZP", C))]


def block_thresholds():
    C = arms(MAIN)["CAZP"][0]
    var = {f"CAZP_S{s}": dict(C, q_thr=float(s), v_on=float(s), v_off=float(s) + 20.0) for s in (40, 60, 75)}
    var.update({f"CAZP_L{c}": dict(C, cell=float(c)) for c in (50, 200)})
    return [spec("thresholds", a, MAIN, 3800, sd, pol) for a, pol in var.items() for sd in S16]


def block_gain_transfer():
    a0 = tuned()[MAIN]["CAZP"]
    out = []
    for lay in CONFIGS:
        if lay == MAIN:
            continue
        C = dict(arms(lay)["CAZP"][0], alpha=float(a0))
        out += [spec("gain_transfer", "CAZP_mainGain", lay, dem, sd, C) for dem in demands(lay)[0] for sd in TEST]
    return out


def block_acceptance_II_III():
    out = []
    for lay in CONFIGS:
        if lay == MAIN:
            continue
        C, dem = arms(lay)["CAZP"][0], demands(lay)[0][-1]
        out += [spec("acceptance_II_III", "CAZP_c0.25", lay, dem, sd, dict(C, p_a=0.25)) for sd in S16]
    return out


def block_bound_I():
    A = arms(MAIN)
    C, vl = A["CAZP"][0], A["VSLL"][0]
    out = []
    for dv in (10.0, 30.0):
        out += [spec("bound_I", f"CAZP_dv{dv:g}", MAIN, 3800, sd, dict(C, cred=dv)) for sd in S16]
        out += [spec("bound_I", f"VSLL_dv{dv:g}", MAIN, 3800, sd, dict(vl, cred=dv)) for sd in S16]
    return out


def block_bound_II_III():
    out = []
    for lay in CONFIGS:
        if lay == MAIN:
            continue
        A, dem = arms(lay), demands(lay)[0][-1]
        C, vl = A["CAZP"][0], A["VSLL"][0]
        for dv in (10.0, 30.0):
            out += [spec("bound_II_III", f"CAZP_dv{dv:g}", lay, dem, sd, dict(C, cred=dv)) for sd in S16]
            out += [spec("bound_II_III", f"VSLL_dv{dv:g}", lay, dem, sd, dict(vl, cred=dv)) for sd in S16]
    return out


TUNING_BLOCKS = dict(demand_III=block_demand_III, tuning_gains=block_tuning_gains,
                     tuning_benchmarks=block_tuning_benchmarks)
TEST_BLOCKS = dict(main=block_main, zipper_variants=block_zipper_variants, ablation_I=block_ablation_I,
                   ablation_II=block_ablation_II, gain=block_gain, pacers=block_pacers, acceptance=block_acceptance,
                   compliance=block_compliance, penetration=block_penetration, robustness=block_robustness, thresholds=block_thresholds,
                   gain_transfer=block_gain_transfer, acceptance_II_III=block_acceptance_II_III, bound_I=block_bound_I,
                   bound_II_III=block_bound_II_III)
BLOCKS = {**TUNING_BLOCKS, **TEST_BLOCKS}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        raise SystemExit(__doc__)
    workers = 10
    for a in sys.argv[1:]:
        if a.startswith("--workers="):
            workers = int(a.split("=")[1])
    fails = []
    for b in args:
        if b == "select":
            t = select_parameters()
            R.log("tuned: " + "; ".join(f"{NUMBER[lay]} CAZP {t[lay]['CAZP']:g} VSL-L {t[lay]['VSLL']:g} VSL-S "
                                        f"{t[lay]['VSL1']} CV speed pacing {t[lay]['PACE']}" for lay in CONFIGS))
            continue
        if b not in BLOCKS:
            raise SystemExit(f"unknown block {b!r}; blocks: {', '.join(BLOCKS)}, select")
        fails += R.run_specs(BLOCKS[b](), workers, b)
        if b == "demand_III":
            select_demand_III()
    R.status(args[-1] if args else "", 0, 0, 0, fails, final=True)
    R.log(f"done {' '.join(args)}, failed runs: {len(fails)}")


if __name__ == "__main__":
    main()

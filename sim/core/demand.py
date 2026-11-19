"""
Demand and vehicle types.

Arrivals are an inhomogeneous Poisson process with the piecewise-constant rate
    warm-up  : warm_frac * q_ref
    peak     : level * q_ref
    cool-down: cool_frac * q_ref
optionally overridden by a linear ramp `Demand.ramp = (t0, t1, q0, q1)` between t0 and t1.
The process is generated at the maximum rate and thinned with probability lambda(t) / lambda_max.
Rates are multiples of the reference flow q_ref (the experiments set q_ref = 1000 veh/h and
level = peak demand / 1000).
"""
from __future__ import annotations
import numpy as np

# three human car-following variants add heterogeneity (desired headway, accel, comfortable decel)
CF_VARIANTS = [(1.0, 1.6, 2.4), (1.3, 1.4, 2.2), (1.6, 1.2, 2.0)]
COL_HV = "0.64,0.68,0.73"
COL_CV = "0.18,0.45,0.75"
COL_TRK = "0.35,0.37,0.40"


def _vtype(tid, cfg, tau, accel, decel, color, length=4.8, vmax=50.0, sf=None):
    d = cfg.drv
    sf = sf or f"normc({d.mu_f:.3f},{d.sigma_f:.3f},0.2,2.0)"
    lc = (f'lcSpeedGain="{d.lcSpeedGain}" lcAssertive="{d.lcAssertive}" '
          f'lcStrategic="{d.lcStrategic}" lcLookaheadLeft="{d.lcLookaheadLeft}" lcCooperative="{d.lcCooperative}"')
    if getattr(d, "lcKeepRight", None) is not None:
        lc += f' lcKeepRight="{d.lcKeepRight}"'
    accel = accel * d.accel_scale
    tau = tau * getattr(d, "tau_scale", 1.0)
    common = (f'id="{tid}" vClass="passenger" length="{length}" minGap="2.0" maxSpeed="{vmax}" '
              f'speedFactor="{sf}" actionStepLength="{d.action_step}" color="{color}" {lc}')
    if d.cf_model == "Krauss":
        return f'    <vType {common} carFollowModel="Krauss" accel="{accel + 0.8 * d.accel_scale}" decel="4.5" sigma="{d.krauss_sigma}" tau="{tau}"/>'
    if d.cf_model == "EIDM":
        return (f'    <vType {common} carFollowModel="EIDM" accel="{accel}" decel="{decel}" '
                f'emergencyDecel="9" tau="{tau}" delta="4" treaction="0.5" tPersEstimate="10" '
                f'ccoolness="0.99" sigmaleader="0.02" sigmagap="0.1" sigmaerror="0.1" jerkmax="3" epsilonacc="1" '
                f'taccmax="1.2" Mflatness="2" Mbegin="0.7"/>')
    return (f'    <vType {common} carFollowModel="IDM" accel="{accel}" decel="{decel}" '
            f'emergencyDecel="9" tau="{tau}" delta="4"/>')


def rate_vph(cfg, t: float) -> float:
    """Arrival rate (veh/h) at time t: block profile, overridden by the linear ramp inside [t0, t1]."""
    dm = cfg.dem
    if dm.ramp is not None:
        t0, t1, q0, q1 = dm.ramp
        if t0 <= t < t1:
            return q0 + (q1 - q0) * (t - t0) / max(t1 - t0, 1e-9)
    if t < dm.warm:
        return dm.warm_frac * dm.q_ref
    if t < dm.warm + dm.peak:
        return dm.level * dm.q_ref
    return dm.cool_frac * dm.q_ref


def write_routes(cfg, path: str, rng_dem: np.random.Generator, rng_pac: np.random.Generator):
    """Write vTypes + explicit vehicles; return list of (vid, depart, type, comp).

    comp: True when a CV's driver follows pacer advisories (drawn from rng_pac with p_a); False for HV/TRK.
    """
    dm = cfg.dem
    types = []
    for k, (tau, a, b) in enumerate(CF_VARIANTS):
        types.append(_vtype(f"HV_{k}", cfg, tau, a, b, COL_HV))
        types.append(_vtype(f"CV_{k}", cfg, tau, a, b, COL_CV))
    types.append(_vtype("TRK", cfg, 1.8, 0.8, 1.8, COL_TRK, length=12.0, vmax=25.0,
                        sf="normc(1.0,0.05,0.8,1.2)"))
    q_max = max(dm.warm_frac, dm.level, dm.cool_frac) * dm.q_ref
    if dm.ramp is not None:
        q_max = max(q_max, dm.ramp[2], dm.ramp[3])
    lam_max = q_max / 3600.0
    veh = []
    n = 0
    t = rng_dem.exponential(1.0 / lam_max)
    while t < dm.t_end:
        if rng_dem.random() < rate_vph(cfg, t) / q_max:      # thinning -> inhomogeneous Poisson
            u = rng_dem.random()
            if u < dm.truck_share:
                typ = "TRK"
            elif u < dm.truck_share + dm.cv_share:
                typ = f"CV_{rng_dem.integers(3)}"
            else:
                typ = f"HV_{rng_dem.integers(3)}"
            comp = bool(rng_pac.random() < cfg.cv.p_a) if typ.startswith("CV") else False
            veh.append((f"v{n}", round(float(t), 2), typ, comp))
            n += 1
        t += rng_dem.exponential(1.0 / lam_max)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<routes>\n" + "\n".join(types) + "\n")
        f.write(f'    <route id="r0" edges="{" ".join(e for e, _ in cfg.geo.chain)}"/>\n')
        # departSpeed="last": enter at the speed of the previous vehicle on the lane. "max" is patched down
        # to any safe speed (entry becomes a rolling queue, ~3.2k veh/h on 3 lanes); "desired" waits
        # for a full-speed gap (10-50 s insertion waits at level >= 1.10).
        for vid, td, typ, _ in veh:
            f.write(f'    <vehicle id="{vid}" type="{typ}" route="r0" depart="{td:.2f}" '
                    f'departLane="free" departSpeed="last" departPos="base"/>\n')
        f.write("</routes>\n")
    return veh

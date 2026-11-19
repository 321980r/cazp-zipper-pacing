"""
Run configuration of the SUMO test bed. An experiment builds a RunConfig from these defaults and overrides what it
varies (runs.build). Speeds are in m/s unless the name ends in _kmh.

Controller strategies of the simulator (CAZP, VSL-L, the zipper-merge sign and CV early merge act through the step
hook of policy.Policy instead):
    NC           no sign controller
    VSL_FIXZONE  VSL-S: one variable speed limit sign with occupancy feedback (fixed deployment)
    CV_MTFC      CV speed pacing: the feedback speed is sent to connected vehicles in all lanes
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
import hashlib
import json
import math

USER_SUMO_HOME: str | None = None     # SUMO folder (contains bin/sumo); None: SUMO_HOME or a standard location

NET_DIR: str | None = None            # network folder of the closure configuration (set by runs.one)

KMH = 1.0 / 3.6
MODEL_VERSION = "3.0"

# driver and simulator settings of the test bed (Table 1 of the paper)
TESTBED = dict(cf_model="EIDM", accel_scale=0.7, action_step=0.5, lcCooperative=1.0, lc_duration=0.0)

FIXZONE = frozenset({"VSL_FIXZONE"})                  # strategies that require the fixed deployment
STRATEGIES = frozenset({"NC", "CV_MTFC"}) | FIXZONE


def ceil10(v_kmh: float) -> float:
    return 10.0 * math.ceil(v_kmh / 10.0 - 1e-9)


@dataclass
class Geometry:
    """Control positions are distances upstream of the final lane end x_drop, where the final bottleneck begins.

    deployment "fixed" (used in every experiment): two sign gantries at x_drop - d_g1_fixed and x_drop - d_g2_fixed,
    release at x_drop - d_release_fixed, pacer zone from x_drop - d_zone_fixed. deployment "tail": a gantry array
    every d_gantry upstream of the acceleration area whose application area follows the estimated queue tail."""
    n_up: int = 3                    # lanes upstream of the lane drop
    n_down: int = 2                  # lanes downstream
    # mainline edges upstream -> downstream with the junction at the end of each edge ("" for the last one);
    # a successive closure has a middle edge, e.g. (("E_up", "m1"), ("E_mid", "n1"), ("E_down", ""))
    chain: tuple = (("E_up", "n1"), ("E_down", ""))
    x_drop: float = 8000.0           # position of the final lane drop (m)
    L_down: float = 1000.0           # downstream section length (m)
    v_base_kmh: float = 100.0        # static speed limit
    deployment: str = "tail"         # "tail" | "fixed"
    d_acc: float = 600.0             # acceleration area [x_drop - d_acc, x_drop]: no limit, no pacer
    d_gantry: float = 500.0          # gantry spacing (m)
    n_gantries: int = 12             # gantries upstream of the release gantry
    L_app: float = 2000.0            # application area = pacer zone extent (m)
    staircase_kmh: float = 20.0      # the gantry upstream of the application area shows min(v0, v_s + staircase)
    m_sel: float = 300.0             # pacer zone ends m_sel upstream of the control end
    m_rel: float = 100.0             # pacers are released m_rel upstream of the estimated tail
    x_min_pacer: float = 600.0       # no pacer selection below this position (insertion region)
    seg_det: float = 200.0           # segment-detector length of the tail estimator (m)
    seg_range: float = 6700.0        # segment detectors cover [x_drop - seg_range, x_drop]
    seg_len: float = 400.0           # length of a coverage cell for pacers (m)
    d_merge: float = 300.0           # occupancy measurement (bottleneck) area = (x_drop - d_merge, x_drop)
    outflow_x: float = 300.0         # outflow loop position on the downstream edge (m)
    inflow_x: float = 300.0          # inflow loop position on the upstream edge (m)
    loop_spacing: float = 200.0      # emulated lane loops in the control zone (m)
    # fixed deployment
    d_g1_fixed: float = 1500.0
    d_g2_fixed: float = 700.0
    d_release_fixed: float = 400.0
    d_zone_fixed: float = 2300.0

    @property
    def fixed(self) -> bool:
        return self.deployment == "fixed"

    @property
    def gantries(self) -> tuple:
        """Gantry positions upstream -> downstream (12 for "tail", 2 for "fixed")."""
        if self.fixed:
            return (self.x_drop - self.d_g1_fixed, self.x_drop - self.d_g2_fixed)
        x0 = self.x_drop - self.d_acc
        return tuple(x0 - j * self.d_gantry for j in range(self.n_gantries, 0, -1))

    @property
    def release_x(self) -> float:
        """Release gantry (always v0); end of the pacer range."""
        return self.x_drop - (self.d_release_fixed if self.fixed else self.d_acc)

    @property
    def zone_start(self) -> float:
        return self.x_drop - self.d_zone_fixed if self.fixed else self.x_min_pacer

    @property
    def n_app(self) -> int:
        """Number of application gantries (tail deployment)."""
        return max(1, int(round(self.L_app / self.d_gantry)))

    @property
    def acc_area(self) -> tuple:
        return (self.release_x, self.x_drop)

    @property
    def merge_area(self) -> tuple:
        return (self.x_drop - self.d_merge, self.x_drop)

    @property
    def segments(self) -> list:
        """(x0, x1) segments of length seg_det ending at x_drop and covering [x_drop - seg_range, x_drop]."""
        x_lo = max(0.0, self.x_drop - self.seg_range)
        n = int(math.ceil((self.x_drop - x_lo) / self.seg_det - 1e-9))
        return [(self.x_drop - (n - i) * self.seg_det, self.x_drop - (n - i - 1) * self.seg_det) for i in range(n)]

    @property
    def v_base(self) -> float:
        return self.v_base_kmh * KMH

    @property
    def x_end(self) -> float:
        """Nominal end of the network (before the internal lane of the drop node)."""
        return self.x_drop + self.L_down


@dataclass
class Demand:
    q_ref: float = 3600.0            # reference flow (veh/h); the experiments set q_ref = 1000 and level = demand/1000
    level: float = 1.05              # peak demand as a multiple of q_ref
    warm: float = 900.0              # warm-up duration (s)
    peak: float = 2700.0             # peak duration (s)
    cool: float = 900.0              # cool-down duration (s)
    warm_frac: float = 0.80          # warm-up demand as a multiple of q_ref
    cool_frac: float = 0.60          # cool-down demand as a multiple of q_ref
    cv_share: float = 0.10           # share of connected (human-driven) vehicles
    truck_share: float = 0.0         # share of heavy vehicles (robustness test)
    # (t0, t1, q0_vph, q1_vph): linear arrival rate between t0 and t1, overrides the block rate there
    ramp: tuple | None = None

    @property
    def t_end(self) -> float:
        return self.warm + self.peak + self.cool


@dataclass
class Driver:
    # desired speed relative to the anchor: SUMO speedFactor ~ normc(mu_f, sigma_f)
    mu_f: float = 1.00
    sigma_f: float = 0.10
    # response to a posted limit: anchor a* = v_base + c (v_disp - v_base)
    c_mean: float = 1.0              # mean compliance
    kappa: float = math.inf          # Beta concentration of the compliance (inf: every driver has c_mean)
    comp: str | None = None          # compliance level VH, H, M or L (core.driver_response.comply); None: posted limit
    tau_med: float = 0.0             # median response time constant (s); 0 = instantaneous
    tau_sig_ln: float = 0.5          # log-normal spread of tau
    d_v: float = 0.0                 # visibility distance upstream of a gantry (m)
    # car following and lane changing
    cf_model: str = TESTBED["cf_model"]          # "IDM", "EIDM" or "Krauss"
    lcSpeedGain: float = 1.0
    lcAssertive: float = 1.0
    lcStrategic: float = 0.5
    lcLookaheadLeft: float = 1.0
    lcCooperative: float = TESTBED["lcCooperative"]   # LC2013 willingness to cooperate
    lcKeepRight: float = None        # None: SUMO default (1.0, not written); 0: no keep-right motivation (robustness)
    lc_duration: float = TESTBED["lc_duration"]       # > 0: continuous lane changes lasting this long (s)
    accel_scale: float = TESTBED["accel_scale"]       # multiplies the maximum acceleration of every human vType
    tau_scale: float = 1.0           # multiplies the desired time headway of every human vType
    krauss_sigma: float = 0.5        # driver imperfection of the Krauss model
    action_step: float = TESTBED["action_step"]       # decision interval of human drivers (s)


@dataclass
class CVConf:
    """Connected vehicles of CV speed pacing."""
    n_max: int = 16                  # maximum number of simultaneously active CVs
    p_a: float = 0.75                # probability that a selected CV driver follows the advisories
    tau_a: float = 2.0               # execution dead time of the driver (s); log-normal per CV (sigma 0.5)
    rate_kmh_per_5s: float = 5.0     # largest change of the advisory per 5 s
    v_min_kmh: float = 60.0          # lowest advisory
    sense_range: float = 100.0       # onboard sensing range (m)
    noise_kmh: float = 0.0           # measurement noise on sensed speeds (std, km/h)
    avoid_drop_lane_dist: float = 900.0  # CVs in a closing lane within this distance of its end are not selected


@dataclass
class CtrlConf:
    strategy: str = "NC"             # one of STRATEGIES
    T_exec: float = 5.0              # advisory and tail-estimation clock (s)
    T_band: float = 300.0            # window of the overtaking counts of the CV pacers (s)
    mtfc_period: float = 30.0        # feedback period (s)
    K_P: float = 2.0                 # feedback gains (occupancy as a fraction)
    K_I: float = 1.0
    occ_set: float = 0.16            # occupancy set-point
    b_min: float = 0.6               # lowest speed factor of the feedback
    n_on: int = 2                    # on after n_on consecutive periods with b < 0.999
    deadband_kmh: float = 1.0
    hold: float = 120.0              # minimum holding time of a displayed value (s)
    step_max_kmh: float = 20.0       # largest change of the displayed value per update
    gantry_diff_kmh: float = 20.0    # largest difference between consecutive gantries
    vs_min_kmh: float = 60.0         # lowest displayed value
    # queue-tail estimator and tail deployment
    tail_v_on_kmh: float = 45.0      # segment congested below this ...
    tail_v_off_kmh: float = 55.0     # ... free above this (hysteresis)
    tail_win_s: float = 30.0         # trailing mean of the 1-s segment speeds
    tail_smooth_n: int = 3           # median over the last n raw tail estimates (5-s clock)
    tail_fill: float = 0.7           # share of congested segments between tail and head
    tail_hys_m: float = 100.0        # control end x_e = x_tail - tail_hys_m
    tail_dwell_s: float = 30.0       # a gantry's mask state changes at most once per dwell
    cred_kmh: float = 20.0           # credibility rule: displayed value >= ceil10(v_app - cred)
    release_dv_kmh: float = 15.0     # a CV is released when v < v_req - release_dv

    def __post_init__(self):
        if self.strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy {self.strategy!r}; valid: {sorted(STRATEGIES)}")


@dataclass
class SimConf:
    version: str = MODEL_VERSION
    seed: int = 1
    step: float = 0.5
    t_clear_max: float = 1200.0      # time allowed after the end of the demand to clear the network (s)
    ts_dt: float = 30.0              # time bin of the time-space grids (s)
    ts_dx: float = 100.0             # space bin (m)
    record: bool = True              # keep time series
    log_traj: bool = False           # record a trajectory window
    gui: bool = False
    overtake_right: bool = False     # SUMO option --lanechange.overtake-right


@dataclass
class RunConfig:
    geo: Geometry = field(default_factory=Geometry)
    dem: Demand = field(default_factory=Demand)
    drv: Driver = field(default_factory=Driver)
    cv: CVConf = field(default_factory=CVConf)
    ctrl: CtrlConf = field(default_factory=CtrlConf)
    sim: SimConf = field(default_factory=SimConf)
    tag: str = ""

    def to_dict(self):
        """JSON-serialisable dict (kappa=inf -> None, ramp -> list or None)."""
        d = asdict(self)
        d["drv"]["kappa"] = None if math.isinf(self.drv.kappa) else self.drv.kappa
        d["dem"]["ramp"] = None if self.dem.ramp is None else [float(v) for v in self.dem.ramp]
        return d

    def key(self) -> str:
        """Stable identifier of the configuration."""
        s = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return hashlib.md5(s.encode()).hexdigest()[:12]

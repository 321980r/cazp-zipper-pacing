"""
Driver response to posted speed limits (VSL-S).

    v_des,i(t) = f_i * a_i(t)
    a_i*       = v_base + c_i (g(v_disp) - v_base)       when passing a gantry
    a_i       <- a_i + dt/tau_i (a_i* - a_i)             first-order response

c_i = 1, tau_i = 0 reproduces SUMO's default drivers; c_i in {0,1} reproduces the
usual "p% of drivers comply" assumption. g is the compliance level (comply); by
default g(v) = v, i.e. drivers treat the posted limit as the static limit. The anchor
is implemented through the per-vehicle speedFactor (the lane speed limit stays at
v_base), so only drivers who actually passed a gantry react, and only to the value
they saw.

state[vid] = dict(f, c, tau, a, a_star, v_disp, next_g, pushed, t_push); v_disp is the
last displayed value the driver saw (m/s; v_base before the first gantry).
"""
from __future__ import annotations
import math
import numpy as np

# Compliance levels of Hellinga and Mandelzys (2011, Table 2): mean free speed under a posted variable limit. Very High,
# High and Moderate: ratio of the mean free speed to the posted limit (about 70, 25 and 6% of the free speeds at or
# below the limit with the 10% spread of the test bed). Low: mean free speed of 119, 104 and 88 km/h at posted limits of
# 100, 80 and 60 km/h, linear in between; the levels are defined down to 60 km/h, and below 60 km/h the Low level keeps
# its 60 km/h value.
RATIO = {"VH": 0.95, "H": 1.08, "M": 1.19}
LOW = ((60.0, 88.0), (80.0, 104.0), (100.0, 119.0))      # (posted limit, mean free speed), km/h


def comply(v_post: float, level, v_base: float) -> float:
    """Anchor (m/s) of the desired speed under a posted variable limit v_post (m/s) at a compliance level (None: the
    posted limit itself). A sign that shows the static limit v_base leaves the anchor at v_base, and no posted limit
    raises the anchor above v_base."""
    if level is None or v_post >= v_base:
        return v_post
    if level == "L":
        v = v_post * 3.6
        (x0, y0), (x1, y1), (x2, y2) = LOW
        if v <= x0:
            m = y0
        elif v <= x1:
            m = y0 + (y1 - y0) * (v - x0) / (x1 - x0)
        else:
            m = y1 + (y2 - y1) * (v - x1) / (x2 - x1)
        return min(v_base, m / 3.6)
    return min(v_base, RATIO[level] * v_post)


class DriverResponse:
    def __init__(self, cfg, rng_drv: np.random.Generator, k):
        self.cfg, self.rng, self.k = cfg, rng_drv, k
        g = cfg.geo
        self.vb = g.v_base
        self.gx = list(g.gantries) + [g.release_x]      # the last one always shows v_base
        self.level = cfg.drv.comp                        # compliance level (None: the posted limit itself)
        self.state = {}                                  # vid -> dict

    # ---------------------------------------------------------------- sampling
    def _c(self):
        d = self.cfg.drv
        if math.isinf(d.kappa) or d.c_mean in (0.0, 1.0):
            return float(d.c_mean)
        if d.kappa == 0:                                   # binary: complies fully with probability c_mean
            return float(self.rng.random() < d.c_mean)
        a, b = d.c_mean * d.kappa, (1.0 - d.c_mean) * d.kappa
        return float(self.rng.beta(a, b))

    def _tau(self):
        d = self.cfg.drv
        if d.tau_med <= 0:
            return 0.0
        return float(self.rng.lognormal(math.log(d.tau_med), d.tau_sig_ln))

    def on_depart(self, vid: str):
        f = self.k.vehicle.getSpeedFactor(vid)
        self.state[vid] = dict(f=f, c=self._c(), tau=self._tau(), a=self.vb, a_star=self.vb, v_disp=self.vb,
                               next_g=0, pushed=f, t_push=-1e9)

    def on_arrive(self, vid: str):
        self.state.pop(vid, None)

    # ---------------------------------------------------------------- dynamics
    def step(self, t: float, dt: float, xs: dict, signs_ms: list):
        """xs: vid -> global position; signs_ms: displayed values (m/s) for G1..Gn."""
        disp = list(signs_ms) + [self.vb]
        dv = self.cfg.drv.d_v
        for vid, s in self.state.items():
            x = xs.get(vid)
            if x is None:
                continue
            ng = s["next_g"]
            while ng < len(self.gx) and x >= self.gx[ng] - dv:
                s["v_disp"] = disp[ng]
                s["a_star"] = self.vb + s["c"] * (comply(disp[ng], self.level, self.vb) - self.vb)
                ng += 1
            s["next_g"] = ng
            if s["a"] != s["a_star"]:
                if s["tau"] <= dt:
                    s["a"] = s["a_star"]
                else:
                    s["a"] += dt / s["tau"] * (s["a_star"] - s["a"])
                    if abs(s["a"] - s["a_star"]) < 0.01:
                        s["a"] = s["a_star"]
                sf = s["f"] * s["a"] / self.vb
                if abs(sf - s["pushed"]) > 0.002 and (t - s["t_push"] >= 1.0 or s["a"] == s["a_star"]):
                    try:
                        self.k.vehicle.setSpeedFactor(vid, sf)
                        s["pushed"], s["t_push"] = sf, t
                    except Exception:
                        pass

    def mean_f(self) -> float:
        if not self.state:
            return self.cfg.drv.mu_f
        return float(np.mean([s["f"] for s in self.state.values()]))

    def mean_anchor(self, vids):
        """Mean anchor a (m/s) over the given vehicles (those still in state); None if none."""
        vals = [self.state[v]["a"] for v in vids if v in self.state]
        return float(np.mean(vals)) if vals else None

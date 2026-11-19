"""
CAZP (Section 3 of the paper) and the benchmarks that act on individual vehicles or lanes: VSL-L, the zipper-merge sign
and CV early merge. Policy is called at every simulation step (sim.core.simulation.run, on_step) and acts through the
TraCI/libsumo interface. Distances that refer to a lane end are measured to the end of the lane concerned:

* closing lane l ends at x_l, d_l = x_l - x. Pacing zone, recruitment, reach, lateral move-in, release, early-merge
  advisories, the zipper-sign zone and the VSL-L signs of lane l refer to x_l;
* activation (Eq. 2) and queue tail refer to the final lane end x_drop; the pacing zone of lane l reaches from its
  lane end to 100 m upstream of the queue tail, Z_l = clip(x_l - x_tail + 100, zmin, zmax), growing at once and
  shrinking by at most 50 m per update (Eq. 3);
* coverage (Eqs. 4-5): at least `depth` pacers per closing lane, in-lane recruitment within the reach, otherwise a
  lateral move-in from the lane lat_from;
* a pacer in lane l is coupled to the lane into which l merges (ref): v_ref is the mean speed of the vehicles in that
  lane within R of the pacer (Eq. 6), v_cap = max(alpha v_ref, v_ref - cred, v_min) (Eq. 7); cred is the bound of the
  hold in km/h (20 km/h; 10 and 30 km/h in the blocks bound_I and bound_II_III of experiments.py);
* the alignment rule of the lead pacers (Eq. 8) applies only when the closing lanes end at the same point;
* the advisory is executed with a latency and a rate limit (Eq. 9); CAZP-DC adds a display step, rounding and an
  execution factor.

Parameters are those of DEFAULTS, overridden by the policy dict of a run specification (runs.S). Names in the code:
`tie` is the speed cap of Eq. (7), `hold` the lane keeping of the pacers, `form` the alignment rule of Eq. (8), `Zmax`
the reach R_max of Eq. (5) and `zmax_dyn` the largest pacing zone Z_max.
"""
from __future__ import annotations
import collections
import numpy as np

from core.driver_response import comply         # compliance levels, shared with VSL-S (sim on the path: runs.py)

DEFAULTS = dict(pol="NC", p_a=0.75, zone=700.0, rel=200.0, alpha=0.8, v_min=3.0, R=75.0, T=2.0,
                v_on=50.0, v_off=70.0, smooth=0.15, hold=True, tie=True, lanes="all", n_max=0,
                b_adv=1.5, reach=False, Zmax=1500.0, lateral=False, stage=500.0, lat_T=12.0,
                q_kmh=0.0, tau=0.0, err=0.0,
                em=False, D_adv=2500.0, T_em=5.0,
                zone_mode="fixed", zmin=700.0, zmax_dyn=1500.0, zone_trig=700.0,
                form=False, form_tol=30.0, form_k=0.05, form_min=0.5, form_max=300.0,
                depth=1,                       # coverage depth: minimum number of pacers per closing lane
                zms=0.0, zms_d=1500.0,
                vsll=0.0, vsll_gx=(1500.0, 1000.0, 500.0), vsll_T=10.0, vsll_min=10.0, vsll_q=10.0,
                vsll_tau=8.0, vsll_sig=0.5,
                vsll_comp=None,                # compliance level with the lane limits (comply); None: the limit itself
                cred=20.0,                    # bound of the hold below the reference lane (km/h)
                nohop=False, nohop_m=1000.0,   # simulator check: no speed-gain lane changes in the closing lanes
                cell=100.0, q_thr=50.0, D_s=2000.0)
FIXED = dict(adapt=False, zms_dyn=2)   # accepted only with these values: options of the design study that are part
                                       # of the run keys (zms_dyn = 2: the zipper sign is shown only during backups)


class Policy:
    def __init__(self, par, seed, lay, dt):
        self.p = dict(DEFAULTS)
        for k_, v in par.items():
            if k_ in FIXED:
                if v != FIXED[k_]:
                    raise ValueError(f"option {k_}={v!r} is not supported")
                continue
            if k_ not in DEFAULTS:
                raise ValueError(f"unknown policy parameter {k_!r}")
            self.p[k_] = v
        self.lay, self.seed, self.x_drop, self.dt = lay, seed, lay.x_drop, dt
        self.comp = {}
        self.next, self.next_em, self.next_zm, self.next_rec = 0.0, 0.0, 0.0, 0.0
        self.state = {}           # active pacers: vid -> dict(cap, tgt, lane, sent, q, th, f, t0)
        self.moving = {}          # vid -> (target lane, t_end): asked to move into an uncovered closing lane
        self.vq, self.on = None, False
        self.zone_s = {}          # closing lane -> smoothed pacing-zone length (m)
        self.log = collections.Counter()
        self.held = set()         # vehicle numbers that were pacers
        self.lane1000 = {}        # vehicle number -> lane when first seen within 1000 m of the final lane drop
        self.rng = np.random.default_rng(seed * 7919 + 13)
        self.closing, self.open = lay.closing, lay.open
        self.xe = dict(lay.ends)                             # closing lane -> x of its lane end
        self.adj, self.lat_from = dict(lay.ref), dict(lay.lat_from)
        self.step = {ln: -1 for ln in self.closing}          # every closure is on the median side
        self.shift = {ln: self.x_drop - x for ln, x in self.xe.items()}
        self.x_lane = np.full(lay.n_up, self.x_drop)         # x of the end of each lane (open lanes: x_drop)
        for ln, x in self.xe.items():
            self.x_lane[ln] = x
        grp = collections.defaultdict(list)                  # closing lanes sharing a lane end: one sign group
        for ln in self.closing:
            grp[self.xe[ln]].append(ln)
        self.groups = {x: tuple(lns) for x, lns in sorted(grp.items(), reverse=True)}
        self.grp_of = {ln: self.xe[ln] for ln in self.closing}
        L = self.p["lanes"]
        self.paced = self.closing if L == "all" else ((self.closing[0],) if L == "inner" else (self.closing[-1],))
        self.zs = set()                                      # drivers following the zipper-merge sign
        self.zq, self.zon = None, False                      # zipper sign: smoothed open-lane speed, shown or not
        self.zf, self.zd = set(), {x: {} for x in self.groups}   # followers (numbers); distances near each sign
        self.vl = dict(on=False, vq=None, next_tr=0.0, next=0.0, disp=None)   # VSL-L signs
        self.vs = {}                                         # VSL-L driver response: vid -> state
        self.next_nh, self.nh = 0.0, set()                   # simulator check (nohop)

    def _dist(self, V):
        """Distance of every vehicle to the end of its own lane (open lanes: to the final lane drop)."""
        return self.x_lane[np.clip(V.lane, 0, len(self.x_lane) - 1)] - V.x

    def _active(self, V, state):
        """CAZP activation rule on the open lanes in the last zone_trig metres; returns (smoothed speed, on)."""
        p = self.p
        d = self.x_drop - V.x
        o = V.on_up & (d > 100.0) & (d <= p["zone_trig"]) & np.isin(V.lane, self.open)
        v_now = float(V.v[o].mean()) if o.any() else 100.0 / 3.6
        vq = v_now if state[0] is None else (1 - p["smooth"]) * state[0] + p["smooth"] * v_now
        return vq, (vq < p["v_on"] / 3.6) or (state[1] and vq < p["v_off"] / 3.6)

    def _zms(self, t, k, V):
        """Zipper-merge sign, zms_d before the end of each closing lane, shown only while the activation rule is on: a
        driver in the lane who passes the sign while it is shown follows it with probability zms, keeps the lane to its
        release line (rel) and then merges."""
        if t < self.next_zm - 1e-9:
            return
        self.next_zm = t + 2.0
        p = self.p
        dl = self._dist(V)
        m = V.on_up & np.isin(V.lane, self.closing) & (dl <= p["zms_d"]) & (dl > p["rel"])
        self.zq, self.zon = self._active(V, (self.zq, self.zon))
        if not self.zon:
            m = np.zeros_like(m)
            self.zf.clear()
        else:
            for i in np.where(m)[0]:
                n = int(V.num[i])
                if self.zd[self.grp_of[int(V.lane[i])]].get(n, 0.0) > p["zms_d"] and \
                        np.random.default_rng(self.seed * 7777 + n).random() < p["zms"]:
                    self.zf.add(n)
        for x in self.groups:
            dg = x - V.x
            near = V.on_up & (dg > p["zms_d"]) & (dg <= p["zms_d"] + 200.0)
            self.zd[x] = {int(V.num[i]): float(dg[i]) for i in np.where(near)[0]}
        m = m & np.isin(V.num, np.fromiter(self.zf, int, len(self.zf)))
        self.zf = {int(V.num[i]) for i in np.where(m)[0]}
        now = {V.ids[i] for i in np.where(m)[0]}
        for vid in now - self.zs:
            try:
                k.vehicle.setLaneChangeMode(vid, 512)
            except Exception:
                pass
        for vid in self.zs - now:
            if vid in V.idx:
                try:
                    k.vehicle.setLaneChangeMode(vid, 1621)
                except Exception:
                    pass
        self.log["zms_n"] += len(now - self.zs)
        self.zs = now

    def _vsll(self, t, k, V):
        """VSL-L: lane signs over each closing lane, vsll_gx metres before the end of that lane, show vsll x the mean
        speed of the lane it merges into, in the segment the sign governs (down to the next sign or the release point),
        in vsll_q steps between vsll_min and 100 km/h, renewed every vsll_T s while the activation rule is on, 100 km/h
        otherwise. A driver adopts the value of a sign of the lane being driven (at the compliance level vsll_comp,
        core.driver_response.comply) as the anchor of the desired speed f x a, with a first-order response (log-normal
        time constant, median vsll_tau), as the drivers of VSL-S respond to its sign; signs of other lanes are
        ignored; the anchor returns to 100 km/h at the release point of the lane or when the driver leaves the closing
        lanes."""
        p, L, vb = self.p, self.vl, 100.0 / 3.6
        gx = list(p["vsll_gx"]) + [p["rel"]]
        ns = len(gx) - 1
        if t >= L["next_tr"] - 1e-9:                                 # activation: the CAZP rule
            L["next_tr"] = t + 2.0
            L["vq"], L["on"] = self._active(V, (L["vq"], L["on"]))
            self.log["vsll_on_s"] += 2.0 * int(L["on"])
        if L["disp"] is None or t >= L["next"] - 1e-9:                # displayed values, per sign group
            L["next"] = t + p["vsll_T"]
            disp = {}
            for x, lanes in self.groups.items():
                dg = x - V.x
                adj = sorted({self.adj[c] for c in lanes})
                vals = []
                for j in range(ns):
                    v = 100.0
                    if L["on"]:
                        m = V.on_up & (dg <= gx[j]) & (dg > gx[j + 1]) & np.isin(V.lane, adj)
                        if m.any():
                            q = p["vsll_q"]
                            vm = float(V.v[m].mean())
                            v = float(np.clip(q * round(p["vsll"] * vm * 3.6 / q), p["vsll_min"], 100.0))
                            v = max(v, min(100.0, q * float(np.ceil((vm * 3.6 - p["cred"]) / q - 1e-9))))
                        self.log["vsll_post_sum"] += v
                        self.log["vsll_post_n"] += 1
                    vals.append(v / 3.6)
                disp[x] = vals
            L["disp"] = disp
        x0 = min(self.groups) - gx[0] - 60.0
        cand = {V.ids[i] for i in np.where(V.on_up & (V.x >= x0))[0]}
        cand |= {v for v, s in self.vs.items() if s["a"] != vb or s["a_star"] != vb}
        dt = self.dt
        for vid in cand:
            i = V.idx.get(vid)
            if i is None:
                self.vs.pop(vid, None)
                continue
            s = self.vs.get(vid)
            if s is None:
                try:
                    f = float(k.vehicle.getSpeedFactor(vid))
                except Exception:
                    continue
                s = self.vs[vid] = dict(f=f, a=vb, a_star=vb, seg={x: 0 for x in self.groups}, cl=False, pushed=f,
                                        t_push=-1e9,
                                        tau=float(np.random.default_rng(self.seed * 6151 + int(V.num[i])).lognormal(
                                            np.log(p["vsll_tau"]), p["vsll_sig"])) if p["vsll_tau"] > 0 else 0.0)
            ln = int(V.lane[i])
            cl = bool(V.on_up[i]) and ln in self.closing
            if not V.on_up[i]:                                       # past the final lane drop
                s["a_star"] = vb
                for x in s["seg"]:
                    s["seg"][x] = ns + 1
            else:
                for x, lanes in self.groups.items():
                    dg = x - float(V.x[i])
                    released = dg <= gx[-1]
                    seg = ns + 1 if released else sum(1 for j in range(ns) if dg <= gx[j])   # signs passed
                    if seg > s["seg"][x]:
                        s["seg"][x] = seg
                        if cl and ln in lanes:                       # a sign (or the release point) of this lane
                            s["a_star"] = comply(L["disp"][x][seg - 1], p["vsll_comp"], vb) if not released else vb
                if s["cl"] and not cl:
                    s["a_star"] = vb                                 # left the closing lanes
            s["cl"] = cl
            if s["a"] != s["a_star"]:
                if s["tau"] <= dt:
                    s["a"] = s["a_star"]
                else:
                    s["a"] += dt / s["tau"] * (s["a_star"] - s["a"])
                    if abs(s["a"] - s["a_star"]) < 0.01:
                        s["a"] = s["a_star"]
                sf = s["f"] * s["a"] / vb
                if abs(sf - s["pushed"]) > 0.002 and (t - s["t_push"] >= 1.0 or s["a"] == s["a_star"]):
                    try:
                        k.vehicle.setSpeedFactor(vid, sf)
                        s["pushed"], s["t_push"] = sf, t
                    except Exception:
                        pass
                if s["a_star"] < vb:
                    self.log["vsll_veh_s"] += dt

    def _nohop(self, t, k, V):
        """Simulator check: no speed-gain lane changes for vehicles in a closing lane within nohop_m of the end of that
        lane (pacers keep their own lane-change mode)."""
        if t < self.next_nh - 1e-9:
            return
        self.next_nh = t + 2.0
        dl = self._dist(V)
        m = V.on_up & np.isin(V.lane, self.closing) & (dl <= self.p["nohop_m"])
        now = set()
        for i in np.where(m)[0]:
            vid = V.ids[i]
            now.add(vid)
            if vid in self.state:
                continue
            try:
                k.vehicle.setLaneChangeMode(vid, 1605)       # 1621 without the speed-gain bits
            except Exception:
                pass
        for vid in self.nh - now:
            if vid in V.idx and vid not in self.state:
                try:
                    k.vehicle.setLaneChangeMode(vid, 1621)
                except Exception:
                    pass
        self.nh = now

    def is_comp(self, num):
        """Acceptance, drawn once per vehicle number (nested along acceptance and penetration)."""
        c = self.comp.get(num)
        if c is None:
            c = bool(np.random.default_rng(self.seed * 1000003 + int(num)).random() < self.p["p_a"])
            self.comp[num] = c
        return c

    def _release(self, k, vid, alive):
        if alive:
            try:
                k.vehicle.setMaxSpeed(vid, 50.0)
                k.vehicle.setLaneChangeMode(vid, 1621)
            except Exception:
                pass
        self.state.pop(vid, None)

    def _em(self, t, k, V):
        """Early-merge advisory: accepting CVs in a closing lane, between D_adv and 300 m before the end of that lane,
        are asked to change towards the open lanes."""
        if t < self.next_em - 1e-9:
            return
        self.next_em = t + self.p["T_em"]
        dl = self._dist(V)
        m = V.is_cv & V.on_up & np.isin(V.lane, self.closing) & (dl <= self.p["D_adv"]) & (dl > 300.0)
        for i in np.where(m)[0]:
            if not self.is_comp(V.num[i]):
                continue
            try:
                k.vehicle.changeLane(V.ids[i], int(V.lane[i]) + self.step[int(V.lane[i])], self.p["T_em"] + 1.0)
                self.log["em_req"] += 1
            except Exception:
                pass

    def _record(self, t, V):
        if t < self.next_rec - 1e-9:
            return
        self.next_rec = t + 1.0
        m = V.on_up & (self.x_drop - V.x <= 1000.0)
        for i in np.where(m)[0]:
            n = int(V.num[i])
            if n not in self.lane1000:
                self.lane1000[n] = int(V.lane[i])

    def _queue_tail(self, V, d):
        """Distance of the queue tail from the final lane drop: last 100-m bin (one free bin tolerated) in which the
        open lanes are slower than 50 km/h, scanning upstream from the lane drop."""
        c, nb = self.p["cell"], int(round(self.p["D_s"] / self.p["cell"]))
        m = V.on_up & np.isin(V.lane, self.open) & (d > 0) & (d <= self.p["D_s"])
        if not m.any():
            return 0.0
        b = (d[m] // c).astype(int)
        n = np.bincount(b, minlength=nb)[:nb]
        s = np.bincount(b, weights=V.v[m], minlength=nb)[:nb]
        slow = (n > 0) & (s < n * (self.p["q_thr"] / 3.6))
        tail, gap = 0.0, 0
        for j in range(nb):
            if slow[j]:
                tail, gap = (j + 1) * c, 0
            else:
                gap += 1
                if gap > 1:
                    break
        return tail

    def __call__(self, t, k, ctrl, pm, V):
        p = self.p
        if not len(V.ids):
            return
        self._record(t, V)
        if p["nohop"]:
            self._nohop(t, k, V)
        if p["em"]:
            self._em(t, k, V)
        if p["zms"] > 0:
            self._zms(t, k, V)
        if p["vsll"] > 0:
            self._vsll(t, k, V)
        if p["pol"] != "ZP":
            return
        if p["tie"]:                                             # every step: execution of the advisory
            for vid, s in self.state.items():
                while s["q"] and s["q"][0][0] <= t + 1e-9:      # latency elapsed
                    s["tgt"] = s["q"].pop(0)[1]
                if s["tgt"] is None:
                    continue
                new = s["tgt"] if (p["b_adv"] <= 0 or s["tgt"] >= s["cap"]) else max(s["tgt"], s["cap"] - p["b_adv"] * self.dt)
                if abs(new - s["cap"]) > 0.02 or not s["sent"]:
                    s["cap"], s["sent"] = new, True
                    try:
                        k.vehicle.setMaxSpeed(vid, max(new, 0.5))
                    except Exception:
                        pass
        if t < self.next - 1e-9:
            return
        self.next = t + p["T"]
        d = self.x_drop - V.x
        dl = self._dist(V)
        dyn = p["zone_mode"] == "queue"
        o = V.on_up & (d > 100.0) & (d <= (p["zone_trig"] if dyn else p["zone"])) & np.isin(V.lane, self.open)
        v_now = float(V.v[o].mean()) if o.any() else 100.0 / 3.6
        self.vq = v_now if self.vq is None else (1 - p["smooth"]) * self.vq + p["smooth"] * v_now
        on = (self.vq < p["v_on"] / 3.6) or (self.on and self.vq < p["v_off"] / 3.6)
        self.on = on
        self.log["on_s"] += int(on) * p["T"]
        Z = {}
        if dyn:
            tail = self._queue_tail(V, d)
            for ln in self.closing:                              # per lane: from its lane end to the queue tail
                zt = min(max(p["zmin"], tail + 100.0 - self.shift[ln]), p["zmax_dyn"])
                zs = self.zone_s.get(ln)
                Z[ln] = self.zone_s[ln] = zt if zs is None else max(zt, zs - 50.0)   # grows at once, shrinks slowly
            for ln in self.closing:
                if self.shift[ln] == 0.0:
                    self.log["zone_sum"] += Z[ln] * int(on)
                    break
            for ln in self.closing:
                if self.shift[ln] != 0.0:
                    self.log[f"zone{ln}_sum"] += Z[ln] * int(on)
        else:
            Z = {ln: p["zone"] for ln in self.closing}
        Zv = np.zeros(len(self.x_lane))
        for ln in self.closing:
            Zv[ln] = Z[ln]
        Zv = Zv[np.clip(V.lane, 0, len(self.x_lane) - 1)]
        for vid in list(self.state):
            i = V.idx.get(vid)
            gone = i is None or not V.on_up[i] or dl[i] <= p["rel"] or V.lane[i] not in self.paced
            if gone or not on:
                self._release(k, vid, i is not None)
        for vid in list(self.moving):
            i = V.idx.get(vid)
            if i is None or not on or t > self.moving[vid][1] or self.xe[self.moving[vid][0]] - V.x[i] <= p["rel"] + 30.0:
                del self.moving[vid]
        if not on:
            return

        def ok(i):
            vid = V.ids[i]
            return vid not in self.state and self.is_comp(V.num[i])

        def recruit(i, why):
            vid = V.ids[i]
            th = 0.0 if p["tau"] <= 0 else float(self.rng.lognormal(np.log(p["tau"]), 0.5))
            f = 1.0 if p["err"] <= 0 else float(self.rng.uniform(1 - p["err"], 1 + p["err"]))
            self.state[vid] = dict(cap=float(V.v[i]), tgt=None, lane=int(V.lane[i]), sent=False, q=[], th=th, f=f, t0=t)
            self.held.add(int(V.num[i]))
            self.log["pacers"] += 1
            self.log["rec_" + why] += 1
            self.moving.pop(vid, None)
            if p["hold"]:
                try:
                    k.vehicle.setLaneChangeMode(vid, 512)
                except Exception:
                    pass

        def full():                                              # cap n_max on simultaneously active pacers
            return bool(p["n_max"]) and len(self.state) >= p["n_max"]

        m = V.is_cv & V.on_up & np.isin(V.lane, self.paced) & (dl <= Zv) & (dl > p["rel"] + 30.0)
        for i in np.where(m)[0][np.argsort(-V.x[m])]:
            if not ok(i):
                continue
            if full():
                break
            recruit(i, "zone")
        for vid, (L, _) in list(self.moving.items()):
            i = V.idx[vid]
            if V.lane[i] == L and vid not in self.state and not full():
                recruit(i, "lateral")
        covered = collections.Counter(int(V.lane[V.idx[vid]]) for vid in self.state)
        for L in self.paced:
            if covered[L] >= p["depth"]:
                continue
            if full():
                break
            dL, ZL = self.xe[L] - V.x, Z[L]                      # positions relative to the end of lane L
            done = False
            if p["reach"]:
                c = V.is_cv & V.on_up & (V.lane == L) & (dL > ZL) & (dL <= max(p["Zmax"], ZL + p["stage"]))
                for i in np.where(c)[0][np.argsort(-V.x[c])]:
                    if ok(i):
                        recruit(i, "reach")
                        done = True
                        break
            if not done and p["lateral"] and not any(v[0] == L for v in self.moving.values()):
                src = self.lat_from[L]
                c = V.is_cv & V.on_up & (V.lane == src) & (dL > p["rel"] + 150.0) & (dL <= ZL + p["stage"])
                best = None
                for i in np.where(c)[0][np.argsort(V.x[c])]:
                    vid = V.ids[i]
                    if vid in self.moving or not self.is_comp(V.num[i]):
                        continue
                    if vid in self.state:
                        n_same = sum(1 for u in self.state if V.lane[V.idx[u]] == src)
                        if n_same < 2:
                            continue
                    best = i
                    break
                if best is not None:
                    vid = V.ids[best]
                    if vid in self.state:
                        self._release(k, vid, True)
                    self.moving[vid] = (L, t + p["lat_T"])
                    self.log["lat_req"] += 1
        for vid, (L, _) in self.moving.items():
            try:
                k.vehicle.changeLane(vid, L, p["T"] + 1.0)
            except Exception:
                pass
        self.log["act_max"] = max(self.log["act_max"], len(self.state))
        lead = None                                              # alignment rule (Eq. 8) of the two lead pacers
        if p["form"] and len(self.paced) == 2 and self.lay.same_end:
            first = {}
            for vid in self.state:
                i = V.idx[vid]
                L = int(V.lane[i])
                if L not in first or d[i] < first[L][0]:
                    first[L] = (float(d[i]), i)
            if len(first) == 2:
                (da, ia), (db, ib) = sorted(first.values())
                if p["form_tol"] < db - da < p["form_max"]:
                    lead = (V.ids[ia], float(V.v[ib]), db - da)
                    self.log["form_s"] += p["T"]
        refm = {ln: V.on_up & (V.lane == ln) for ln in set(self.adj.values())}
        lanes_cov = set()
        for vid, s in self.state.items():
            i = V.idx[vid]
            L = int(V.lane[i])
            lanes_cov.add(L)
            near = refm[self.adj.get(L, self.open[-1])] & (np.abs(V.x - V.x[i]) < p["R"])
            v_ref = float(V.v[near].mean()) if near.any() else self.vq
            tg = max(p["alpha"] * v_ref, p["v_min"] / 3.6)
            if lead is not None and vid == lead[0]:
                tg = max(p["form_min"] * v_ref, p["v_min"] / 3.6,
                         min(tg, lead[1] - p["form_k"] * (lead[2] - p["form_tol"])))
            tg = max(tg, v_ref - p["cred"] / 3.6)
            if p["q_kmh"] > 0:
                tg = max(p["q_kmh"], p["q_kmh"] * round(tg * 3.6 / p["q_kmh"])) / 3.6      # displayed value
            tg *= s["f"]                                                                  # what the driver drives
            s["q"].append((t + s["th"], tg))
            self.log["dv_sum"] += (float(V.v[i]) - v_ref) * 3.6
            self.log["dv_n"] += 1
        self.log["pacer_s"] += len(self.state) * p["T"]
        for ln in self.closing:
            self.log[f"cov{ln}_s"] += int(ln in lanes_cov) * p["T"]

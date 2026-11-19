"""
Online queue-tail estimator from segment detectors for the controllers in sim/: it supplies the queue tail used by CV
speed pacing and the segment speeds from which VSL-S takes the approach speed. CAZP estimates its own queue tail in
policy.py; VSL-L and the zipper-merge sign use the activation rule of CAZP.

* Segments S_i of length seg_det cover [x_drop - seg_range, x_drop] (cfg.geo.segments). V_i(t) is the
  tail_win_s trailing mean of the 1-s space-mean speed of the vehicles inside S_i; an EMPTY segment counts as
  v_base (a standing queue produces no loop crossings).
* Congestion flag with hysteresis: c_i = 1 if V_i < tail_v_on (45 km/h), 0 if V_i > tail_v_off (55 km/h),
  else unchanged (brackets the 0.5 v_ff rule of metrics.queue_from_grid).
* Queue exists if any of the last 3 segments is congested (head test). Tail: scanning upstream from the head,
  the upstream-most CONGESTED segment i such that the share of congested segments between i and the head is
  >= tail_fill (same fill rule as the metric) -> x_tail_raw = x0_i. Detached waves upstream fail the fill test.
* Smoothing: median of the last tail_smooth_n raw estimates (estimate() is called on the 5-s clock: t, t-5, t-10).

update_1s(t, x, v, on_up) every 1 s with the positions (m) / speeds (m/s) of all vehicles and the upstream mask;
estimate(t) -> dict(x_tail: float | None, queued: bool, n_cong: int, head_ok: bool) (also in .last).
"""
from __future__ import annotations
import numpy as np

from config import KMH

HEAD_SEGS = 3


class TailEstimator:
    def __init__(self, cfg):
        g, c = cfg.geo, cfg.ctrl
        segs = np.asarray(g.segments, dtype=float)
        self.x0, self.x1 = segs[:, 0], segs[:, 1]
        self.n = len(segs)
        self.seg = float(g.seg_det)
        self.v0 = g.v_base
        self.v_on, self.v_off = c.tail_v_on_kmh * KMH, c.tail_v_off_kmh * KMH
        self.fill, self.n_smooth = float(c.tail_fill), max(1, int(c.tail_smooth_n))
        self.win = max(1, int(round(c.tail_win_s)))
        self._hist = np.full((self.win, self.n), self.v0)     # ring buffer of 1-s segment speeds
        self._ptr, self._n_hist = 0, 0
        self.seg_v = np.full(self.n, self.v0)                 # trailing mean per segment (m/s)
        self.cong = np.zeros(self.n, dtype=bool)
        self._raw = []                                        # last n_smooth raw estimates (nan = no queue)
        self.last = dict(x_tail=None, queued=False, n_cong=0, head_ok=False)

    # ------------------------------------------------------------ 1-s clock
    def update_1s(self, t, x, v, on_up=None):
        x = np.asarray(x, dtype=float)
        v = np.asarray(v, dtype=float)
        m = (x >= self.x0[0]) & (x < self.x1[-1])
        if on_up is not None:
            m &= np.asarray(on_up, dtype=bool)
        idx = ((x[m] - self.x0[0]) // self.seg).astype(int)
        cnt = np.bincount(idx, minlength=self.n)[:self.n]
        vs = np.bincount(idx, weights=v[m], minlength=self.n)[:self.n]
        row = np.where(cnt > 0, vs / np.maximum(cnt, 1), self.v0)
        self._hist[self._ptr] = row
        self._ptr = (self._ptr + 1) % self.win
        self._n_hist = min(self._n_hist + 1, self.win)
        self.seg_v = self._hist[:self._n_hist].mean(axis=0)

    # ------------------------------------------------------------ 5-s clock
    def _raw_tail(self):
        """(x_tail_raw | nan, head_ok) from the current congestion flags."""
        c = self.cong
        head_ok = bool(c[-HEAD_SEGS:].any())
        if not head_ok:
            return np.nan, False
        h = int(np.where(c)[0].max())
        n_c, tl = 0, h
        for i in range(h, -1, -1):
            if c[i]:
                n_c += 1
                if n_c >= self.fill * (h - i + 1):
                    tl = i
        return float(self.x0[tl]), True

    def estimate(self, t):
        V = self.seg_v
        self.cong = np.where(V < self.v_on, True, np.where(V > self.v_off, False, self.cong))
        raw, head_ok = self._raw_tail()
        self._raw = (self._raw + [raw])[-self.n_smooth:]
        vals = sorted(r for r in self._raw if np.isfinite(r))
        x_tail = float(vals[len(vals) // 2]) if head_ok and vals else None   # element median (upper for even n)
        self.last = dict(x_tail=x_tail, queued=x_tail is not None, n_cong=int(self.cong.sum()), head_ok=head_ok)
        return dict(self.last)

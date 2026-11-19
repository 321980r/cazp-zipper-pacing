"""
Figures and key numbers of the paper from the stored run records (results/runs, or the folder in CAZP_RUNS).

    python analysis.py      -> figures/Fig2-Fig7 (.png, .pdf) and results/key_numbers.json (the numbers of the paper)

CAZP_OUT sets another output folder (figures/ and results/ are created inside it). Effects are seed-wise paired
differences against no control or another strategy with 95% t-intervals (summ); p-values are Holm-adjusted where the
paper says so (holm); the heat maps use medians of the paired changes and Wilcoxon signed-rank tests.
"""
from __future__ import annotations
import json
import math
import os
import pickle
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                     # noqa: E402
import matplotlib.text as mtext                                     # noqa: E402
from matplotlib.lines import Line2D                                 # noqa: E402
from matplotlib.colors import TwoSlopeNorm                         # noqa: E402
from matplotlib.patches import Rectangle                           # noqa: E402
from scipy import stats                                             # noqa: E402
from scipy.optimize import curve_fit                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import runs as R                                                    # noqa: E402
import experiments as E                                             # noqa: E402
from closures import CLOSURES                                       # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)   # cells without vehicles
OUT_ROOT = os.environ.get("CAZP_OUT", HERE)
OUT = os.path.join(OUT_ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT_ROOT, "results"), exist_ok=True)
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5, "axes.labelsize": 10.5, "legend.fontsize": 9,
                     "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 110,
                     "font.family": "Times New Roman", "mathtext.fontset": "stix", "font.weight": "normal",
                     "axes.titleweight": "normal"})
plt.rcParams.update({"legend.fontsize": 8.4, "axes.axisbelow": True})

# ------------------------------------------------------------------ strategies, labels, indicators
COL = {"NC": "#6e6e6e", "VSL1": "#3b75af", "VSLL": "#17a2b8", "PACE": "#8e6bb3", "EM": "#c07a2c", "ZMSD": "#d4a017", "VSL1_ZMSD": "#5f9ea0", "VSLL_ZMSD": "#8c564b", "CAZP": "#1b7a3a", "CAZP_VSL1": "#b5367a"}
LAB = {"NC": "No control", "VSL1": "VSL-S", "VSLL": "VSL-L", "PACE": "CV speed pacing", "EM": "CV early merge",
       "ZMSD": "Zipper-merge sign", "VSL1_ZMSD": "VSL-S + zipper sign",
       "VSLL_ZMSD": "VSL-L + zipper sign", "CAZP": "CAZP (proposed)", "CAZP_VSL1": "CAZP + VSL-S"}
ARMS = ["VSL1", "VSLL", "PACE", "EM", "ZMSD", "VSL1_ZMSD", "VSLL_ZMSD", "CAZP", "CAZP_VSL1"]
L3 = R.L3
DCOL = ("#9ecae1", "#4292c6", "#08519c")
METRICS = ["q_queue", "delay", "queue_mean", "queue_max", "hb", "svar", "stops", "co2_g", "nox_g"]   # Section 4.3
SCORE = [("q_queue", "Queue\ndischarge", +1, "Efficiency"), ("delay", "Delay", -1, "Efficiency"),
         ("queue_mean", "Mean\nqueue", -1, "Efficiency"), ("queue_max", "Max.\nqueue", -1, "Efficiency"),
         ("hb", "Hard\nbraking", -1, "Safety"), ("svar", "Speed\nvariance", -1, "Safety"), ("stops", "Stops", -1, "Safety"),
         ("co2_g", "CO2", -1, "Emissions"), ("nox_g", "NOx", -1, "Emissions")]
GROUP_LAB = {"core": "Component removed", "variant": "Design variant", "control": "Simulator check"}   # Fig. 5


# ------------------------------------------------------------------ paired statistics
def val(r, m):
    """Indicator m of a run record."""
    return float(r.get(m, np.nan))


def summ(x, base=None):
    """Mean, 95% t-interval, median, paired t test (p), Wilcoxon signed-rank test (pw), number of negative differences
    (wins) and the change relative to the mean of base (rel, rlo, rhi; %)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return dict(n=n, mean=np.nan, lo=np.nan, hi=np.nan, half=np.nan, median=np.nan, p=np.nan, pw=np.nan,
                    wins=0, base=np.nan, rel=np.nan, rlo=np.nan, rhi=np.nan)
    m, sd = x.mean(), x.std(ddof=1)
    h = stats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    p = stats.ttest_1samp(x, 0.0).pvalue if sd > 0 else (1.0 if m == 0 else 0.0)     # identical runs: p = 1
    try:
        pw = stats.wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    except ValueError:
        pw = np.nan
    b = float(np.nanmean(base)) if base is not None and len(base) else np.nan
    return dict(n=n, mean=m, lo=m - h, hi=m + h, half=h, median=float(np.median(x)), p=float(p), pw=float(pw),
                wins=int((x < 0).sum()), base=b, rel=100 * m / b if b else np.nan,
                rlo=100 * (m - h) / b if b else np.nan, rhi=100 * (m + h) / b if b else np.nan)


def pairs(D, arm, dem, m, ref="NC", seeds=None, Dref=None):
    """Seed-wise differences arm - ref of indicator m at one demand, and the values of ref."""
    Dref = Dref if Dref is not None else D
    d, b = [], []
    for (a, dm, sd), r in D.items():
        if a == arm and dm == dem and (seeds is None or sd in seeds):
            rr = Dref.get((ref, dem, sd))
            if rr is not None:
                d.append(val(r, m) - val(rr, m))
                b.append(val(rr, m))
    return np.array(d), np.array(b)


def paired(D, arm, dem, m, **kw):
    """summ of the seed-wise differences at one demand."""
    d, b = pairs(D, arm, dem, m, **kw)
    return summ(d, b)


def pooled(D, arm, m, dems=L3, ref="NC"):
    """Paired change pooled over demands: per seed the mean difference over the demands."""
    seeds = sorted({sd for (a, dm, sd) in D if a == arm})
    d, b = [], []
    for sd in seeds:
        xs = [(D.get((arm, dm, sd)), D.get((ref, dm, sd))) for dm in dems]
        if all(x is not None and y is not None for x, y in xs):
            d.append(np.mean([val(x, m) - val(y, m) for x, y in xs]))
            b.append(np.mean([val(y, m) for _, y in xs]))
    return summ(d, b)


def holm(ps):
    """Holm-adjusted p-values."""
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty_like(ps)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def runs_of(D, arm, dem):
    """Run records of an arm at one demand."""
    return [r for (a, d, s), r in D.items() if a == arm and d == dem]


def med_ci(d, base, nb=4000):
    """Median of the paired differences relative to the mean of base (%) with a bootstrap 95% interval."""
    d = np.asarray(d, float)
    if len(d) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(7)
    bs = np.median(d[rng.integers(0, len(d), (nb, len(d)))], axis=1)
    b = float(np.mean(base))
    return 100 * np.median(d) / b, 100 * np.percentile(bs, 2.5) / b, 100 * np.percentile(bs, 97.5) / b


def pacer_burden(a, b):
    """Mean delay change (s/veh) of the vehicles that acted as pacers, against the same vehicles without control."""
    if a is None or b is None or len(a["veh"]["delay"]) != len(b["veh"]["delay"]):
        return np.nan
    dd = a["veh"]["delay"].astype(float) - b["veh"]["delay"].astype(float)
    m = a["veh"]["held"] & np.isfinite(dd)
    return float(dd[m].mean()) if m.any() else np.nan


def main_table(M, KN):
    """Main comparison in the main test bed by demand and pooled over the demands (KN["main"]): mean of every indicator
    (abs) and paired change against no control; delay p-values Holm-adjusted over the nine strategies."""
    groups = [str(d) for d in L3] + ["pooled"]
    pdelay = {g: [(pooled(M, a, "delay") if g == "pooled" else paired(M, a, int(g), "delay"))["p"] for a in ARMS]
              for g in groups}
    KN["main"] = {}
    for g in groups:
        pdl = dict(zip(ARMS, holm(pdelay[g])))
        for a in ARMS:
            sel = [r for (aa, d, s), r in M.items() if aa == a and (g == "pooled" or d == int(g))]
            if not sel:
                continue
            for m in METRICS:
                mu = float(np.nanmean([val(r, m) for r in sel]))
                s = pooled(M, a, m) if g == "pooled" else paired(M, a, int(g), m)
                KN["main"].setdefault(f"{g}|{a}", {})[m] = dict(mean=s["mean"], lo=s["lo"], hi=s["hi"], rel=s["rel"],
                                                               median=s["median"], wins=s["wins"], n=s["n"], p=s["p"],
                                                               pw=s["pw"], abs=mu)
            KN["main"][f"{g}|{a}"]["p_holm_delay"] = float(pdl[a])


# ------------------------------------------------------------------ figures: common settings
MINUS = "−"
SCORE_LAB = {"co2_g": "CO$_2$", "nox_g": "NO$_x$"}
STY = {"NC": dict(color=COL["NC"], ls="--", lw=1.2), "VSL1": dict(color=COL["VSL1"], ls="-", lw=1.0),
       "VSLL": dict(color=COL["VSLL"], ls="-", lw=1.4), "PACE": dict(color=COL["PACE"], ls="-.", lw=1.0),
       "CAZP": dict(color=COL["CAZP"], ls="-", lw=1.9), "CAZP_VSL1": dict(color=COL["CAZP_VSL1"], ls="-", lw=1.3)}


def mt(s):
    """Typographic minus signs in a numeric label."""
    return s.replace("-", MINUS)


MAIN = E.MAIN
ORDER = E.CONFIGS
NUM = E.NUMBER
NAME = {"seq": "successive closure", "base": "lanes 2 and 3 end together", "l43": "four-to-three closure"}
X3 = {"seq": 675.0, "base": 0.0, "l43": 0.0}                       # end of lane 3 upstream of the final lane end (m)
_cache, _first = {}, {}           # run records by file; block and specification of the first load


def load(block, lay=MAIN):
    """Stored runs of a block in configuration lay (all configurations: lay=None), keyed by (arm, demand, seed); missing
    runs are skipped, and a block whose tuned parameters do not exist yet is empty."""
    out = {}
    try:
        specs = E.BLOCKS[block]()
    except RuntimeError:
        return out
    for s in specs:
        if lay is not None and s["layout"] != lay:
            continue
        p = R.path(s)
        if p not in _cache and os.path.isfile(p):
            with open(p, "rb") as f:
                _cache[p] = pickle.load(f)
            _first[p] = (block, s)
        if p in _cache:
            out[(s["arm"], s["dem"], s["seed"])] = _cache[p]
    return out


def alias(D, src, dst, dems=None, seeds=None):
    """The runs of arm src under the name dst (the base case of a sensitivity family)."""
    return {(dst, d, s): r for (a, d, s), r in D.items() if a == src and (dems is None or d in dems)
            and (seeds is None or s in seeds)}


def dems_of(lay):
    """Peak demands of a configuration."""
    return tuple(E.demands(lay)[0])


def queue_len(r, vthr=50.0, head_cells=3, fill=0.7):
    """Extent of the queue upstream of the final lane end by the head and tail rule of the queue-length indicator
    (sim/core/metrics.py, queue_from_grid), applied to the stored speed grid only: cells slower than vthr (the
    indicator also counts cells denser than 45 veh/km per lane); the head lies in the
    last head_cells cells before the lane end (in the successive closure the last 100-300 m carry the accelerating
    merged traffic), and the tail is the upstream-most slow cell such that at least `fill` of the cells between tail
    and head are slow."""
    g = r["grid_v"].astype(float)
    xd = int(round(R.X_DROP / r["grid_dx"]))
    with np.errstate(invalid="ignore"):
        slow = np.isfinite(g[:, :xd]) & (g[:, :xd] < vthr)
    out = np.zeros(g.shape[0])
    lo = max(0, xd - head_cells)
    for k in range(g.shape[0]):
        row = slow[k]
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
        out[k] = (xd - tl) * r["grid_dx"]
    return out


WIDTH_CM, MAX_H_CM, FS = 15.8, 23.5, 9.0          # journal: below 16 cm wide and 24 cm high; text 9 pt Times New Roman


def _panel(fig, label):
    return next(a for a in fig.axes if a.get_title(loc="left").startswith(label))


def _legend_items(ax):
    leg = ax.get_legend()
    h, lab = leg.legend_handles, [t.get_text() for t in leg.get_texts()]
    leg.remove()
    return h, lab


def _unify_text(fig):
    for t in fig.findobj(mtext.Text):
        if t.get_text():
            t.set_fontsize(FS)
            t.set_fontweight("normal")
            t.set_fontstyle("normal")
            t.set_fontfamily("Times New Roman")


def _fig2(fig):                                       # (j): legend below the bottom row
    h, lab = _legend_items(_panel(fig, "(j)"))
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, frameon=False, handlelength=2.2)


def _fig3(fig):                                       # (a) title in two lines; (b) legend in the free corner
    _panel(fig, "(a)").set_title("(a) Change against no control (%), pooled over 3400-3800 veh/h;\ncolour: green = "
                                 "better, saturating at 50%", loc="left", pad=22)
    b = _panel(fig, "(b)")
    h, lab = _legend_items(b)
    b.legend(h, lab, loc="lower left", frameon=False, labelspacing=0.22, handlelength=1.8, borderaxespad=0.15)


def _fig4(fig):                                       # legends in free corners, label left of the lane-3 line in (b)
    a = _panel(fig, "(a)")
    h, lab = _legend_items(a)
    a.legend(h, lab, loc="lower left", frameon=False, labelspacing=0.22, handlelength=1.6, borderaxespad=0.15)
    b = _panel(fig, "(b)")
    for t in b.texts:
        if t.get_text() == "lane 3 ends":
            x, y = t.get_position()
            t.set_position((x + 50.0, 0.62))
            t.set_ha("right")
    d = _panel(fig, "(d)")
    h, lab = _legend_items(d)
    d.legend(h, lab, loc="lower right", frameon=False, labelspacing=0.3, borderaxespad=0.2)


def _fig5(fig):                                       # calibrated gain named in the caption; one legend below (d)-(f)
    for a in fig.axes:
        for t in list(a.texts):
            if t.get_text().startswith("calibrated"):
                t.remove()
    h, lab = _legend_items(_panel(fig, "(e)"))
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, columnspacing=1.6)


def _fig6(fig):                                       # shorter title of (c); fit formula in the free corner of (d)
    _panel(fig, "(c)").set_title("(c) Delay change (%), 3600 veh/h", loc="left")
    d = _panel(fig, "(d)")
    for t in d.texts:
        if t.get_text().startswith("fit:"):
            t.set_transform(d.transAxes)
            t.set_position((0.98, 0.86))
            t.set_ha("right")
            t.set_va("top")


def _fig7(fig):                                       # legend of (c) below the figure
    h, lab = _legend_items(_panel(fig, "(c)"))
    fig.legend(h, lab, loc="outside lower center", ncol=3, frameon=False, handletextpad=0.3, columnspacing=1.2)


ADJUST = {"Fig2_queue_dynamics": _fig2, "Fig3_performance": _fig3, "Fig4_mechanism_burden": _fig4,
          "Fig5_ablation_gain": _fig5, "Fig6_penetration": _fig6, "Fig7_robustness_configurations": _fig7}


def save(fig, name):
    """Figure at its final size (below 16 x 24 cm; all text 9 pt Times New Roman) as PNG (300 dpi) and PDF."""
    w, h = fig.get_size_inches()
    s = WIDTH_CM / 2.54 / w
    fig.set_size_inches(w * s, min(h * s, MAX_H_CM / 2.54))
    _unify_text(fig)
    if name in ADJUST:
        ADJUST[name](fig)
        _unify_text(fig)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)
    print("figure", name)


def tuned():
    """Tuned parameters of every configuration (results/tuning/tuned.json)."""
    return E.tuned()


# ------------------------------------------------------------------ the main test bed without control (Section 4.1)
def bed_numbers(M, KN):
    """Queue discharge rate, largest 5-min flow before breakdown and delay without control at each demand (KN["bed"])."""
    qd = [np.mean([r["q_queue"] for r in runs_of(M, "NC", d)]) for d in L3]
    q5 = [np.mean([r["q_max5"] for r in runs_of(M, "NC", d)]) for d in L3]
    KN["bed"] = dict(q_queue_NC=dict(zip(map(str, L3), qd)), q_max5_NC=dict(zip(map(str, L3), q5)),
                     delay_NC={str(d): float(np.mean([r["delay"] for r in runs_of(M, "NC", d)])) for d in L3})


# ------------------------------------------------------------------ Fig. 2: queue dynamics (main test bed)
def fig02(M, KN):
    """Fig. 2: (a)-(i) mean speed without control, with CAZP and their difference at the three demands; (j)-(l)
    cumulative counts, queue length and delay distribution at 3800 veh/h."""
    fig = plt.figure(figsize=(7.4, 9.9))
    outer = fig.add_gridspec(2, 1, height_ratios=[3.05, 1.1], hspace=0.42, left=0.105, right=0.985, top=0.955,
                             bottom=0.05)
    top = outer[0].subgridspec(3, 3, hspace=0.24, wspace=0.08)
    axs = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            kw = {} if (i, j) == (0, 0) else dict(sharex=axs[0, 0], sharey=axs[0, 0])
            axs[i, j] = fig.add_subplot(top[i, j], **kw)
    im, imd = ts_maps(M, axs)
    for i, dem in enumerate(L3):
        axs[i, 0].set_ylabel(f"{dem} veh/h\nposition (m)")
    for ax in axs.flat:
        ax.axhline(-X3[MAIN], color="k", lw=0.5, ls=":")
    axs[0, 0].text(1.0, -X3[MAIN] + 40, "end of lane 3", fontsize=7.8, va="bottom")
    map_frame(fig, axs, im, imd, "abcdefghi", 2)
    bot = outer[1].subgridspec(1, 3, wspace=0.42)
    queue_panels(M, KN, [fig.add_subplot(bot[0, j]) for j in range(3)])
    save(fig, "Fig2_queue_dynamics")


# ------------------------------------------------------------------ Fig. 3: overall performance
def fig03(M, KN):
    """Fig. 3: (a) change of the nine indicators against no control, pooled over the demands; (b)-(d) mean speed
    25-40 min; (e)-(g) 5-min outflow."""
    fig = plt.figure(figsize=(7.4, 9.4))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.3, 2.05], hspace=0.28, left=0.095, right=0.985, top=0.95,
                             bottom=0.115)
    ax0 = fig.add_subplot(outer[0])
    pos = ax0.get_position()
    ax0.set_position([0.2, pos.y0, 0.985 - 0.2, pos.height])
    scorecard(M, KN, ax0)
    inner = outer[1].subgridspec(2, 3, hspace=0.5, wspace=0.1)
    axs = np.empty((2, 3), dtype=object)
    for i in range(2):
        for j in range(3):
            axs[i, j] = fig.add_subplot(inner[i, j], **({} if j == 0 else dict(sharey=axs[i, 0])))
    speed_outflow(M, KN, axs)
    for i in range(2):
        for j in (1, 2):
            axs[i, j].tick_params(labelleft=False)
    h1, l1 = axs[0, 0].get_legend_handles_labels()
    axs[0, 0].legend(h1, l1, loc="lower left", fontsize=7.8, frameon=False, handlelength=2.2)
    h2, l2 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h2, l2, loc="lower center", ncol=3, frameon=False, fontsize=8.2, bbox_to_anchor=(0.54, 0.0),
               handlelength=2.2)
    save(fig, "Fig3_performance")


# ------------------------------------------------------------------ Fig. 4: mechanism (main test bed)
def lc_rate(r, a, b):
    """Lane changes from lane a to lane b per 100 m and 1000 vehicles (last 2.5 km before the final lane drop)."""
    return r["lch"][a, b, :] / max(r["q_peak"] * 1500 / 3600, 1) * 1000


def fig04(M, KN):
    """Fig. 4 (no control dashed, CAZP solid): (a) mean speed of each lane, (b) merges out of the closing lanes,
    (c) share of the closing lanes, (d) delay change by vehicle group. Lane 3 is drawn only upstream of its end."""
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.4))
    axs = axs.flat
    dist = 2500 - (np.arange(25) + 0.5) * 100
    xb = (np.arange(25) + 0.5) * 100                                  # lane-change bins: distance to the final lane end
    x3 = X3[MAIN]
    for arm, ls in (("NC", "--"), ("CAZP", "-")):
        runs = runs_of(M, arm, 3800)
        with np.errstate(all="ignore"):
            v = np.nanmean([r["vprof"] for r in runs], axis=0)
            k = np.nanmean([r["kprof"] for r in runs], axis=0)
        for ln, c in zip(range(4), ("#bdbdbd", "#636363", "#fd8d3c", "#e6550d")):
            axs[0].plot(dist, np.where(dist > x3, v[ln], np.nan) if ln == 3 else v[ln], color=c, ls=ls, lw=1.1,
                        label=f"lane {ln}" if arm == "CAZP" else None)
        out = np.array([lc_rate(r, 2, 1) for r in runs]).mean(0)       # lane 2 into lane 1: the only way out
        axs[1].step(xb, out, where="mid", color=COL[arm], lw=1.3, label=LAB[arm])
        axs[2].plot(dist, (np.nan_to_num(k[2]) + np.nan_to_num(k[3])) / np.nansum(k, axis=0), color=COL[arm], ls=ls,
                    lw=1.3, label=LAB[arm])
    for ax in (axs[0], axs[1], axs[2]):
        ax.set_xlim(2000, 0)
        ax.set_xlabel("Distance to the end of lane 2 (m)")
        ax.grid(color="#eee", lw=0.6)
        ax.axvline(x3, color="k", lw=0.5, ls=":")
        bottom = ax is axs[2]                                         # (c): the curves pass the top of the line
        ax.text(x3 - 25, 0.03 if bottom else 0.98, "lane 3 ends", transform=ax.get_xaxis_transform(), fontsize=7.8,
                va="bottom" if bottom else "top")
    axs[0].set_ylabel("Mean speed 25-40 min (km/h)")
    axs[0].set_title("(a) Mean speed by lane", loc="left", fontsize=9.5)
    axs[0].plot([], [], color="#555555", ls="--", lw=1.1, label="no control (dashed)")
    axs[0].legend(frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.42), fontsize=7.8)
    axs[1].set_ylabel("Merges out of closing lanes\n(per 100 m per 1000 veh)")
    axs[1].set_title("(b) Where vehicles leave the closing lanes", loc="left", fontsize=9.5)
    axs[1].legend(frameon=False, fontsize=7.8, loc="upper left")
    axs[2].set_ylabel("Share of vehicles in closing lanes")
    axs[2].set_title("(c) Share of vehicles in the closing lanes", loc="left", fontsize=9.5)
    axs[2].legend(loc="lower left", fontsize=7.8, frameon=False)
    ax = axs[3]
    groups = [("pacers", "pacers"), ("closing", "others in\nlanes 2-3"), ("lane1", "lane 1"),
              ("lane0", "lane 0"), ("all", "all\nvehicles")]
    KN["burden"] = {}
    for di, dem in enumerate(L3):
        per, per_nc = {g: [] for g, _ in groups}, {g: [] for g, _ in groups}
        for sd in range(1, 41):
            a, b = M.get(("CAZP", dem, sd)), M.get(("NC", dem, sd))
            if a is None or b is None or len(a["veh"]["delay"]) != len(b["veh"]["delay"]):
                continue
            dd = a["veh"]["delay"].astype(float) - b["veh"]["delay"].astype(float)
            ln, held = b["veh"]["lane1000"], a["veh"]["held"]
            for g, m in dict(pacers=held, closing=(ln >= 2) & ~held, lane1=(ln == 1) & ~held, lane0=(ln == 0) & ~held,
                             all=np.ones(len(dd), bool)).items():
                m = m & np.isfinite(dd)
                if m.any():
                    per[g].append(float(dd[m].mean()))
                    per_nc[g].append(float(b["veh"]["delay"][m].astype(float).mean()))   # same vehicles without control
        for gi, (g, lab) in enumerate(groups):
            s = summ(per[g], per_nc[g])
            KN["burden"][f"{dem}|{g}"] = s
            if np.isfinite(s["mean"]):
                ax.errorbar(gi + (di - 1) * 0.22, s["mean"], yerr=s["half"], fmt="o", ms=3.5, color=DCOL[di],
                            capsize=1.5, label=f"{dem} veh/h" if gi == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylabel("Delay change per vehicle (s)")
    ax.set_title("(d) Delay change by vehicle group", loc="left", fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    ax.grid(axis="y", color="#eee", lw=0.6)
    fig.tight_layout(w_pad=1.4, h_pad=1.3)
    save(fig, "Fig4_mechanism_burden")
    KN["mechanism_3800"] = mechanism_numbers(M, MAIN)


def ablation_mech(AB, KN, lay=MAIN, key="ablation_mech"):
    """Speed advantage of the closing lanes, lane changes into lane 3, share of the closing lanes and pacers per run
    for every ablation variant at 3800 veh/h."""
    lay_ = CLOSURES[lay]
    per = lambda r: max(r["q_peak"] * 1500 / 3600, 1)                      # noqa: E731
    out = {}
    for a in sorted({k[0] for k in AB}):
        rs = [r for (aa, d, s), r in AB.items() if aa == a and d == 3800]
        o = {}
        for ln, ref in dict(lay_.ref).items():
            sh = 5000.0 - dict(lay_.ends)[ln]
            j = [jj for jj in range(25) if 200 + sh <= 2500 - (jj + 0.5) * 100 <= 700 + sh]
            with np.errstate(all="ignore"):
                o[f"dv_{ln}_{ref}"] = float(np.nanmean([np.nanmean(r["vprof"][ln, j]) - np.nanmean(r["vprof"][ref, j])
                                                        for r in rs]))
        o["into3_per_kveh"] = float(np.mean([r["lch"][2, 3, :].sum() / per(r) * 1000 for r in rs]))
        cl = sorted(dict(lay_.ends))                                     # lanes that close in this configuration
        o["share_550"] = float(np.nanmean([sum(np.nan_to_num(r["kprof"][ln, 19]) for ln in cl)
                                           / np.nansum(r["kprof"][:, 19]) for r in rs]))
        o["pacers_per_run"] = float(np.mean([r["n_pacers"] for r in rs]))
        out[a] = o
    KN[key] = out


def mechanism_numbers(M, lay, dem=None):
    """Speed advantage of each closing lane over the lane it merges into, 200-700 m before the end of the closing
    lane; lane changes; share of the closing lanes; pacers and activation (highest demand)."""
    dem = dem or dems_of(lay)[-1]
    out = {}
    lay_ = CLOSURES[lay]
    for a in ("NC", "CAZP", "VSLL", "ZMSD"):
        rs = runs_of(M, a, dem)
        if not rs:
            continue
        o = {}
        for ln, ref in dict(lay_.ref).items():
            sh = 5000.0 - dict(lay_.ends)[ln]                              # m between this lane end and the final drop
            j = [jj for jj in range(25) if 200 + sh <= 2500 - (jj + 0.5) * 100 <= 700 + sh]
            with np.errstate(all="ignore"):
                o[f"dv_{ln}_{ref}"] = float(np.nanmean([np.nanmean(r["vprof"][ln, j]) - np.nanmean(r["vprof"][ref, j])
                                                        for r in rs]))
        per = lambda r: max(r["q_peak"] * 1500 / 3600, 1)                  # noqa: E731
        o["into3_per_kveh"] = float(np.mean([r["lch"][2, 3, :].sum() / per(r) * 1000 for r in rs]))
        cl = sorted(dict(lay_.ends))                                     # lanes that close in this configuration
        op = [ln for ln in range(4) if ln not in cl]                     # lanes that continue
        o["out_last300_per_kveh"] = float(np.mean([r["lch"][np.ix_(cl, op, range(3))].sum() / per(r) * 1000
                                                   for r in rs]))        # out of the closing lanes, last 300 m
        o["three_to_two_per_kveh"] = float(np.mean([r["lch"][3, 2, :].sum() / per(r) * 1000 for r in rs]))
        o["share_550"] = float(np.nanmean([sum(np.nan_to_num(r["kprof"][ln, 19]) for ln in cl)
                                           / np.nansum(r["kprof"][:, 19]) for r in rs]))
        if a == "CAZP":
            o["pacers_per_run"] = float(np.mean([r["n_pacers"] for r in rs]))
            o["active_min"] = float(np.mean([r["log"].get("on_s", 0) / 60 for r in rs]))
            o["act_max"] = float(np.mean([r["log"].get("act_max", 0) for r in rs]))
        if a == "VSLL":
            o["posted_kmh"] = float(np.mean([r["log"]["vsll_post_sum"] / r["log"]["vsll_post_n"] for r in rs
                                             if r["log"].get("vsll_post_n")]))
        out[a] = o
    return out


# ------------------------------------------------------------------ Fig. 5: ablation and gain (main test bed)
ABL = [("CAZP", "CAZP (full)", "full"),
       ("no_coverage", "without coverage rule", "core"), ("depth_1", "one pacer per lane (depth 1)", "core"),
       ("no_formation", "without alignment rule", "core"), ("tie_only", "speed cap only, no lane keeping", "core"),
       ("hold_only", "lane keeping only, no speed cap", "core"), ("outer_only", "only lane 3 paced", "core"),
       ("inner_only", "only lane 2 paced", "core"),
       ("fixed_zone", "fixed 700-m zone", "variant"), ("display", "CAZP-DC (driver executes)", "variant"),
       ("NC_nohop", "lane hopping off, no control", "control"), ("CAZP_nohop", "lane hopping off, CAZP", "control")]


def fig05(AB, AL, KN, a0, gains):
    """Fig. 5: (a)-(c) ablation at 3800 veh/h, Holm adjustment within each group of variants; (d)-(f) gains around the
    calibrated gain a0."""
    rows = [r for r in ABL if any(k[0] == r[0] for k in AB)]
    disp, prev = [], "full"
    for r in rows:
        if r[2] != prev:
            disp.append(GROUP_LAB[r[2]])
            prev = r[2]
        disp.append(r)
    fig = plt.figure(figsize=(7.4, 7.0))
    top = fig.add_gridspec(1, 3, wspace=0.08, left=0.335, right=0.99, top=0.96, bottom=0.47)
    axs = [fig.add_subplot(top[0, 0])]
    axs += [fig.add_subplot(top[0, j], sharey=axs[0]) for j in (1, 2)]
    vs = {a: paired(AB, a, 3800, "delay", ref="CAZP") for a, _, g in rows if g != "full"}
    adj = {}
    for grp in ("core", "variant", "control"):
        fam = [a for a, _, g in rows if g == grp]
        if fam:
            adj.update(zip(fam, holm([vs[a]["p"] for a in fam])))
    KN["ablation"] = {}
    labels = []
    for i, row in enumerate(disp):
        if isinstance(row, str):
            labels.append(row)
            continue
        a, l, g = row
        s, q = paired(AB, a, 3800, "delay"), paired(AB, a, 3800, "q_queue")
        qu, h = paired(AB, a, 3800, "queue_mean"), paired(AB, a, 3800, "hb")
        w, pa = vs.get(a), adj.get(a)
        b = summ([pacer_burden(r, AB.get(("NC", 3800, sd))) for (aa, d, sd), r in AB.items() if aa == a]) \
            if a not in ("NC_nohop",) else None
        KN["ablation"][a] = dict(delay=s, q=q, queue=qu, hb=h, vs_full=w, p_holm=pa, burden=b)
        mark = ""
        if w is not None and g != "control" and pa < 0.05:
            mark = " †" if w["mean"] > 0 else " ‡"
        c = COL["CAZP"] if a == "CAZP" else ("#999999" if g == "control" else ("#6aa66e" if g == "core" else "#b39ddb"))
        for ax, v in ((axs[0], s), (axs[1], q), (axs[2], h)):
            if np.isfinite(v["mean"]):
                ax.barh(i, v["mean"], color=c, height=0.68, alpha=0.9)
                ax.errorbar(v["mean"], i, xerr=v["half"], fmt="none", ecolor="#333", elinewidth=0.8, capsize=1.5)
        labels.append(l + mark)
    axs[0].set_yticks(range(len(disp)), labels)
    for i, t in enumerate(axs[0].get_yticklabels()):
        if isinstance(disp[i], str):
            t.set_color("#444")
    axs[0].set_ylim(len(disp) - 0.5, -0.6)
    for ax in axs[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)
    for ax, lab, tt in zip(axs, ("Change (s/veh)", "Change (veh/h)", "Change (per 1000 veh)"),
                           ("(a) Delay", "(b) Queue discharge rate", "(c) Hard braking")):
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(lab)
        ax.set_title(tt, loc="left")
        ax.grid(axis="x", color="#eee", lw=0.6)
    bot = fig.add_gridspec(1, 3, wspace=0.45, left=0.085, right=0.99, top=0.35, bottom=0.075)
    ax4, ax5, ax6 = (fig.add_subplot(bot[0, j]) for j in range(3))
    KN["gain"] = {}
    ref = f"fix{a0:g}"
    allg = sorted(set(gains) | {a0})
    off = (-0.008, 0.0, 0.008)
    for di, dem in enumerate(L3):
        ys, hs, os_, bs, qh, bh = [], [], [], [], [], []
        for al in allg:
            arm = f"fix{al:g}"
            s, q = paired(AL, arm, dem, "delay"), paired(AL, arm, dem, "q_queue")
            rr = paired(AL, arm, dem, "delay", ref=ref) if arm != ref else dict(mean=0.0, half=0.0, lo=0.0, hi=0.0,
                                                                                   p=np.nan)
            b = summ([pacer_burden(r, AL.get(("NC", dem, sd))) for (a, d, sd), r in AL.items()
                        if a == arm and d == dem])
            ys.append(rr["mean"]), hs.append(rr["half"]), os_.append(q["mean"]), bs.append(b["mean"])
            qh.append(q["half"]), bh.append(b["half"])
            KN["gain"][f"{dem}|{al:g}"] = dict(delay=s, vs_cal=rr, q=q, burden=b)
        ax4.errorbar(np.array(allg) + off[di], ys, yerr=hs, fmt="-o", color=DCOL[di], ms=3, lw=1.1, capsize=1.5,
                     elinewidth=0.8, label=f"{dem} veh/h")
        ax5.errorbar(np.array(allg) + off[di], os_, yerr=qh, fmt="-o", color=DCOL[di], ms=3, lw=1.1, capsize=1.5,
                     elinewidth=0.8, label=f"{dem} veh/h")
        ax6.errorbar(np.array(allg) + off[di], bs, yerr=bh, fmt="-o", color=DCOL[di], ms=3, lw=1.1, capsize=1.5,
                     elinewidth=0.8)
    for ax, lab, tt in ((ax4, f"Delay relative to gain {a0:g} (s/veh)", "(d) Delay"),
                        (ax5, "Discharge change (veh/h)", "(e) Queue discharge rate"),
                        (ax6, "Delay change per pacer (s)", "(f) Delay change of the pacers")):
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("Coupling gain α")
        ax.set_ylabel(lab)
        ax.set_title(tt, loc="left")
        ax.set_xticks(allg)
        ax.grid(color="#eee", lw=0.6)
        ax.axvline(a0, color=COL["CAZP"], lw=0.9, ls=":", zorder=0)
    ax4.text(a0 + 0.008, ax4.get_ylim()[1], f"calibrated {a0:g}", color=COL["CAZP"], fontsize=8, va="top")
    ax5.legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, "Fig5_ablation_gain")


# ------------------------------------------------------------------ configurations: statistics and Fig. 7
def config_stats(Mc, lay):
    """Pooled paired changes against no control for every strategy and indicator; Holm over the strategies for every
    indicator (p_holm; p_holm_delay repeats it for the delay); head-to-head CAZP minus each other strategy."""
    dems = dems_of(lay)
    arms = [a for a in ARMS if any(k[0] == a for k in Mc)]
    out = {"n_runs": len(Mc)}
    for a in arms:
        out[a] = {m: pooled(Mc, a, m, dems=dems) for m in METRICS}
    ph = holm([out[a]["delay"]["p"] if np.isfinite(out[a]["delay"]["p"]) else 1.0 for a in arms])
    for a, p in zip(arms, ph):
        out[a]["p_holm_delay"] = float(p)
        out[a]["n_sig"] = int(sum((out[a][m]["p"] < 0.05) and ((out[a][m]["mean"] > 0) == (m == "q_queue"))
                                  for m in METRICS))
        out[a]["n_worse"] = int(sum((out[a][m]["p"] < 0.05) and ((out[a][m]["mean"] > 0) != (m == "q_queue"))
                                    for m in METRICS))
    for m in METRICS:                                                    # Holm over the strategies, every indicator
        pm = holm([out[a][m]["p"] if np.isfinite(out[a][m]["p"]) else 1.0 for a in arms])
        for a, p in zip(arms, pm):
            out[a][m]["p_holm"] = float(p)
    out["h2h"] = {}
    for b in arms:
        if b == "CAZP":
            continue
        out["h2h"][b] = {m: pooled(Mc, "CAZP", m, dems=dems, ref=b) for m in ("delay", "q_queue", "hb", "queue_mean")}
    fam = [b for b in out["h2h"] if b != "CAZP_VSL1"]
    for m in ("delay", "q_queue", "hb", "queue_mean"):
        for b, p in zip(fam, holm([out["h2h"][b][m]["p"] for b in fam])):
            out["h2h"][b][m]["p_holm"] = float(p)
    out["by_demand"] = {f"{d}|{a}|{m}": paired(Mc, a, d, m) for d in dems for a in arms for m in ("delay", "q_queue")}
    out["nc_abs"] = {f"{d}|{m}": float(np.nanmean([val(r, m) for r in runs_of(Mc, "NC", d)])) for d in dems
                     for m in METRICS}
    return out


def aggregate(C, MC):
    """Equal-weight mean of the pooled relative delay change over the three configurations and the worst
    configuration. The 95% interval comes from the seed-wise means over the configurations on the seeds that all
    configurations share (1-30), so that it accounts for the common random numbers of the configurations."""
    out = {}
    seeds = range(1, 31)
    base = {lay: float(np.mean([r["delay"] for (a, d, sd), r in MC[lay].items() if a == "NC" and sd in seeds]))
            for lay in ORDER}
    for a in ARMS:
        r = [C[lay][a]["delay"] for lay in ORDER if a in C.get(lay, {})]
        if len(r) < len(ORDER):
            continue
        rel = [x["rel"] for x in r]
        per = []
        for sd in seeds:
            vals = []
            for lay in ORDER:
                xs = [(MC[lay].get((a, dm, sd)), MC[lay].get(("NC", dm, sd))) for dm in dems_of(lay)]
                if any(x is None or y is None for x, y in xs):
                    break
                vals.append(100 * np.mean([x["delay"] - y["delay"] for x, y in xs]) / base[lay])
            else:
                per.append(float(np.mean(vals)))
        sw = summ(per)
        m = float(np.mean(rel))
        out[a] = dict(mean=m, lo=m - sw["half"], hi=m + sw["half"], worst=float(max(rel)), best=float(min(rel)),
                      by=dict(zip(ORDER, rel)), seedwise=dict(mean=sw["mean"], half=sw["half"], n=sw["n"], p=sw["p"]))
    return out


LABELS_RB = {"main": "main test bed", "bed_F": "no passing on the right",
             "lcStrategic_1.0": "earlier lane changes (lcStrategic 1.0)",
             "lcAssertive_1.5": "smaller gaps (lcAssertive 1.5)",
             "lcAssertive_2": "smaller gaps (lcAssertive 2)",
             "lcAssertive_3": "smaller gaps (lcAssertive 3)",
             "lcCooperative_0.75": "less cooperative (lcCooperative 0.75)",
             "lcCooperative_0.5": "less cooperative (lcCooperative 0.5)", "tau_1.3": "30% longer time headways",
             "trucks_10": "10% heavy vehicles", "keepRight_0": "no keep-right rule (lcKeepRight 0)"}


def robustness(axs, RB, KN):
    """Fig. 7a-b: CAZP against no control in the modified test beds at 3800 veh/h; KN["robustness"]. Also the share of
    lane 3 without control, with and without the keep-right rule (25-40 min, 2.4-2.5 km and 0.7-1.2 km
    before the final lane end, i.e. the last 500 m of lane 3): KN["lane3_share_NC"]."""
    rows = [r for r in ["main"] + list(E.ROBUSTNESS_ROWS) if any(k[0] == f"CAZP|{r}" for k in RB)]
    KN["robustness"], KN["lane3_share_NC"] = {}, {}
    for row in rows:
        KN["robustness"][row] = dict(delay=paired(RB, f"CAZP|{row}", 3800, "delay", ref=f"NC|{row}"),
                                     q=paired(RB, f"CAZP|{row}", 3800, "q_queue", ref=f"NC|{row}"))
    for row in ("main", "keepRight_0"):
        rs = runs_of(RB, f"NC|{row}", 3800)
        if rs:
            with np.errstate(all="ignore"):
                k = np.nanmean([r["kprof"] for r in rs], axis=0)
            sh = np.nan_to_num(k[3]) / np.nansum(k, axis=0)
            KN["lane3_share_NC"][row] = dict(at_2_5km=float(sh[0]), last_500m=float(np.mean(sh[13:18])))
    for i, row in enumerate(rows):
        s, q = KN["robustness"][row]["delay"], KN["robustness"][row]["q"]
        for ax, v, x, lo, hi in ((axs[0], s, s["rel"], s["rel"] - s["rlo"], s["rhi"] - s["rel"]),
                                 (axs[1], q, q["mean"], q["half"], q["half"])):
            if np.isfinite(v["mean"]):
                ax.errorbar(x, i, xerr=[[lo], [hi]], fmt="o", ms=4,
                            color=COL["CAZP"] if row == "main" else "#4d8f5e", capsize=1.5)
        if np.isfinite(s["base"]):
            axs[0].text(-1.0, i, f"{s['base']:.0f} s/veh", fontsize=7.8, va="center", ha="right", color="#666")
    axs[0].text(-1.0, -0.75, "no control", fontsize=7.8, va="center", ha="right", color="#666")
    axs[0].set_yticks(range(len(rows)), [LABELS_RB[r] for r in rows])
    axs[0].invert_yaxis()
    axs[0].set_xlim(-57, 0.5)
    for ax, lab, tt in ((axs[0], "Delay change vs no control (%)", "(a) Delay, modified test beds"),
                        (axs[1], "Discharge change (veh/h)", "(b) Queue discharge rate")):
        ax.axvline(0, color="k", lw=0.6)
        ax.set_xlabel(lab)
        ax.set_title(tt, loc="left", fontsize=9.5)
        ax.grid(axis="x", color="#eee", lw=0.6)


MK = {"seq": "o", "base": "s", "l43": "D"}


def col7(a):
    """Colours of Fig. 7c-d: CAZP, CAZP + VSL-S and the best benchmark (VSL-L) in their colours, other benchmarks grey."""
    return COL[a] if a in ("VSLL", "CAZP", "CAZP_VSL1") else "#8a8a8a"


def by_configuration(ax, C):
    """Fig. 7c: pooled delay change of every strategy in configurations I-III."""
    arms = [a for a in ARMS if all(a in C[lay] for lay in ORDER)]
    w = 0.26
    for k, lay in enumerate(ORDER):
        for i, a in enumerate(arms):
            s = C[lay][a]["delay"]
            y = i + (k - 1) * w
            ax.errorbar(s["rel"], y, xerr=[[s["rel"] - s["rlo"]], [s["rhi"] - s["rel"]]], fmt=MK[lay], ms=3.8,
                        color=col7(a), mfc=col7(a) if k == 0 else "white", mew=1.0, elinewidth=0.9, capsize=1.2)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(range(len(arms)), [LAB[a] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel("Delay change vs no control (%)")
    ax.set_title("(c) Delay in each closure configuration", loc="left", fontsize=9.5)
    hs = [Line2D([], [], marker=MK[lay], ls="", color="k", mfc="k" if k == 0 else "white", ms=4.5,
                 label=f"{NUM[lay]}: {NAME[lay]}") for k, lay in enumerate(ORDER)]
    leg = ax.legend(handles=hs, loc="lower right", frameon=True, framealpha=1.0, facecolor="white", edgecolor="none",
                    fontsize=7.8, handletextpad=0.3)
    leg.set_zorder(5)
    ax.grid(axis="x", color="#eee", lw=0.6)


def aggregate_panel(ax, A):
    """Fig. 7d: mean delay change over the three configurations."""
    arms = [a for a in ARMS if a in A]
    lo_x = min(min(g["lo"] for g in A.values()) - 4, -46)
    for i, a in enumerate(arms):
        g = A[a]
        ax.barh(i, g["mean"], height=0.55, color=col7(a), alpha=0.85, lw=0)
        ax.errorbar(g["mean"], i, xerr=[[g["mean"] - g["lo"]], [g["hi"] - g["mean"]]], fmt="none", ecolor="k",
                    elinewidth=0.8, capsize=1.6)
        ax.text(30.0, i, mt(f"{g['mean']:+.1f}"), va="center", ha="right", fontsize=8.0)
    ax.text(30.0, -0.75, "mean", va="center", ha="right", fontsize=8.0, color="#555")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_xlim(lo_x, 31)
    ax.set_xlabel("Delay change vs no control (%)")
    ax.set_title("(d) Mean of configurations I-III", loc="left", fontsize=9.5)
    ax.grid(axis="x", color="#eee", lw=0.6)


def fig07(RB, C, A, KN):
    """Fig. 7: modified test beds (a, b) and closure configurations (c, d)."""
    fig = plt.figure(figsize=(7.6, 7.3), layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[3.6, 3.3], hspace=0.04)
    a = top.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[1.0, 0.62]))
    robustness(a, RB, KN)
    b = bot.subplots(1, 2, sharey=True, gridspec_kw=dict(width_ratios=[1.0, 0.62]))
    by_configuration(b[0], C)
    aggregate_panel(b[1], A)
    save(fig, "Fig7_robustness_configurations")


# ------------------------------------------------------------------ numbers of the text (Sections 4-5)
def pacer_ratio(MC, KN):
    """Pacers of CAZP in each configuration (Sections 5.7 and 6): mean delay change per pacer against the same vehicle
    without control (burden; test seeds 1-30, all demands), and the delay added to all pacers together against the net
    saving of all vehicles (vehicle-seconds)."""
    out = {}
    for lay in ORDER:
        bur = [pacer_burden(MC[lay].get(("CAZP", d, s)), MC[lay].get(("NC", d, s))) for d in dems_of(lay)
               for s in range(1, 31)]
        save = pace = 0.0
        for (a, d, s), r in MC[lay].items():
            b = MC[lay].get(("NC", d, s))
            if a != "CAZP" or b is None or len(r["veh"]["delay"]) != len(b["veh"]["delay"]):
                continue
            dd = r["veh"]["delay"].astype(float) - b["veh"]["delay"].astype(float)
            ok = np.isfinite(dd)
            save -= float(dd[ok].sum())
            pace += float(dd[ok & r["veh"]["held"]].sum())
        out[lay] = dict(saving_vehs=save, pacer_vehs=pace, share=pace / save if save > 0 else float("nan"),
                        burden=summ([b for b in bur if np.isfinite(b)]))
    KN["pacer_ratio"] = out


def acceptance_configs(MC, KN):
    """Acceptance of 25% instead of 75% in configurations II and III (block acceptance_II_III), highest demand, test
    seeds 1-16: CAZP against no control and against VSL-L."""
    out, sd = {}, set(range(1, 17))
    for lay in ORDER:
        if lay == MAIN:
            continue
        ac = load("acceptance_II_III", lay)
        if not ac:
            return
        D, top = {**MC[lay], **ac}, dems_of(lay)[-1]
        o = {a: dict(delay=paired(D, a, top, "delay", seeds=sd), q=paired(D, a, top, "q_queue", seeds=sd),
                     vs_vsll=paired(D, a, top, "delay", ref="VSLL", seeds=sd) if a != "VSLL" else None)
             for a in ("CAZP", "CAZP_c0.25", "VSLL")}
        o["dem"] = top
        out[lay] = o
    KN["acceptance_cfg"] = out


def gain_transfer(MC, KN):
    """CAZP with the gain of the main test bed in configurations II and III (block gain_transfer, test seeds 1-30):
    change against no control, paired differences to every strategy of the comparison, and the delay of the pacers."""
    out = {}
    for lay in ORDER:
        if lay == MAIN:
            continue
        tg = load("gain_transfer", lay)
        if not tg:
            return
        D, dems = {**MC[lay], **tg}, dems_of(lay)
        a = "CAZP_mainGain"
        o = {m: pooled(D, a, m, dems=dems) for m in ("delay", "q_queue", "hb")}
        o["vs"] = {b: {m: pooled(D, a, m, dems=dems, ref=b) for m in ("delay", "q_queue", "hb")}
                   for b in ("CAZP", "VSL1", "VSLL", "PACE", "EM", "ZMSD", "VSL1_ZMSD", "VSLL_ZMSD", "CAZP_VSL1")
                   if any(k[0] == b for k in D)}
        bur = [pacer_burden(D.get((a, d, s)), D.get(("NC", d, s))) for d in dems for s in range(1, 31)]
        o["burden"] = summ([b for b in bur if np.isfinite(b)])
        save = pace = 0.0
        for (aa, d, s), r in D.items():
            b = D.get(("NC", d, s))
            if aa != a or b is None or len(r["veh"]["delay"]) != len(b["veh"]["delay"]):
                continue
            dd = r["veh"]["delay"].astype(float) - b["veh"]["delay"].astype(float)
            ok = np.isfinite(dd)
            save -= float(dd[ok].sum())
            pace += float(dd[ok & r["veh"]["held"]].sum())
        o["pacer_share"] = pace / save if save > 0 else float("nan")
        o["pacers_per_run"] = float(np.mean([r["n_pacers"] for (aa, d, s), r in D.items() if aa == a]))
        out[lay] = o
    KN["gain_transfer"] = out


def bound_size_configs(M, MC, KN):
    """Bounds of 10 and 30 km/h instead of 20 km/h in every configuration (blocks bound_I, bound_II_III), highest
    demand, test seeds 1-16: change against no control, against the same strategy with 20 km/h, and CAZP against VSL-L
    with the same bound."""
    sd, out = set(range(1, 17)), {}
    for lay in ORDER:
        dv = load("bound_I" if lay == MAIN else "bound_II_III", lay)
        if not dv:
            return
        D, top = {**(M if lay == MAIN else MC[lay]), **dv}, dems_of(lay)[-1]
        o = {}
        for a in ("CAZP", "VSLL", "CAZP_dv10", "CAZP_dv30", "VSLL_dv10", "VSLL_dv30"):
            base = a.split("_")[0]
            o[a] = dict(delay=paired(D, a, top, "delay", seeds=sd), q=paired(D, a, top, "q_queue", seeds=sd),
                        hb=paired(D, a, top, "hb", seeds=sd),
                        vs_20=paired(D, a, top, "delay", ref=base, seeds=sd) if a != base else None)
        for x, y in (("CAZP", "VSLL"), ("CAZP_dv10", "VSLL_dv10"), ("CAZP_dv30", "VSLL_dv30")):
            o[f"{x}-{y}"] = {m: paired(D, x, top, m, ref=y, seeds=sd) for m in ("delay", "q_queue", "hb")}
        bur = {a: summ([b for b in (pacer_burden(D.get((a, top, s)), D.get(("NC", top, s))) for s in sd)
                          if np.isfinite(b)]) for a in ("CAZP", "CAZP_dv10", "CAZP_dv30")}
        o["burden"] = bur
        o["dem"] = top
        out[lay] = o
    KN["bound_size_cfg"] = out


def compliance_levels(MC, KN):
    """Compliance with the speed limits (block compliance): VSL-S, VSL-L and CAZP + VSL-S at the compliance levels of
    Hellinga and Mandelzys (2011), highest demand of every configuration, test seeds 1-16; level D = block main, in which
    drivers treat a posted limit as the static limit. Change against no control, CAZP against VSL-L at the same level
    (Holm over the five levels of a configuration) and CAZP + VSL-S against CAZP."""
    sd, out = set(range(1, 17)), {}
    for lay in ORDER:
        cp = load("compliance", lay)
        if not cp:
            return
        top = dems_of(lay)[-1]
        D = dict(cp)
        D.update({k: r for k, r in MC[lay].items() if k[1] == top and k[2] in sd and k[0] in ("NC", "CAZP")})
        for a in ("VSL1", "VSLL", "CAZP_VSL1"):
            D.update(alias(MC[lay], a, f"{a}_D", dems=(top,), seeds=sd))
        o = {"dem": top, "CAZP": {m: paired(D, "CAZP", top, m, seeds=sd) for m in ("delay", "q_queue", "hb")}}
        lv_all = ("D",) + E.LEVELS
        for lv in lv_all:
            o[lv] = {a: {m: paired(D, f"{a}_{lv}", top, m, seeds=sd) for m in ("delay", "q_queue", "hb")}
                     for a in ("VSL1", "VSLL", "CAZP_VSL1")}
            o[lv]["CAZP-VSLL"] = {m: paired(D, "CAZP", top, m, ref=f"VSLL_{lv}", seeds=sd)
                                  for m in ("delay", "q_queue", "hb")}
            o[lv]["CAZP_VSL1-CAZP"] = {m: paired(D, f"CAZP_VSL1_{lv}", top, m, ref="CAZP", seeds=sd)
                                       for m in ("delay", "hb")}
        adj = holm([o[lv]["CAZP-VSLL"]["delay"]["p"] for lv in lv_all])
        for lv, p in zip(lv_all, adj):
            o[lv]["CAZP-VSLL"]["delay"]["p_holm"] = float(p)
        posted = [r["log"]["vsll_post_sum"] / r["log"]["vsll_post_n"] for (a, d, s), r in D.items()
                  if a == "VSLL_D" and r["log"].get("vsll_post_n")]
        o["vsll_posted_kmh"] = float(np.mean(posted))           # mean limit of the VSL-L signs while active
        out[lay] = o
    KN["compliance_cfg"] = out


def benchmark_numbers(M, PS, KN):
    """Zipper-sign followers (block zipper_variants), hard braking near the queue tail, lane speeds at 3800 veh/h,
    cell length and speed threshold of CAZP (block thresholds)."""
    z = slice(18, 23)
    zf = {}
    for c, a in ((25, "ZMSD_c25"), (50, "ZMSD"), (75, "ZMSD_c75"), (100, "ZMSD_c100")):
        rs = runs_of(M, a, 3800)
        if not rs:
            continue
        zf[str(c)] = dict(delay=pooled(M, a, "delay"), hb=pooled(M, a, "hb"), q=pooled(M, a, "q_queue"),
                          stops=pooled(M, a, "stops"),
                          dv3800=float(np.nanmean([np.nanmean(r["vprof"][2:, z]) - np.nanmean(r["vprof"][1, z]) for r in rs])),
                          followers=float(np.mean([r["log"].get("zms_n", 0) for r in rs])))
    KN["zipper_followers"] = zf
    KN["hb_tail"], KN["mech3800"] = {}, {}
    for a in ["NC"] + ARMS:
        rs = [r for (aa, d, s), r in M.items() if aa == a]
        if not rs:
            continue
        KN["hb_tail"][a] = dict(total=float(np.mean([r["sm"]["hard_brake_per_kveh"] for r in rs])),
                                tail=float(np.mean([r["sm"]["hb_tail_per_kveh"] for r in rs])))
        r3 = runs_of(M, a, 3800)
        KN["mech3800"][a] = dict(
            v1=float(np.nanmean([np.nanmean(r["vprof"][1, z]) for r in r3])),
            vc=float(np.nanmean([np.nanmean(r["vprof"][2:, z]) for r in r3])),
            share=float(np.nanmean([(r["kprof"][2, 19] + r["kprof"][3, 19]) / np.nansum(r["kprof"][:, 19]) for r in r3])))
    if PS:
        D, sd = {**M, **PS}, set(range(1, 17))
        KN["param_sens"] = {a: dict(delay=paired(D, a, 3800, "delay", seeds=sd), q=paired(D, a, 3800, "q_queue", seeds=sd))
                            for a in ("CAZP", "CAZP_S40", "CAZP_S60", "CAZP_S75", "CAZP_L50", "CAZP_L200")
                            if a == "CAZP" or any(k[0] == a for k in PS)}


def bed_extras(M, KN):
    """Numbers of the test-bed description: queue lengths without control, presence of the queue in the discharge
    window, discharge per open lane, active control time and pacers."""
    o = {}
    for dem in L3:
        nc = runs_of(M, "NC", dem)
        qm = np.array([r["queue_max"] for r in nc])
        pres = []
        for r in nc:
            q = queue_len(r)
            t = np.arange(len(q)) * r["grid_dt"]
            w = (t >= 1800) & (t < 2400)
            pres.append(float(np.mean(q[w] > 0)) if w.any() else np.nan)
        o[str(dem)] = dict(queue_max_mean=float(qm.mean()), queue_max_p95=float(np.percentile(qm, 95)),
                           queue_max_max=float(qm.max()), queue_present_30_40=float(np.nanmean(pres)),
                           queue_present_min=float(np.nanmin(pres)))
        cz = runs_of(M, "CAZP", dem)
        o[str(dem)].update(active_min=float(np.mean([r["log"].get("on_s", 0) / 60 for r in cz])),
                           pacers=float(np.mean([r["n_pacers"] for r in cz])),
                           pacer_share=float(np.mean([r["n_pacers"] / max(len(r["veh"]["delay"]), 1) for r in cz])),
                           act_max=float(np.mean([r["log"].get("act_max", 0) for r in cz])))
    KN["bed_extras"] = o


def validity(KN):
    """Checks over all test runs loaded: runs with vehicles still in the network at the end of the simulation (block,
    arm, configuration, demand, seed, number of vehicles) and the pacers among them, runs with collisions, runs with
    invalid insertions, size of the run records (GB)."""
    rs = list(_cache.values())
    KN["validity"] = dict(n_runs=len(rs), n_trunc=int(sum(r["n_trunc"] > 0 for r in rs)),
                          trunc_runs=[(b, s["arm"], s["layout"], s["dem"], s["seed"], int(_cache[p]["n_trunc"]))
                                      for p, (b, s) in _first.items() if _cache[p]["n_trunc"] > 0],
                          trunc_pacers=int(sum(int((np.isnan(r["veh"]["delay"]) & r["veh"]["held"]).sum())
                                               for r in rs if r["n_trunc"] > 0)),
                          n_collision_runs=int(sum(r["coll"] > 0 for r in rs)),
                          n_invalid_insertion=int(sum(not r["valid"] for r in rs)),
                          gbytes=float(sum(os.path.getsize(p) for p in _cache) / 1e9))


# ------------------------------------------------------------------ panels of Figs. 2, 3 and 6
def mean_map(D, arm, dem, min_share=0.5):
    """Mean time-space speed field over the runs of an arm; cells observed in fewer than min_share of the runs (the
    first vehicles entering, the last ones leaving) are left blank."""
    runs = runs_of(D, arm, dem)
    nt, nx = max(r["grid_v"].shape[0] for r in runs), runs[0]["grid_v"].shape[1]
    Mx = np.full((len(runs), nt, nx), np.nan)
    for k, r in enumerate(runs):
        g = r["grid_v"].astype(float)
        Mx[k, :g.shape[0], :] = g[:, :nx]
    with np.errstate(all="ignore"):
        m = np.nanmean(Mx, axis=0)
    m[np.isfinite(Mx).mean(axis=0) < min_share] = np.nan
    return m, runs


def ts_maps(M, axs, dems=None, xmax=62.0):
    """Mean time-space speed fields without control and with CAZP and their difference, one row per demand."""
    for i, dem in enumerate(dems or L3):
        (mn, rn), (mc, rc) = mean_map(M, "NC", dem), mean_map(M, "CAZP", dem)
        nt = max(mn.shape[0], mc.shape[0])
        mn, mc = (np.vstack([A, np.full((nt - A.shape[0], A.shape[1]), np.nan)]) for A in (mn, mc))
        ext = [0, nt * rn[0]["grid_dt"] / 60, -R.X_DROP, -R.X_DROP + mn.shape[1] * rn[0]["grid_dx"]]
        for j, (Mx, runs) in enumerate(((mn, rn), (mc, rc))):
            im = axs[i, j].imshow(Mx.T, origin="lower", aspect="auto", cmap="RdYlGn", vmin=0, vmax=100, extent=ext,
                                  interpolation="nearest")
            axs[i, j].text(0.98, 0.97, f"delay {np.mean([r['delay'] for r in runs]):.0f} s/veh\n"
                           f"discharge {np.mean([r['q_queue'] for r in runs]):.0f} veh/h", ha="right", va="top",
                           transform=axs[i, j].transAxes, fontsize=7.8,
                           bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        with np.errstate(invalid="ignore"):
            imd = axs[i, 2].imshow((mc - mn).T, origin="lower", aspect="auto", cmap="RdBu", vmin=-40, vmax=40,
                                   extent=ext, interpolation="nearest")
        s = paired(M, "CAZP", dem, "delay")
        axs[i, 2].text(0.98, 0.97, mt(f"delay {s['mean']:+.1f} s/veh\n[{s['lo']:+.1f}, {s['hi']:+.1f}]"), ha="right",
                       va="top", transform=axs[i, 2].transAxes, fontsize=7.8,
                       bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
        for ax in axs[i]:
            ax.set_ylim(-2500, 800)
            ax.set_xlim(0, xmax)
            ax.axhline(0, color="k", lw=0.6)
    return im, imd


def map_frame(fig, axs, im, imd, letters, xlabel_row):
    """Panel letters above the maps, column headings, and horizontal colour bars below the bottom row."""
    for k, ax in enumerate(axs.flat):
        ax.set_title(f"({letters[k]})", loc="left", fontsize=9.5, pad=2.5)
    for j, t in enumerate(("No control", "CAZP (proposed)", f"Difference: CAZP {MINUS} no control")):
        axs[0, j].annotate(t, xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 14), textcoords="offset points",
                           ha="center", va="bottom", fontsize=10)
    nr = axs.shape[0]
    for i in range(nr - 1):
        for j in range(3):
            axs[i, j].tick_params(labelbottom=False)
    for i in range(nr):
        for j in (1, 2):
            axs[i, j].tick_params(labelleft=False)
    for ax in axs[xlabel_row]:
        ax.set_xlabel("Time (min)")
    p0, p1, p2 = (axs[nr - 1, j].get_position() for j in range(3))
    y = p0.y0 - 0.075 * 8.0 / fig.get_figheight()
    cb = fig.colorbar(im, cax=fig.add_axes([p0.x0 + 0.05, y, p1.x1 - p0.x0 - 0.10, 0.010]), orientation="horizontal")
    cb.set_label("Mean speed (km/h), first two columns", fontsize=8.6)
    cb2 = fig.colorbar(imd, cax=fig.add_axes([p2.x0 + 0.015, y, p2.x1 - p2.x0 - 0.03, 0.010]), orientation="horizontal",
                       extend="both")
    cb2.set_label("Speed change (km/h)", fontsize=8.6)
    for c in (cb, cb2):
        c.ax.tick_params(labelsize=8)


def queue_panels(M, KN, axs):
    """Fig. 2j-l at 3800 veh/h: cumulative counts, queue length and delay distribution."""
    dem, q0 = 3800, 3000.0
    arms = [("NC", COL["NC"], "--"), ("CAZP", COL["CAZP"], "-"), ("CAZP_VSL1", COL["CAZP_VSL1"], "-")]
    ax = axs[0]
    nc = runs_of(M, "NC", dem)
    n = min(len(r["out"]) for a_, _, _ in arms for r in runs_of(M, a_, dem))
    t = nc[0]["t"][:n] + 30.0
    nout = {a: np.mean([np.cumsum(r["out"][:n]) * 30 / 3600 for r in runs_of(M, a, dem)], axis=0) for a, _, _ in arms}
    arr_t = [np.sort(r["veh"]["t_dep"]) for r in nc]
    taus = np.arange(60, 600, 10)
    early = (t > 300) & (t < 720)
    err = [np.mean(np.abs(np.mean([np.searchsorted(a_, t[early] - tau) for a_ in arr_t], axis=0) - nout["NC"][early]))
           for tau in taus]
    tau = float(taus[int(np.argmin(err))])
    nin = np.mean([np.searchsorted(a_, t - tau) for a_ in arr_t], axis=0)
    obl = lambda y: y - q0 * t / 3600                                # noqa: E731
    ax.plot(t / 60, obl(nin), color="k", ls=":", lw=1.2, label="arrivals (shifted)")
    for a, c, ls in arms:
        ax.plot(t / 60, obl(nout[a]), color=c, ls=ls, lw=1.4, label=LAB[a])
    ax.set_xlim(5, 60)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.33 * (hi - lo))                          # room for the legend above the arrival curve
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Cumulative count minus 3000 veh/h × t (veh)")
    ax.set_title("(j) Cumulative counts", loc="left", fontsize=9.5)
    ax.legend(loc="upper left", fontsize=7.8, frameon=False, handlelength=1.8)
    ax.grid(color="#eee", lw=0.6)
    KN["queue_dyn"] = dict(tau_s=tau, max_stored={a: float(np.max(nin - nout[a])) for a, _, _ in arms})
    ax = axs[1]
    for a, c, ls in arms:
        Q = np.array([queue_len(r)[:n] for r in runs_of(M, a, dem)])
        Q = np.array([np.convolve(q_, np.ones(3) / 3, mode="same") for q_ in Q])
        mu = Q.mean(0)
        h = stats.t.ppf(0.975, len(Q) - 1) * Q.std(0, ddof=1) / math.sqrt(len(Q))
        tt = runs_of(M, a, dem)[0]["t"][:n] / 60
        ax.plot(tt, mu, color=c, ls=ls, lw=1.4)
        ax.fill_between(tt, mu - h, mu + h, color=c, alpha=0.15, lw=0)
        KN["queue_dyn"][f"{a}|peak_m"] = float(mu.max())
        KN["queue_dyn"][f"{a}|clear_min"] = float(tt[np.where(mu > 100)[0][-1]]) if np.any(mu > 100) else np.nan
    ax.set_xlim(5, 60)
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Queue length (m)")
    ax.set_title("(k) Queue length", loc="left", fontsize=9.5)
    ax.grid(color="#eee", lw=0.6)
    ax = axs[2]
    for a, c, ls in arms:
        d = np.concatenate([r["veh"]["delay"].astype(float) for r in runs_of(M, a, dem)])
        d = np.sort(d[np.isfinite(d)])
        ax.plot(d, np.arange(1, len(d) + 1) / len(d), color=c, ls=ls, lw=1.4)
        KN["queue_dyn"][f"{a}|p50"], KN["queue_dyn"][f"{a}|p95"] = float(np.percentile(d, 50)), float(np.percentile(d, 95))
    for p in (0.5, 0.95):
        ax.axhline(p, color="#bbbbbb", lw=0.6, ls=":")
    ax.set_xlim(0, 700)
    ax.set_xlabel("Delay per vehicle (s)")
    ax.set_ylabel("Cumulative share of vehicles")
    ax.set_title("(l) Delay per vehicle", loc="left", fontsize=9.5)
    ax.grid(color="#eee", lw=0.6)


def scorecard(M, KN, ax):
    """Fig. 3a: mean paired change of the nine indicators against no control, pooled over the demands; one colour scale
    for all indicators (green = better, capped at ±50%); Holm adjustment over the strategies. Also the
    head-to-head comparisons on the same seeds (KN["head2head"]; Holm over CAZP against the seven benchmarks)."""
    arms = [a for a in ARMS if any(k[0] == a for k in M)]
    V_, P = np.full((len(arms), len(SCORE)), np.nan), np.ones((len(arms), len(SCORE)))
    for j, (m, _, sg, _) in enumerate(SCORE):
        ss = [pooled(M, a, m) for a in arms]
        ps = holm([s["p"] if np.isfinite(s["p"]) else 1.0 for s in ss])
        for i, s in enumerate(ss):
            V_[i, j], P[i, j] = s["rel"], ps[i]
    G = V_ * np.array([sg for _, _, sg, _ in SCORE])[None, :]
    Gs = np.clip(G / 50.0, -1, 1)
    ax.imshow(Gs, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(SCORE)):
            if np.isfinite(V_[i, j]):
                ax.text(j, i, mt(f"{V_[i, j]:+.1f}") + ("*" if P[i, j] < 0.05 else ""), ha="center", va="center",
                        fontsize=8.4, color="w" if abs(Gs[i, j]) > 0.62 else "k")
    ax.set_xticks(range(len(SCORE)), [SCORE_LAB.get(m, l) for m, l, _, _ in SCORE], fontsize=8.6)
    ax.set_yticks(range(len(arms)), [LAB[a] for a in arms])
    ax.tick_params(length=0)
    for x in (3.5, 6.5):
        ax.axvline(x, color="w", lw=3)
    for grp, x in (("Efficiency", 1.5), ("Safety", 5.0), ("Emissions", 7.5)):
        ax.text(x, -0.85, grp, ha="center", va="bottom", fontsize=9.5)
    ax.set_ylim(len(arms) - 0.5, -1.1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("(a) Change against no control (%), pooled over 3400-3800 veh/h; colour: green = better, capped at "
                 "±50%", loc="left", pad=22, fontsize=9.5)
    KN["scorecard"] = {a: {m: dict(rel=float(V_[i, j]), p_holm=float(P[i, j])) for j, (m, _, _, _) in enumerate(SCORE)}
                       for i, a in enumerate(arms)}
    mets = ["delay", "q_queue", "queue_mean", "hb", "svar", "co2_g"]
    KN["head2head"] = {}
    pairs_ = [(x, y) for x, y in (("CAZP", "VSL1"), ("CAZP", "VSLL"), ("CAZP", "PACE"), ("CAZP", "EM"), ("CAZP", "ZMSD"),
                                  ("CAZP", "VSL1_ZMSD"), ("CAZP", "VSLL_ZMSD"), ("CAZP_VSL1", "VSL1"),
                                  ("CAZP_VSL1", "CAZP"))
              if any(k[0] == x for k in M) and any(k[0] == y for k in M)]
    for x, y in pairs_:
        for m in mets:
            KN["head2head"].setdefault(f"{x}-{y}", {})[m] = pooled(M, x, m, ref=y)
    fam = [f"{x}-{y}" for x, y in pairs_ if x == "CAZP"]                 # CAZP against the seven benchmarks
    for m in mets:
        for k_, p in zip(fam, holm([KN["head2head"][k_][m]["p"] for k_ in fam])):
            KN["head2head"][k_][m]["p_holm"] = float(p)


def speed_outflow(M, KN, axs):
    """Fig. 3b-g: speed profiles 25-40 min and 5-min outflow at the three demands, with VSL-L."""
    dist = 2500 - (np.arange(25) + 0.5) * 100
    KN["speed_profile"] = {}
    for j, dem in enumerate(L3):
        ax = axs[0, j]
        for a in ("NC", "VSL1", "VSLL", "PACE", "CAZP", "CAZP_VSL1"):
            runs = runs_of(M, a, dem)
            with np.errstate(all="ignore"):
                k = np.nanmean([r["kprof"] for r in runs], axis=0)
                v = np.nanmean([r["vprof"] for r in runs], axis=0)
                vs = np.nansum(k * v, axis=0) / np.nansum(k, axis=0)
            ax.plot(dist, vs, label=LAB[a], **STY[a])
            KN["speed_profile"][f"{dem}|{a}"] = float(np.nanmean(vs[-7:]))
        ax.set_xlim(2500, 0)
        ax.set_xticks([2000, 1500, 1000, 500, 0], ["2000", "", "1000", "", "0"])
        ax.set_title(f"({'bcd'[j]}) Speed profile, {dem} veh/h", loc="left", fontsize=9.5)
        ax.set_xlabel("Distance to the end of lane 2 (m)")
        ax.axvline(X3[MAIN], color="k", lw=0.5, ls=":")
        ax.grid(color="#eee", lw=0.6)
        ax = axs[1, j]
        for a in ("NC", "VSLL", "CAZP", "CAZP_VSL1"):
            runs = runs_of(M, a, dem)
            n = min(len(r["out"]) for r in runs) // 10 * 10
            o = np.array([r["out"][:n].reshape(-1, 10).mean(1) for r in runs])
            t = runs[0]["t"][:n].reshape(-1, 10)[:, 0] + 150
            keep = t >= 300
            o, t = o[:, keep], t[keep]
            mu, h = o.mean(0), stats.t.ppf(0.975, len(o) - 1) * o.std(0, ddof=1) / math.sqrt(len(o))
            st = STY[a]
            ax.plot(t / 60, mu, marker="o", ms=2.3, color=st["color"], ls=st["ls"], lw=1.2, label=LAB[a])
            ax.fill_between(t / 60, mu - h, mu + h, color=st["color"], alpha=0.13, lw=0)
        ncr = runs_of(M, "NC", dem)
        ax.axhline(np.mean([r["q_max5"] for r in ncr]), color="k", lw=0.8, ls="-.",
                   label="largest 5-min flow, no control")
        ax.axhline(np.mean([r["q_queue"] for r in ncr]), color="#c0392b", lw=0.8, ls=":",
                   label="queue discharge rate, no control")
        ax.set_xlim(0, 60)
        ax.set_ylim(1900, 3500)
        ax.set_xlabel("Time (min)")
        ax.grid(color="#eee", lw=0.6)
        ax.set_title(f"({'efg'[j]}) Outflow, {dem} veh/h", loc="left", fontsize=9.5)
    axs[0, 0].set_ylabel("Mean speed 25-40 min (km/h)")
    axs[1, 0].set_ylabel("Outflow (veh/h), 5-min bins")


def heat8(ax, Mx, xt, yt, xl, yl, title, lo, hi, good_low=True, sig=None):
    """Heat map with a colour scale from lo to hi around zero (green = better) and the cell values."""
    im = ax.imshow(Mx, origin="lower", cmap="RdYlGn_r" if good_low else "RdYlGn", norm=TwoSlopeNorm(0, lo, hi),
                   aspect="auto")
    for i in range(Mx.shape[0]):
        for j in range(Mx.shape[1]):
            if np.isfinite(Mx[i, j]):
                strong = Mx[i, j] < 0.6 * lo if good_low else Mx[i, j] > 0.6 * hi
                ax.text(j, i, mt(f"{Mx[i, j]:+.0f}"), ha="center", va="center", fontsize=8.6,
                        color="white" if strong else "k")
            if sig is not None and not sig[i, j]:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="////", ec="#999999", lw=0))
    ax.set_xticks(range(len(xt)), xt)
    ax.set_yticks(range(len(yt)), yt)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title, loc="left", fontsize=9.5)
    return im


def fig06(HM, KN):
    """Fig. 6: penetration x demand, acceptance x penetration at 3600 veh/h and the saturation with the share of acting
    vehicles (medians of the paired changes, Wilcoxon signed-rank tests)."""
    dems, pens = (3200, 3400, 3600, 3800, 4000), (0.025, 0.05, 0.10, 0.20, 0.40)
    Md, Mq, S = (np.full((5, 5), np.nan) for _ in range(3))
    KN["heat_pen_demand"] = {}
    for i, dem in enumerate(dems):
        for j, pen in enumerate(pens):
            s = paired(HM, f"CAZP_pen{pen}", dem, "delay", seeds=range(1, 13))
            q = paired(HM, f"CAZP_pen{pen}", dem, "q_queue", seeds=range(1, 13))
            Md[i, j], Mq[i, j] = 100 * s["median"] / s["base"], q["median"]
            S[i, j] = s["median"] < 0 and s["pw"] < 0.05
            KN["heat_pen_demand"][f"{dem}|{pen}"] = dict(rel_median=Md[i, j], dq_median=Mq[i, j], wins=s["wins"],
                                                        n=s["n"], p_wilcoxon=s["pw"])
    fig, axs = plt.subplots(2, 2, figsize=(7.4, 6.2), gridspec_kw=dict(width_ratios=[1, 1.12]))
    xt = [f"{100 * p:g}" for p in pens]
    im = heat8(axs[0, 0], Md, xt, [str(d) for d in dems], "CV penetration (%)", "Peak demand (veh/h)",
               "(a) Delay change vs no control (%)", -55, 55, sig=S.astype(bool))
    fig.colorbar(im, ax=axs[0, 0], fraction=0.046)
    im = heat8(axs[0, 1], Mq, xt, [str(d) for d in dems], "CV penetration (%)", "Peak demand (veh/h)",
               "(b) Change in queue discharge rate (veh/h)", -330, 330, good_low=False)
    fig.colorbar(im, ax=axs[0, 1], fraction=0.046)
    accs, pens2 = (0.25, 0.5, 0.75, 1.0), (0.05, 0.10, 0.20)
    Mx, S2 = np.full((4, 3), np.nan), np.full((4, 3), False)
    pts = []
    KN["heat_acc_pen"] = {}
    for i, acc in enumerate(accs):
        for j, pen in enumerate(pens2):
            d, b = pairs(HM, f"CAZP_acc{acc}_pen{pen}", 3600, "delay", seeds=range(1, 13))
            if not len(d):
                continue
            s = summ(d, b)
            m, lo, hi = med_ci(d, b)
            Mx[i, j], S2[i, j] = m, (s["median"] < 0 and s["pw"] < 0.05)
            pts.append((acc * pen, m, lo, hi))
            KN["heat_acc_pen"][f"{acc}|{pen}"] = dict(rel_median=m, lo=lo, hi=hi, wins=s["wins"], n=s["n"],
                                                     p_wilcoxon=s["pw"])
    for pen in (0.025, 0.40):
        d, b = pairs(HM, f"CAZP_pen{pen}", 3600, "delay", seeds=range(1, 13))
        if len(d):
            pts.append((0.75 * pen,) + med_ci(d, b))
    im = heat8(axs[1, 0], Mx, [f"{100 * p:g}" for p in pens2], [f"{a:g}" for a in accs], "CV penetration (%)",
               "Acceptance rate", "(c) Delay change vs no control (%), 3600 veh/h", -55, 55, sig=S2)
    fig.colorbar(im, ax=axs[1, 0], fraction=0.046)
    ax = axs[1, 1]
    sx = np.array([p[0] for p in pts]) * 100
    sy = np.array([p[1] for p in pts])
    ax.errorbar(sx, sy, yerr=[sy - np.array([p[2] for p in pts]), np.array([p[3] for p in pts]) - sy], fmt="o", ms=3.5,
                color=COL["CAZP"], ecolor="#9cc7a8", capsize=1.5)
    try:
        f = lambda s, a, s0: -a * (1 - np.exp(-s / s0))            # noqa: E731
        par, _ = curve_fit(f, sx, sy, p0=(20, 5), bounds=([0, 0.1], [100, 60]))
        xs = np.linspace(0, sx.max() * 1.05, 100)
        ax.plot(xs, f(xs, *par), color=COL["CAZP"], lw=1)
        ax.text(0.98, 0.86, mt(f"fit: -{par[0]:.0f}% × (1 - exp(-s / {par[1]:.1f}%))"), transform=ax.transAxes,
                ha="right", va="top", fontsize=8.2)
        KN["saturation_fit"] = dict(A_pct=float(par[0]), s0_pct=float(par[1]))
    except Exception as e:                                          # noqa: BLE001
        print("fit failed", e)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Share of acting vehicles s (penetration × acceptance, %)")
    ax.set_ylabel("Median delay change vs no control (%)")
    ax.set_title("(d) Delay change against the share of acting\nvehicles, 3600 veh/h", loc="left", fontsize=9.5)
    ax.grid(color="#eee", lw=0.6)
    fig.tight_layout()
    save(fig, "Fig6_penetration")


def pacers_compliance(SN, SC, KN):
    """Section 5.5: cap on the number of simultaneously active pacers of CAZP (three demands) and of the CVs of CV speed
    pacing (3800 veh/h), KN["sens_nmax"]; acceptance of the CV drivers of CAZP and CV speed pacing at 3800 veh/h,
    KN["sens_comp"]. Paired changes against no control."""
    nmax = [(2, "CAZP_nmax2"), (4, "CAZP_nmax4"), (8, "CAZP_nmax8"), (12, "CAZP_nmax12"), ("none", "CAZP_nmaxinf")]
    pmax = [(2, "PACE_nmax2"), (4, "PACE_nmax4"), (8, "PACE_nmax8"), (12, "PACE_nmax12"), (16, "PACE_nmax16")]
    comp = {"CAZP": [(c, f"CAZP_c{c}") for c in (0.25, 0.5, 0.75, 1.0)],
            "PACE": [(c, f"PACE_c{c}") for c in (0.25, 0.5, 0.75, 1.0)]}
    KN["sens_nmax"], KN["sens_comp"] = {}, {}
    for m in ("delay", "q_queue"):
        for dem in L3:
            for x, a in nmax:
                s = paired(SN, a, dem, m)
                if s["n"] >= 2:
                    KN["sens_nmax"][f"CAZP|{dem}|{x}|{m}"] = s
        for x, a in pmax:
            s = paired(SN, a, 3800, m)
            if s["n"] >= 2:
                KN["sens_nmax"][f"PACE|3800|{x}|{m}"] = s
    for arm, pts in comp.items():
        for x, a in pts:
            for m in ("delay", "q_queue", "hb", "co2_g"):
                KN["sens_comp"][f"{arm}|{x}|{m}"] = paired(SC, a, 3800, m)


# ------------------------------------------------------------------ main
def main():
    """Figures and key numbers; parts whose runs are missing are skipped."""
    KN = {"tuned": tuned()}
    a0 = KN["tuned"][MAIN]["CAZP"]
    M = {**load("main"), **load("zipper_variants")}
    if not M:                       # without the run records nothing is written (the shipped results stay as they are)
        raise SystemExit(f"no run records in {R.RUNS}: run the experiment blocks first, or set CAZP_RUNS to their folder")
    MC = {lay: load("main", lay) for lay in ORDER}
    AB = {**load("ablation_I"), **{k: r for k, r in M.items() if k[0] in ("NC", "CAZP") and k[1] == 3800 and k[2] <= 20}}
    AL = {**load("gain"), **alias(M, "CAZP", f"fix{a0:g}", seeds=range(1, 21)),
          **{k: r for k, r in M.items() if k[0] == "NC" and k[2] <= 20}}
    S16 = range(1, 17)
    SN = {**load("pacers"), **alias(M, "CAZP", "CAZP_nmaxinf", seeds=S16), **alias(M, "NC", "NC", seeds=S16),
          **alias(M, "PACE", "PACE_nmax16", dems=(3800,), seeds=S16)}
    SC = {**load("acceptance"), **alias(M, "NC", "NC", dems=(3800,), seeds=S16),
          **alias(M, "CAZP", "CAZP_c0.75", dems=(3800,), seeds=S16),
          **alias(M, "PACE", "PACE_c0.75", dems=(3800,), seeds=S16)}
    HM = load("penetration")
    RB = {**load("robustness"), **alias(M, "NC", "NC|main", dems=(3800,), seeds=S16),
          **alias(M, "CAZP", "CAZP|main", dems=(3800,), seeds=S16)}
    PS = load("thresholds")
    print({k: len(v) for k, v in dict(M=M, AB=AB, AL=AL, SN=SN, SC=SC, HM=HM, RB=RB, PS=PS).items()})
    if M:
        bed_numbers(M, KN)
        main_table(M, KN)                                       # absolute values and results by demand: KN["main"]
        fig02(M, KN)
        fig03(M, KN)
        fig04(M, KN)
        benchmark_numbers(M, PS, KN)
        bed_extras(M, KN)
        KN["pairs_extra"] = {f"{x}-{y}": {m: pooled(M, x, m, ref=y) for m in ("delay", "q_queue", "hb")}
                             for x, y in (("VSLL_ZMSD", "VSLL"), ("VSL1_ZMSD", "VSL1"), ("CAZP_VSL1", "VSLL"))}
        ablation_mech({k: r for k, r in M.items() if k[0] in ("NC", "ZMSD", "ZMSD_c25", "ZMSD_c75", "ZMSD_c100")},
                      KN, MAIN, "zipper_mech")
    if AB and AL:
        fig05(AB, AL, KN, a0, E.gain_values())
        ablation_mech(AB, KN)
    if SN and SC:
        pacers_compliance(SN, SC, KN)
    if HM:
        fig06(HM, KN)
    C = {lay: config_stats(MC[lay], lay) for lay in ORDER if MC[lay]}
    KN["configs"] = C
    if len(C) == len(ORDER):
        KN["aggregate"] = A = aggregate(C, MC)
        if RB:
            fig07(RB, C, A, KN)
        KN["mechanism_configs"] = {lay: mechanism_numbers(MC[lay], lay) for lay in ORDER}
        pacer_ratio(MC, KN)
    AB2 = load("ablation_II", "base")
    if AB2:
        AB2.update({k: r for k, r in MC["base"].items() if k[0] in ("NC", "CAZP") and k[1] == 3800 and k[2] <= 20})
        rows = [a for a, _, g in ABL if g in ("core", "variant") and any(k[0] == a for k in AB2)]
        vs = {a: paired(AB2, a, 3800, "delay", ref="CAZP") for a in rows}
        adj = {}
        for grp in ("core", "variant"):
            fam = [a for a, _, g in ABL if g == grp and a in rows]
            adj.update(zip(fam, holm([vs[a]["p"] for a in fam])))
        KN["ablation_II"] = {a: dict(delay=paired(AB2, a, 3800, "delay"), q=paired(AB2, a, 3800, "q_queue"),
                                     vs_full=vs.get(a), p_holm=adj.get(a)) for a in ["CAZP"] + rows}
        ablation_mech(AB2, KN, "base", "ablation_mech_II")
    gain_transfer(MC, KN)
    bound_size_configs(M, MC, KN)
    acceptance_configs(MC, KN)
    compliance_levels(MC, KN)
    validity(KN)
    KN["dems_l43"] = list(E.demands("l43")[0])
    tr = {R.key(s): s for f in E.TUNING_BLOCKS.values() for s in f()}          # runs of the tuning blocks
    KN["n_training_runs"] = int(sum(os.path.isfile(R.path(s)) for s in tr.values()))
    with open(os.path.join(OUT_ROOT, "results", "key_numbers.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(KN, f, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
    print("validity", KN["validity"])


if __name__ == "__main__":
    main()

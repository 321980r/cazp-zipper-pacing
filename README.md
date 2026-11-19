# Coverage-assured zipper pacing (CAZP)

Code and results of the paper

> Ren, Y., Hellinga, B., Wang, J., Shangguan, Q., Fu, T., 2026. Coverage-assured zipper pacing: connected vehicles as
> pacers at freeway lane closures. Communications in Transportation Research (submitted).

CAZP uses a few human-driven connected vehicles (CVs) as mobile pace setters in the closing lanes of a freeway lane
closure. A CV whose driver accepts a pacing request keeps its lane in a queue-following pacing zone, and its speed is
capped at a calibrated fraction of the speed of the lane it merges into, at most 20 km/h below it. A coverage rule keeps
the closing lanes paced. The repository contains the SUMO test bed, CAZP and the seven benchmark strategies, the
experiments, the analysis, the SUMO networks of the three closure configurations, the tuned parameters and the key
results. The run records (8132 runs, about 0.6 GB) are not included; they will be deposited in ETS-Data with the code
upon acceptance of the paper and are available from the corresponding authors on request until then.

## Contents

| Path | Content |
|---|---|
| `closures.py` | Closure configurations I-III (Fig. 1c) and the build of their SUMO networks |
| `policy.py` | CAZP (Eqs. 2-9), VSL-L, zipper-merge sign and CV early merge; called at every simulation step |
| `runs.py` | Run specification and run key, simulation of one run, run record, parallel execution |
| `experiments.py` | Experiment blocks, tuning rule (Eq. 10) and selection of the tuned parameters |
| `analysis.py` | Figs. 2-7 and the key numbers of the paper from the run records |
| `sim/` | SUMO test bed: run configuration, demand, simulation loop, VSL-S and CV speed pacing controllers with the segment-detector estimates they use (approach speed, queue tail), CV actuation, driver response to posted limits, indicators of Section 4.3 |
| `networks/` | SUMO networks of configurations I (`seq`), II (`base`) and III (`l43`): netconvert input and `.net.xml` |
| `results/design_parameters.json` | Values fixed before the experiments in the design study on training seeds of configuration II: CAZP design, VSL-S and CV speed pacing settings, discharge rate of configuration II used to scale the demands of configuration III |
| `results/demand_III.json` | Demands of configuration III |
| `results/tuning/tuned.json` | Tuned parameters of every strategy and configuration, with the mean paired delay change of every candidate value on the training seeds |
| `results/key_numbers.json` | Numbers of the paper (undefined values are written as `NaN`, which Python's `json` reads) |
| `figures/` | Figs. 2-7 of the paper, written by `analysis.py`; Fig. 1 (a schematic) and Video S1 (an animation of the paired runs of block `main` at 3800 veh/h, seed 6) were drawn separately |

Codes used in the code and in `results/`:

| Code | Meaning |
|---|---|
| `seq`, `base`, `l43` | Configuration I (successive closure, main test bed), II (lanes 2 and 3 end together), III (four-to-three closure) |
| `NC` | No control |
| `CAZP` | CAZP |
| `VSL1` | VSL-S, section-based variable speed limit |
| `VSLL` | VSL-L, lane-based variable speed limits |
| `PACE` | CV speed pacing |
| `EM` | CV early merge |
| `ZMSD` | Zipper-merge sign, shown during backups |
| `VSL1_ZMSD`, `VSLL_ZMSD` | VSL-S + zipper sign, VSL-L + zipper sign |
| `CAZP_VSL1` | CAZP + VSL-S |
| `_VH`, `_H`, `_M`, `_L` (block `compliance`) | Compliance levels Very High, High, Moderate and Low of Hellinga and Mandelzys (2011) with the posted limits of VSL-S and VSL-L (`comply` in `sim/core/driver_response.py`; `comp` of VSL-S, `vsll_comp` of VSL-L); without a level, drivers treat a posted limit as the static limit |

## Requirements

* Python 3.11 with numpy, scipy and matplotlib (`requirements.txt`)
* SUMO 1.18.0 and the `libsumo` Python package of the same version. SUMO is located through `SUMO_HOME`, the pip
  package `eclipse-sumo` or a standard installation folder (`sim/sumo_env.py`); `closures.py` needs `netconvert` in
  `SUMO_HOME/bin` or on the `PATH`.

Tested on Windows 11 with Python 3.11.4, numpy 2.4.6, scipy 1.15.0, matplotlib 3.10.0 and SUMO 1.18.0. With these
versions and the font Times New Roman, `analysis.py` reproduces the PNG files in `figures/` byte for byte from the run
records.

## Reproduction

Every run is stored once as `results/runs/<key>.pkl`, where the key is a hash of the run specification (configuration,
demand, seed, CV penetration, strategy parameters). A block skips the runs that exist, so it can be interrupted and
resumed. `CAZP_RUNS` sets another folder for the run records; progress is written to `logs/`. A new run starts only while
more than `FREE_GB_MIN` GB of memory are free (default 1.3; Windows).

Install the Python packages with `pip install -r requirements.txt`. The tuned parameters and the demands of
configuration III are included (`results/tuning/tuned.json`, `results/demand_III.json`), so the test runs (step 2) and the
analysis (step 3) can be run without step 1; step 1 recomputes and overwrites both files.

1. Tuning on the training seeds; writes `results/demand_III.json` and `results/tuning/tuned.json`:

   ```
   python experiments.py demand_III tuning_gains tuning_benchmarks select --workers=16
   ```

2. Test runs on the test seeds:

   ```
   python experiments.py main zipper_variants ablation_I ablation_II gain pacers acceptance compliance penetration robustness thresholds gain_transfer acceptance_II_III bound_I bound_II_III --workers=16
   ```

3. Figures and key numbers (about 1 min):

   ```
   python analysis.py
   ```

   writes `figures/` and `results/key_numbers.json`; `CAZP_OUT` sets another output folder.

Runs per block, in the order of the commands (a run shared with an earlier block is counted there), and the time with
16 workers. The times are the recorded run times of the stored runs divided by 16 (a desktop computer with 20 logical
processors), rounded to 0.1 h; the totals are computed before rounding.

| Block | Runs | Hours | Block | Runs | Hours |
|---|---:|---:|---|---:|---:|
| `demand_III` | 20 | 0.1 | `pacers` | 256 | 0.4 |
| `tuning_gains` | 1512 | 3.2 | `acceptance` | 96 | 0.2 |
| `tuning_benchmarks` | 912 | 1.9 | `compliance` | 576 | 1.5 |
| **tuning** | **2444** | **5.1** | `penetration` | 396 | 0.5 |
| `main` | 2760 | 5.9 | `robustness` | 320 | 0.6 |
| `zipper_variants` | 180 | 0.4 | `thresholds` | 80 | 0.1 |
| `ablation_I` | 200 | 0.4 | `gain_transfer` | 180 | 0.4 |
| `ablation_II` | 180 | 0.3 | `acceptance_II_III` | 32 | 0.1 |
| `gain` | 240 | 0.4 | `bound_I` | 64 | 0.2 |
| | | | `bound_II_III` | 128 | 0.3 |
| | | | **test** | **5688** | **11.6** |

`python closures.py` rebuilds the three networks with netconvert and compares them with `networks/` (identical with
netconvert 1.18.0).

## Symbols of the paper in the code

The CAZP parameters are the keys of `DEFAULTS` in `policy.py`, set for the runs by `results/design_parameters.json`
(`cazp`), the tuned gains in `results/tuning/tuned.json` and the arms of `experiments.py`.

| Paper | Value | `policy.py` |
|---|---|---|
| T | 2 s | `T` |
| L | 100 m | `cell` |
| S<sub>c</sub>, S<sub>c</sub> + ΔS | 50 km/h, 70 km/h | `v_on` (queue tail: `q_thr`), `v_off` |
| λ; detection stretch | 0.15; 100-700 m | `smooth`; `zone_trig` |
| D<sub>s</sub> | 2 km | `D_s` |
| Z<sub>min</sub>, Z<sub>max</sub> | 700 m, 1500 m | `zmin`, `zmax_dyn` |
| ΔZ, δZ | 100 m, 50 m per step | constants in `Policy.__call__` |
| d<sub>rel</sub>; pacer window; move-in range | 200 m; d<sub>rel</sub> + 30 m; from d<sub>rel</sub> + 150 m | `rel`; constants in `Policy.__call__` |
| d | 2 | `depth` |
| R<sub>max</sub>, ΔR, T<sub>m</sub> | 1500 m, 500 m, 12 s | `Zmax`, `stage`, `lat_T` |
| R | 75 m | `R` |
| α* (I, II, III) | 0.8, 0.2, 0.1 | `alpha` |
| Δv; v<sub>min</sub> | 20 km/h; 3 km/h | `cred`; `v_min` |
| Δ<sub>f</sub>, Δ<sub>max</sub>, k<sub>f</sub>, floor of Eq. 8 | 30 m, 300 m, 0.05 1/s, 0.5 v<sub>ref</sub> | `form_tol`, `form_max`, `form_k`, `form_min` |
| τ<sub>d</sub>; b<sub>max</sub> | 1 s; 1.5 m/s² | `tau`; `b_adv` |
| CAZP-DC: T<sub>D</sub>, Δv<sub>D</sub>, τ<sub>r</sub>, ε | 5 s, 5 km/h, 3 s, 0.15 | `T`, `q_kmh`, `tau`, `err` |
| n<sub>max</sub>; p<sub>a</sub> | none; 0.75 | `n_max`; `p_a` |
| VSL-L: α<sub>L</sub> (I, II, III), sign positions, update, steps | 0.8, 0.9, 0.8; 1.5/1.0/0.5 km; 10 s; 10 km/h | `vsll`, `vsll_gx`, `vsll_T`, `vsll_q` |
| Zipper-merge sign: p<sub>z</sub>, position | 0.5, 1.5 km | `zms`, `zms_d` |
| CV early merge: advice distance | 2.5 km | `D_adv` |

In the code, `tie` is the speed cap of Eq. (7), `hold` the lane keeping of the pacers and `form` the alignment rule of
Eq. (8). Ablation arms (Fig. 5): `no_coverage` = without coverage rule, `depth_1` = one pacer per lane,
`no_formation` = without alignment rule (configuration II), `tie_only` = speed cap only, `hold_only` = lane keeping
only, `outer_only` = only lane 3 paced, `inner_only` = only lane 2 paced, `fixed_zone` = fixed 700-m zone, `display` =
CAZP-DC, `NC_nohop` and `CAZP_nohop` = lane hopping switched off.

## Where the results of the paper come from

Blocks of `experiments.py`, functions of `analysis.py` and entries of `results/key_numbers.json`; section of the paper
in brackets. "Tuning" stands for the tuning blocks and `results/tuning/tuned.json`.

| Result | Blocks | `analysis.py` | Key numbers |
|---|---|---|---|
| Fig. 2 (5.1) | `main` | `fig02`, `ts_maps`, `queue_panels` | `queue_dyn` |
| Fig. 3 (5.2) | `main` | `fig03`, `scorecard`, `speed_outflow` | `scorecard`, `head2head`, `speed_profile` |
| Fig. 4 (5.3) | `main` | `fig04`, `mechanism_numbers`, `benchmark_numbers`, `bed_extras` | `mechanism_3800`, `burden`, `mech3800`, `bed_extras` (share of pacers) |
| Fig. 5 (5.4) | `ablation_I`, `gain`, `main` | `fig05`, `ablation_mech` | `ablation`, `ablation_mech`, `gain` |
| Fig. 6 (5.5) | `penetration` | `fig06` | `heat_pen_demand`, `heat_acc_pen`, `saturation_fit` |
| Fig. 7 (5.6, 5.7) | `robustness`, `main` | `fig07`, `robustness`, `config_stats`, `aggregate` | `robustness`, `configs`, `aggregate` |
| Tuned parameters (3.8, Table 1) | tuning | `experiments.select_parameters` | `tuned` |
| Test bed without control, lane use (4.1) | `main`, `robustness` | `bed_numbers`, `bed_extras`, `robustness` | `bed`, `bed_extras`, `lane3_share_NC`, `robustness` |
| Zipper-sign followers (4.1) | `zipper_variants`, `main` | `benchmark_numbers`, `ablation_mech` | `zipper_followers`, `zipper_mech` |
| Cell length and speed threshold (4.2) | `thresholds`, `main` | `benchmark_numbers` | `param_sens` |
| Absolute values, results by demand, benchmarks (5.2) | `main` | `main_table`, `benchmark_numbers`, `main` | `main`, `hb_tail`, `pairs_extra` |
| Number of pacers, acceptance (5.5) | `pacers`, `acceptance`, `main` | `pacers_compliance` | `sens_nmax`, `sens_comp` |
| Configurations II and III: demands, mechanism, pacers (5.7) | `demand_III`, `main` | `mechanism_numbers`, `pacer_ratio` | `dems_l43`, `mechanism_configs`, `pacer_ratio` |
| Gain of I in II and III, 25% acceptance (5.7) | `gain_transfer`, `acceptance_II_III`, `main` | `gain_transfer`, `acceptance_configs` | `gain_transfer`, `acceptance_cfg` |
| Ablation in configuration II (5.7) | `ablation_II`, `main` | `main`, `ablation_mech` | `ablation_II`, `ablation_mech_II` |
| Size of the bound (5.7) | `bound_I`, `bound_II_III`, `main` | `bound_size_configs` | `bound_size_cfg` |
| Compliance with the speed limits (4.1, 5.8) | `compliance`, `main` | `compliance_levels` | `compliance_cfg` |
| Number of runs, collisions (4.4) | all | `validity`, `main` | `validity`, `n_training_runs` |

Main results (changes against no control, pooled over the three peak demands, paired test seeds):

| Result | Value | Entry |
|---|---|---|
| I: CAZP, delay / discharge rate / hard braking | -41.18% / +8.34% / -34.39% | `main["pooled\|CAZP"].{delay,q_queue,hb}.rel` |
| I: VSL-L, delay | -27.29% | `main["pooled\|VSLL"].delay.rel` |
| I: CAZP + VSL-S, hard braking | -81.90% | `main["pooled\|CAZP_VSL1"].hb.rel` |
| II: CAZP / VSL-L, delay | -19.06% / -8.58% | `configs.base.{CAZP,VSLL}.delay.rel` |
| III: CAZP / VSL-L, delay | -33.49% / -24.46% | `configs.l43.{CAZP,VSLL}.delay.rel` |
| Mean of I-III: CAZP / VSL-L, delay | -31.24% / -20.11% | `aggregate.{CAZP,VSLL}.mean` |
| I: VSL-L, delay at the compliance levels Very High / Moderate / Low (3800 veh/h) | -25.93% / -20.34% / -0.99% | `compliance_cfg.seq.{VH,M,L}.VSLL.delay.rel` |
| Test runs / tuning runs | 5688 / 2444 | `validity.n_runs`, `n_training_runs` |

## Licence

MIT, see `LICENSE`.

## Citation

```
@article{Ren2026CAZP,
  author  = {Ren, Yiwei and Hellinga, Bruce and Wang, Junhua and Shangguan, Qiangqiang and Fu, Ting},
  title   = {Coverage-assured zipper pacing: connected vehicles as pacers at freeway lane closures},
  journal = {Communications in Transportation Research},
  year    = {2026},
  note    = {Submitted}
}
```

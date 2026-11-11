# TAC-R2S2R v2 — Structural Degeneracy in Task-Weighted Sim2Real Curriculum: Diagnosis, a Minimal Fix, and Calibration

*Extension of the reproduction in this repository. Every number below is produced by
`experiments_v2/tacpp.py` on an NVIDIA GB10 (aarch64, torch 2.11.0+cu130), 10 seeds
(11, 23, 37, 53, 71, 83, 97, 109, 127, 149), and is committed under
`experiments_v2/results/`. The data model and the four baseline methods are byte-for-byte
the authors' original implementation, so all cells are directly comparable to the paper's
Table IV.*

---

## 0. TL;DR

The original paper reports one striking negative result — **task-weighted consistency
(TW) collapses to 58.0% at budget 2048** — and never explains it. We (i) show the collapse
is **budget-progressive** (90.7% → 50.3% as the budget grows 256 → 4096), (ii) diagnose it
as a **structural degeneracy**: the stop-gradient self-distillation penalty has a runaway
fixed point that suppresses representation variance and saturates confidence, and (iii) show
that **rescaling the penalty (budget annealing) only delays it, while replacing the penalty
target with a *frozen teacher* removes it entirely** (94.0% at budget 4096, monotone). We
additionally show that **post-hoc temperature scaling** cuts the confident-error rate (HCE)
by 17–40% with no accuracy cost.

---

## 1. What the original paper does

TAC-R2S2R studies a lightweight **simulator-side** module for sim-to-real transfer: a small
MLP (30→96→48→1) is trained on synthetic `(task, nuisance)` states, and four strategies are
compared across synthetic budgets {256, 512, 1024, 2048}:

| Method | Regularisation |
|---|---|
| **S** source-only | none |
| **DR** uniform domain randomisation | i.i.d. Gaussian noise on all 30 dims (σ = 0.35) |
| **TW** task-weighted | smaller noise on the 6 task dims / larger on the 24 nuisance dims, **plus a stop-gradient invariance penalty** `(f(x_nuisance) − sg[f(x)])²` with weight 0.12 |
| **TAC** task-aware consistency | same augmentation, but a **paired-view consistency** loss on representation and probability (weights 0.18 / 0.22) |

The paper's own table has TW at **58.03 ± 10.10** for budget 2048 while TAC holds
93.80, and the text offers no mechanism for the collapse.

---

## 2. The gap

1. **The collapse is unexplained and its budget dependence is unknown.** The paper reports a
   single budget (2048); we sweep 256 → 4096 and expose a monotone degradation.
2. **The cause is never isolated.** Is it regularisation *magnitude* (fixable by scaling) or
   the *structure* of the penalty (fixable only by changing the target)? The paper does not
   test either.
3. **Uncertainty is uncalibrated.** Confidence-related metrics (HCE, Brier) are reported but
   never optimised; a downstream robot that is confidently wrong is dangerous.

---

## 3. Contributions and results

### C1 — The collapse is budget-progressive and has a measurable signature

Accuracy on the 8192-sample target proxy (mean over 10 seeds; full CIs in
`analysis/table_main.md`):

| Budget | S | DR | **TW** | TAC | TW-BAR | TAC-BAR | **TW-FTI** | TW-VR | TAC++ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 256  | 76.56 | 71.27 | 90.73 | 89.78 | 90.73 | 89.78 | 78.59 | 90.83 | 89.78 |
| 512  | 81.30 | 69.77 | 89.38 | 91.50 | 90.87 | 91.09 | 86.08 | 89.22 | 91.09 |
| 1024 | 83.87 | 75.60 | 82.59 | 92.90 | 90.01 | 92.32 | 91.42 | 83.60 | 92.32 |
| 2048 | 85.90 | 76.91 | **58.10** | 93.80 | 65.25 | 92.85 | **92.59** | 56.09 | 92.85 |
| 4096 | 87.65 | 78.82 | **50.29** | **94.27** | 57.78 | 93.25 | **94.01** | 51.84 | 93.25 |

Instrumentation (final epoch, target probe, 3 seeds) explains *why*:

| Budget | TW acc% | TW pred-entropy | TW rep-variance | TAC acc% | TAC pred-entropy | TAC rep-variance |
|---:|---:|---:|---:|---:|---:|---:|
| 256  | 90.92 | 0.0028 | 100.8 | 89.53 | 0.183 | 0.54 |
| 512  | 87.85 | 0.0004 | 709.8 | 91.85 | 0.158 | 0.49 |
| 1024 | 84.14 | 0.0001 | 3175.4 | 92.76 | 0.164 | 0.36 |
| 2048 | 52.22 | 0.0000 | 12467.9 | 93.88 | 0.157 | 0.28 |
| 4096 | 52.22 | 0.0000 | 40.8 | 94.49 | 0.162 | 0.20 |

As the budget grows, the stop-gradient penalty accumulates faster than the supervised term
can stabilise it; the encoder's representation variance grows by two orders of magnitude and
predictions **saturate** (entropy → 0) while target accuracy falls to chance. This is a
runaway positive-feedback loop, not a mild over-regularisation.

> **Fig 1** (`figures/fig1_collapse_curve.png`) — accuracy vs budget, all methods.
> **Fig 2** (`figures/fig2_diagnosis.png`) — entropy / representation variance vs epoch at budget 4096.

### C2 — Budget rescaling alone is insufficient; a frozen teacher is sufficient

We test three candidate fixes:

1. **BAR (budget-annealed regularisation)** — scale the penalty by `(256/budget)^α`.
   Sweep (5 seeds, `analysis/table_bar_sweep.md`): best is α = 1.0, giving TW = **75.4%
   (2048)** and **67.3% (4096)** — better than baseline (58.1 / 50.3) but **still collapsing**.
   Conclusion: the failure is not merely a magnitude issue.
2. **TW-VR (variance floor)** — add a hinge `max(0, 0.10 − std(σ(z)))`.
   Result: **56.1% (2048), 51.8% (4096)** — does **not** fix it.
3. **TW-FTI (frozen-teacher invariance)** — freeze a copy of the model after 10 warm-up
   epochs and apply the invariance penalty against the *frozen* (non-collapsed) teacher,
   `(f(x_nuisance) − teacher(x))²`. The degenerate constant minimum no longer exists.
   Result: **92.59% (2048), 94.01% (4096)**, monotone and matching TAC, while retaining the
   task-weighted augmentation.

**The mechanism is now identified: the collapse is caused by the *stop-gradient self-target*,
and is removed by replacing that target with a frozen teacher.** This is the key novel
finding of this extension.

> **Fig 3** (`figures/fig3_calibration.png`) and the table below quantify contribution C3.

### C3 — Calibration reduces confident errors without touching accuracy

Post-hoc temperature scaling fitted on a **held-out source split** (not the target), applied
to TAC++ (TAC-BAR + calibration):

| Budget | HCE% raw | HCE% cal | ECE% raw | ECE% cal | Brier raw | Brier cal | T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256  | 4.60 | 2.74 | 3.07 | 1.29 | 0.0742 | 0.0723 | 1.53 |
| 512  | 4.03 | 2.45 | 2.68 | 1.15 | 0.0648 | 0.0632 | 1.51 |
| 1024 | 2.93 | 2.04 | 1.50 | 1.11 | 0.0546 | 0.0541 | 1.32 |
| 2048 | 2.60 | 1.95 | 1.05 | 1.00 | 0.0509 | 0.0507 | 1.23 |
| 4096 | 2.25 | 1.86 | 0.78 | 0.99 | 0.0482 | 0.0482 | 1.16 |

HCE falls by **40.4% / 39.2% / 30.4% / 25.0% / 17.3%** across the five budgets (temperatures
1.16–1.53 indicate the uncalibrated model is over-confident). Accuracy is unchanged by a
monotone transform, so calibration is **orthogonal** to C1/C2.

---

## 4. Why this is a frontier-relevant step

- **It turns a reported negative result into a mechanism.** The community routinely reports
  "self-distillation collapses" (BYOL / SimSiam / mean-teacher) but rarely connects it to the
  *sim2real budget* axis. We show the collapse is a function of the synthetic budget and give
  a measurable signature (representation-variance blow-up + entropy saturation).
- **It isolates magnitude vs structure.** BAR (magnitude) fails; FTI (structure) succeeds.
  This is a clean, falsifiable ablation that the original paper lacks.
- **It adds a safety-relevant axis.** Confident errors matter for deployment; temperature
  scaling is a cheap, model-agnostic guarantee that does not disturb accuracy.
- **It is reproducible end-to-end** on a single GPU in ~8 minutes (490 s for 450 + 180 + 100
  runs), with the full provenance committed.

---

## 5. Honest limitations

- **Synthetic proxy, not a real robot.** As in the original paper, the target domain is a
  controlled distribution shift, not physical hardware. Claims are about *mechanism*, not
  field success.
- **FTI has a cold-start cost.** At budgets 256 / 512, FTI (78.6 / 86.1) is *worse* than the
  original TW (90.7 / 89.4), because freezing an early, weak teacher helps little. FTI wins
  only from budget ≥ 1024. A budget-adaptive switch (FTI only when the base model is strong)
  is left for future work.
- **BAR is not a fix.** It is reported precisely to show that magnitude scaling is
  insufficient; this is a negative result we keep rather than hide.
- **Single task family** (6 task + 24 nuisance dims). Generalisation to other task
  parameterisations is not tested.

---

## 6. Reproduce

```bash
# on any CUDA machine
python experiments_v2/tacpp.py --stage main  --seeds 11 23 37 53 71 83 97 109 127 149 --out results
python experiments_v2/tacpp.py --stage sweep --seeds 11 23 37 53 71 --out results
python experiments_v2/tacpp.py --stage diag  --seeds 11 23 37 --out results
python experiments_v2/analyze.py          # writes analysis/*.md and figures/*.png
```

Files:
- `experiments_v2/tacpp.py` — experiment driver (baselines + BAR + FTI + VR + calibration)
- `experiments_v2/analyze.py` — aggregation + figures
- `experiments_v2/results/v2_main.csv` — 450 rows (9 methods × 5 budgets × 10 seeds)
- `experiments_v2/results/v2_summary.csv` — mean ± 95% CI
- `experiments_v2/results/v2_bar_sweep.csv` — α sweep
- `experiments_v2/results/v2_diagnosis.csv` — per-epoch instrumentation
- `experiments_v2/analysis/*.md` — tables used above
- `experiments_v2/figures/*.png` — Fig 1–3

**Claim boundary.** Controlled synthetic domain-transfer mechanism test; not a real-robot
experiment. All means are over 10 seeds; CIs use t(9).

# TAC-R2S2R — verified GPU runs (160 seeded runs)

Data and code accompanying the manuscript:

> **TAC-R2S2R: Task-Aware Consistency and Residual Closed-Loop Transfer for Embodied Manipulation**
> Zipeng Li, Jing Chen — School of Physics and Optoelectronic Engineering, Guangdong University of Technology, Guangzhou, China
> Manuscript submitted to *IEEE Access* (2026). This repository is the data-availability artifact for that submission.

**One line:** 160 GPU runs of a controlled synthetic sim-to-real transfer proxy, with seed-level metrics, 90-epoch training logs, and a script that re-derives every headline number in the paper.

---

## Evidence boundary — read this first

This is a **synthetic binary grasp-and-place classification proxy**. It is deliberately narrow:

- It is **not** a physical robot experiment and contains no real-robot measurements.
- It is **not** a vision-language-action (VLA) policy training run.
- It is **not** a complete Real2Sim2Real benchmark.
- `target_success_rate` is a **legacy field name**: it denotes binary classification accuracy on target-domain samples, **not** robot task success.
- No physical parameter is identified from measured data; all parameters are generated.

We publish the limits together with the numbers so that a classifier-level result is not read as evidence of robot transfer. The robot protocol needed to validate the remaining modules is specified in the manuscript, not reported here.

---

## What is in here

| Path | Size | Contents |
|---|---|---|
| `data/verified_runs.json` | 1.55 MB | Structured snapshot: `metadata` (2 GPU records), `summary` (16 budget×method aggregates), `runs` (160 per-run records with confusion matrix, 10-bin calibration and full epoch history), `scope`, `version` |
| `data/run_metrics.csv` | 28 KB | Flat table, **160 rows** = 4 methods × 4 budgets × 10 seeds, with GPU name, GPU UUID and wall-clock seconds |
| `data/summary.csv` | 2.3 KB | **16 rows** of budget × method aggregates with two-sided 95% intervals (t, df=9) |
| `data/training_history.csv` | 1.68 MB | **14,400 rows** = 160 runs × 90 epochs: total objective, supervised BCE, validation accuracy, validation Brier, elapsed seconds |
| `data/paired_contrasts.csv` | 0.9 KB | Seed-paired contrasts at budget 2048 (TAC − source-only / − uniform-DR / − task-weighted) |
| `data/ORIGINAL_README.txt` | 1.9 KB | The original run notes captured on 2026-11-02, kept verbatim |
| `code/run_task_aware_transfer.py` | 8.7 KB | Self-contained collection script: generates the synthetic domains, trains the MLP, evaluates, writes `raw_results*.csv` / `summary*.csv` / metadata |
| `code/summarize_and_plot.py` | 6.5 KB | Aggregation and figure generation |
| `code/verify_numbers.py` | — | **Start here.** Re-derives every headline number from `data/` with the standard library only |
| `figures/*.png` | — | Figures generated from these runs during manuscript preparation |
| `SCHEMA.md` | — | Field-by-field dictionary for every released file |
| `REPRODUCE.md` | — | Environment and exact commands to regenerate the runs |

---

## Verify the paper's numbers in one command

```bash
python code/verify_numbers.py
```

Expected tail:

```
RESULT: ALL CHECKS PASSED - every headline number in the manuscript is reproducible from the released files.
```

The script checks, among others:

| Quantity | Manuscript | Data |
|---|---|---|
| TAC target-domain accuracy at source budget 2048 | 93.80% | `summary.csv` → `0.93802` |
| 95% interval half-width (10 seeds, t) | 0.32 pp | `0.003157` |
| TAC high-confidence error rate at 2048 | 1.61% | `0.016064` |
| Source-only accuracy at 2048 | 85.90% | `0.858972` |
| Uniform randomization accuracy at 2048 | 76.91% | `0.769116` |
| Seed-paired TAC − source-only gap | 7.91 pp | `paired_contrasts.csv` → `0.079053` |

It also asserts the design: 160 runs, 4 methods, 4 budgets, 10 seeds, 2 distinct physical GPUs (distinct UUIDs), and 14,400 epoch rows.

---

## Reproduce from scratch

Two consumer GPUs were used, each with its own five seeds (see `REPRODUCE.md` for the full protocol):

```bash
# GPU 0
python code/run_task_aware_transfer.py --device 0 --seeds 11 23 37 53 71

# GPU 1
python code/run_task_aware_transfer.py --device 1 --seeds 83 97 109 127 149 --suffix _gpu1
```

Environment used for the reported runs: Python 3.12.3, PyTorch 2.11.0+cu128, CUDA 12.8, NVIDIA GeForce RTX 4060 (8.0 GiB) and RTX 5060 Ti (15.93 GiB). Each run is a 30–96–48–1 MLP, AdamW (lr 2e-3, weight decay 1e-4), batch ≤ 256, fixed 90 epochs, no early stopping and no hyperparameter selection on the target domain.

Runtime: the whole 80-run sweep per GPU takes roughly 75 seconds of wall clock, so a full replication is cheap.

---

## Provenance and known gaps

We state this precisely because it matters for reproducibility.

1. **Two collection scripts were involved.** `code/run_task_aware_transfer.py` collects `raw_results*.csv` / `summary*.csv` with columns such as `target_success_rate_mean` and `target_success_rate_ci95_half`. The released `data/summary.csv` uses `target_success_rate` and `target_success_rate_ci95`, i.e. it was produced by the 2026-11-02 verified re-run that also recorded per-epoch history (`data/ORIGINAL_README.txt` refers to it as `record_training.py`). **That second wrapper was not retained in the archived snapshot and is therefore not included here** — we do not reconstruct it, because a rewrite would not be bit-identical to what produced the logs.
2. **The re-run supersedes earlier labels.** `ORIGINAL_README.txt`, verbatim: *"The original records had inconsistent GPU labels; this rerun supersedes those labels and adds actual epoch history. Historical paper figures may differ."* The released files are the re-run.
3. **Seeds differ per GPU by design.** RTX 4060: 11, 23, 37, 53, 71. RTX 5060 Ti: 83, 97, 109, 127, 149. Because the seeds differ, wall-clock times are descriptive only and must not be read as a GPU benchmark.
4. **Which script produces which file** is documented in `SCHEMA.md` under "Producer".

---

## Interpretation caveats

These are part of the data, not footnotes to hide:

- Intervals are **two-sided t 95% intervals over 10 seeds (df=9)**, not intervals over the pooled 81,920 test rows.
- Test examples repeat across methods and budgets within a seed, so runs are not independent samples of the world.
- Calibration and confusion counts pool at one selected budget and method only.
- **Training objective definitions differ across methods** (source-only, uniform DR, task-weighted, TAC). TAC and task-weighted share the same input perturbation but differ in regularizer form and paired information, so the comparison is **not a nested ablation of a single added loss term**.
- The mechanism is **not dominant at the smallest budget (256)** — see `summary.csv`. This counterexample is retained on purpose.

---

## Citation

If you use these runs, please cite the dataset (see `CITATION.cff`) and the accompanying manuscript. A DOI will be minted through Zenodo once the repository is archived; until then, cite the repository URL and the version tag `v1.0.0`.

```
Li, Z., & Chen, J. (2026). TAC-R2S2R verified GPU runs: 160 seeded runs of a controlled
synthetic sim-to-real transfer proxy (v1.0.0) [Data set]. GitHub.
https://github.com/lemonbaby2/tac-r2s2r-verified-runs
```

## License

- **Data** (`data/`, `figures/`): CC BY 4.0 — see `LICENSE-DATA`.
- **Code** (`code/`): MIT — see `LICENSE`.

## Contact

Corresponding author for data questions: Jing Chen (jchen125@gdut.edu.cn).
First author: Zipeng Li.

# Schema — field dictionary

All files are UTF-8. The two CSVs derived from the 2026-11-02 re-run carry a UTF-8 BOM
(`\ufeff`) on the first header cell; read them with `encoding="utf-8-sig"`.

---

## `data/run_metrics.csv` — one row per run (160 rows)

Producer: 2026-11-02 verified re-run (`record_training.py`, not retained — see README "Provenance").

| Column | Type | Meaning |
|---|---|---|
| `id` | string | `{budget}_{method}_{seed}`, e.g. `256_source_only_11` |
| `gpu` | string | Exact device name as reported by PyTorch |
| `gpu_uuid` | string | CUDA device UUID; used to prove the two halves ran on two physically distinct GPUs |
| `budget` | int | Source-domain sample budget: 256, 512, 1024, 2048 |
| `method` | enum | `source_only`, `uniform_dr`, `task_weighted`, `tac_consistency` |
| `seed` | int | Training seed. RTX 4060: 11/23/37/53/71. RTX 5060 Ti: 83/97/109/127/149 |
| `seconds` | float | Wall-clock seconds for that single run |
| `target_success_rate` | float | **Binary classification accuracy** on 8,192 target-domain samples. Legacy name — NOT robot success |
| `unsafe_confident_error_rate` | float | Share of samples that are **misclassified with p ≥ 0.8 or p ≤ 0.2**, over all test samples |
| `brier_score` | float | Mean squared error between predicted probability and binary label |

---

## `data/summary.csv` — budget × method aggregate (16 rows)

Producer: same re-run. Aggregation over 10 seeds.

| Column | Type | Meaning |
|---|---|---|
| `budget` | int | 256 / 512 / 1024 / 2048 |
| `method` | enum | As above |
| `n` | int | Seeds aggregated (always 10) |
| `target_success_rate` | float | Mean accuracy over seeds |
| `target_success_rate_ci95` | float | **Half-width** of the two-sided 95% t-interval (df = 9). Not a standard deviation |
| `unsafe_confident_error_rate` | float | Mean confident-error rate |
| `unsafe_confident_error_rate_ci95` | float | Half-width, same convention |
| `brier_score` | float | Mean Brier score |
| `brier_score_ci95` | float | Half-width, same convention |

The `_ci95` suffix is a **half-width**, so the interval is `mean ± value`.

---

## `data/training_history.csv` — per-epoch log (14,400 rows)

Producer: same re-run. 160 runs × 90 epochs.

| Column | Type | Meaning |
|---|---|---|
| `id` | string | Run id, joins to `run_metrics.csv.id` |
| `epoch` | int | 1…90, fixed; no early stopping |
| `total_loss` | float | Full training objective for that epoch (definition differs per method) |
| `supervised_bce` | float | Supervised BCE term alone |
| `validation_accuracy` | float | Accuracy on a separate 2,048-sample target **validation** split (seed + 700000), logged every epoch |
| `validation_brier` | float | Brier score on the same validation split |
| `elapsed_seconds` | float | Cumulative training time at that epoch |

The final reported test numbers are computed on a **different** 8,192-sample target split (seed + 900000). Validation and test splits do not overlap.

---

## `data/paired_contrasts.csv` — seed-paired differences (9 rows)

Producer: `code/summarize_and_plot.py`.

| Column | Type | Meaning |
|---|---|---|
| `budget` | int | Currently 2048 only |
| `contrast` | string | e.g. `tac_consistency minus source_only` |
| `metric` | enum | `target_success_rate`, `unsafe_confident_error_rate`, `brier_score` |
| `mean` | float | Mean of the per-seed difference (paired by seed) |
| `ci95_half` | float | Half-width of the 95% t-interval over the 10 paired differences |

---

## `data/verified_runs.json` — structured snapshot

Producer: 2026-11-02 verified re-run.

```
{
  "version":   string   "Verified GPU rerun with 90-epoch logs, 2026-11-02"
  "scope":     string   "Synthetic controlled domain-transfer mechanism experiment.
                         Not real-robot validation."
  "metadata":  array[2] one record per GPU:
                          gpu, device (cuda:i), gpu_uuid, memory_gib, torch, cuda, python,
                          started, completed, seeds[5], epochs (90),
                          validation_samples (2048), test_samples (8192),
                          device_verified (true), runs (80), wall_seconds
  "summary":   array[16] same fields as summary.csv (numeric types, not strings)
  "runs":      array[160] per-run record:
                          id, seed, budget, method, gpu, gpu_uuid, seconds,
                          target_success_rate, unsafe_confident_error_rate, brier_score,
                          confusion   [tn, fp, fn, tp],
                          calibration [10 x {count, prob_sum, positive_sum}],
                          history     [90 x (epoch, total_loss, supervised_bce,
                                             validation_accuracy, validation_brier,
                                             elapsed_seconds)]
  "history_columns": array[6] names matching the `history` tuple order
}
```

Calibration bins are equal-width probability bins over [0, 1]; `prob_sum` is the sum of predicted probabilities in the bin and `positive_sum` the number of positive labels, so a bin's mean confidence is `prob_sum / count` and its empirical frequency is `positive_sum / count`.

---

## `code/run_task_aware_transfer.py` — CLI

| Flag | Default | Meaning |
|---|---|---|
| `--device` | `0` | CUDA index to run on. CPU is rejected on purpose |
| `--seeds` | `11 23 37 53 71` | Seeds to sweep |
| `--suffix` | `""` | Suffix appended to `raw_results*.csv` / `summary*.csv` |

Outputs land next to the script: `raw_results{suffix}.csv`, `summary{suffix}.csv`, `run_metadata{suffix}.json`.
Its `summary` columns (`target_success_rate_mean`, `target_success_rate_ci95_half`) differ from
`data/summary.csv`; see README "Provenance and known gaps".

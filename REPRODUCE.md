# Reproducing the runs

Everything here runs on a single consumer GPU in about 75 seconds per 80-run sweep.

## 1. Environment

Reported runs used:

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU A | NVIDIA GeForce RTX 4060, 8.00 GiB (UUID `a5f9610b-7d58-d0e3-29e9-9ae29e54cdbf`) |
| GPU B | NVIDIA GeForce RTX 5060 Ti, 15.93 GiB (UUID `a81ae40f-4b9e-3fb5-2ea7-f081de446fee`) |

Only `numpy` and `torch` are required. The verification script uses the standard library only.

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

The collection script refuses to run on CPU by design (`RuntimeError("CUDA GPU is required for this experiment")`).

## 2. What is fixed (no tuning, no early stopping)

| Item | Value |
|---|---|
| Input | 30 dims: 6 task variables (position error, aperture, friction, mass, delay, clearance) + 24 nuisance appearance variables |
| Labels | Determined **only** by the 6 task variables, through a nonlinear contact-inspired margin plus sensing noise |
| Model | MLP 30–96–48–1, GELU + LayerNorm |
| Optimizer | AdamW, lr 2e-3, weight decay 1e-4 |
| Batch | ≤ 256 |
| Epochs | 90 fixed |
| Budgets | 256, 512, 1024, 2048 source samples |
| Target validation | 2,048 samples, seed + 700000 |
| Target test | 8,192 samples, seed + 900000 |
| Seeds A (RTX 4060) | 11, 23, 37, 53, 71 |
| Seeds B (RTX 5060 Ti) | 83, 97, 109, 127, 149 |

Methods: `source_only` (no perturbation), `uniform_dr` (Gaussian noise σ = 0.35 on all 30 dims),
`task_weighted` (σ = 0.18 on task dims, σ = 0.70 on nuisance dims, plus a nuisance-sensitivity penalty),
`tac_consistency` (same perturbation as `task_weighted`, plus paired representation and prediction consistency).

## 3. Two-GPU protocol

```bash
# terminal 1 — GPU 0
python code/run_task_aware_transfer.py --device 0 --seeds 11 23 37 53 71

# terminal 2 — GPU 1
python code/run_task_aware_transfer.py --device 1 --seeds 83 97 109 127 149 --suffix _gpu1
```

Outputs appear next to the script:

```
code/raw_results.csv          code/summary.csv          code/run_metadata.json
code/raw_results_gpu1.csv     code/summary_gpu1.csv     code/run_metadata_gpu1.json
```

Each script run prints one line per run so progress is visible in the terminal.

## 4. Aggregate and plot

```bash
python code/summarize_and_plot.py
```

Produces the paired contrasts and the figures in `figures/`.

## 5. Verify against the manuscript

```bash
python code/verify_numbers.py
```

This is the check a reviewer should run first: it needs no GPU and no network, and exits non-zero
if any number reported in the manuscript stops matching the released data.

## 6. Expected magnitudes

Not a byte-identical guarantee, because GPU nondeterminism and differing seeds shift individual runs,
but the published run should reproduce at this scale:

| Budget | Method | Target accuracy (mean of 10 seeds) |
|---|---|---|
| 2048 | `tac_consistency` | ≈ 0.938 |
| 2048 | `source_only` | ≈ 0.859 |
| 2048 | `uniform_dr` | ≈ 0.769 |
| 2048 | `task_weighted` | ≈ 0.580 |
| 256 | `task_weighted` | ≈ 0.907 — **higher than TAC at 0.898** (retained counterexample) |

The last row matters: the mechanism is not dominant at the smallest budget, and the manuscript says so.

## 7. Known non-reproducible pieces

- The exact per-epoch log producer (`record_training.py`) was not retained; see README, "Provenance and known gaps".
- `raw_results*.csv` from the original 2026-11-01 collection are not redistributed here; the 2026-11-02 re-run supersedes them (the original notes record that earlier GPU labels were inconsistent).
- Absolute wall-clock seconds depend on the machine; they are descriptive only.

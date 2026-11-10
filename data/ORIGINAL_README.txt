# TAC-R2S2R verified GPU experiment snapshot

This is a synthetic, binary grasp-place proxy task. It is not a physical robot
or complete Real2Sim2Real/VLA benchmark. 30 inputs (6 task + 24 nuisance),
MLP 30-96-48-1, GELU and LayerNorm. AdamW lr=0.002, weight_decay=0.0001,
batch<=256, fixed 90 epochs. Four methods, four sample budgets, ten seeds.

GPU allocation is verified by exact device name and GPU UUID. RTX 4060:
seeds 11,23,37,53,71. RTX 5060 Ti: seeds 83,97,109,127,149.
Different seeds on each GPU mean timings are descriptive, not a GPU benchmark.

Reproduce with Python 3.12 + PyTorch 2.11 CUDA 12.8:
python record_training.py --gpu-name "NVIDIA GeForce RTX 4060" --seeds 11 23 37 53 71 --output runs/rtx4060
python record_training.py --gpu-name "NVIDIA GeForce RTX 5060 Ti" --seeds 83 97 109 127 149 --output runs/rtx5060ti

Every epoch logs total training objective, supervised BCE, and accuracy/Brier
on a separate 2048-sample target validation set (seed+700000). No early stopping
or hyperparameter selection. Final evaluation uses 8192 target samples
(seed+900000). target_success_rate is a legacy field name: it denotes binary
classification accuracy, NOT robot success. Confident error means incorrect
classification with p>=0.8 or p<=0.2, divided by all test samples.

Intervals are two-sided t 95% intervals over 10 seeds (df=9); not an interval
over the pooled 81920 rows. Test examples repeat across methods and budgets
within each seed. Calibration and confusion pool counts only at one selected
budget and method. Training objective definitions differ across methods.

The original records had inconsistent GPU labels; this rerun supersedes those
labels and adds actual epoch history. Historical paper figures may differ.
The website is a deployed, read-only snapshot, with recorded-epoch playback.
It does not offer remote GPU training or an always-connected GPU monitor.

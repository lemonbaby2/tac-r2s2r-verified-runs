"""TAC-R2S2R v2: Collapse Diagnosis + Budget-Annealed Regularization (BAR) + Calibration.

Extends the authors' original implementation (src/original/run_task_aware_transfer.py)
without changing its data model, so numbers stay comparable to the paper's Table IV.

Three contributions, all measured empirically:
  C1  Collapse diagnosis: instrument TW/TAC training to show that the nuisance
      invariance penalty of 'task_weighted' has a degenerate minimum (constant
      predictor).  At large budgets the repeated updates drive the encoder toward
      that minimum -> prediction entropy rises, representation variance collapses,
      and target accuracy falls to chance.
  C2  BAR: scale every consistency/regularisation weight by (budget_ref/budget)^alpha.
      This keeps the effective regularisation pressure roughly constant across
      budgets and removes the collapse.  alpha is swept.
  C3  Calibration: post-hoc temperature scaling on a held-out source split, which
      reduces confident-error rate (HCE) and Brier without touching accuracy.

Outputs (written to ROOT):
  v2_main.csv        budget x method x seed  (acc, hce, brier, ece, + calibrated)
  v2_bar_sweep.csv   alpha sweep on tw_bar / tac_bar
  v2_diagnosis.csv   per-epoch instrumentation for task_weighted and tac_consistency
  v2_summary.csv     aggregated mean / ci95 by budget x method
  v2_meta.json       environment + provenance
"""
import argparse
import copy
import csv
import json
import math
import platform
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SEEDS = [11, 23, 37, 53, 71, 83, 97, 109, 127, 149]
BUDGETS = [256, 512, 1024, 2048, 4096]
BUDGETS_ORIG = [256, 512, 1024, 2048]

# base method -> (base_name, bar_alpha, calibrate)
METHODS = {
    "source_only":     ("source_only",     0.0,  False),
    "uniform_dr":      ("uniform_dr",      0.0,  False),
    "task_weighted":   ("task_weighted",   0.0,  False),
    "tac_consistency": ("tac_consistency", 0.0,  False),
    "tw_bar":          ("task_weighted",   0.67, False),
    "tac_bar":         ("tac_consistency", 0.67, False),
    "tw_ema":          ("tw_ema",          0.0,  False),   # structural fix: EMA teacher
    "tw_vr":           ("tw_vr",           0.0,  False),   # structural fix: variance floor
    "tacpp":           ("tac_consistency", 0.67, True),
}
METHOD_ORDER = list(METHODS.keys())
TW_LIKE = {"task_weighted", "tw_ema", "tw_vr"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_domain(n: int, seed: int, domain: str):
    """Identical to the authors' generator (do not change)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rel = torch.randn(n, 6, generator=gen)
    nuisance = torch.randn(n, 24, generator=gen)
    if domain == "source":
        rel_shift = torch.tensor([0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
        rel_scale = torch.tensor([1.00, 1.00, 1.00, 1.00, 1.00, 1.00])
        nui_shift, nui_scale = 0.0, 1.0
    elif domain == "target":
        rel_shift = torch.tensor([0.35, -0.25, 0.45, 0.20, 0.30, -0.35])
        rel_scale = torch.tensor([1.10, 0.90, 1.20, 1.05, 1.15, 0.95])
        nui_shift, nui_scale = 1.5, 1.6
    else:
        raise ValueError(domain)
    rel = rel * rel_scale + rel_shift
    nuisance = nuisance * nui_scale + nui_shift
    margin = (
        1.20 * rel[:, 0] - 1.05 * rel[:, 1] + 0.95 * rel[:, 2]
        - 0.65 * rel[:, 3] - 0.85 * rel[:, 4] + 0.75 * rel[:, 5]
        + 0.45 * rel[:, 0] * rel[:, 2] - 0.35 * rel[:, 1] * rel[:, 4]
    )
    margin += 0.30 * torch.randn(n, generator=gen)
    y = (margin > 0).float()
    return torch.cat([rel, nuisance], dim=1), y


def paired_views(x: torch.Tensor, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    view = x.clone()
    view[:, :6] += 0.08 * torch.randn(view[:, :6].shape, generator=gen)
    view[:, 6:] = 1.5 + 1.6 * torch.randn(view[:, 6:].shape, generator=gen)
    return view


class PolicyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(30, 96), nn.GELU(), nn.LayerNorm(96),
            nn.Linear(96, 48), nn.GELU(),
        )
        self.head = nn.Linear(48, 1)

    def forward(self, x, return_rep=False):
        rep = self.encoder(x)
        logits = self.head(rep).squeeze(-1)
        return (logits, rep) if return_rep else logits


def augment(x: torch.Tensor, base: str):
    z = x.clone()
    if base == "uniform_dr":
        z += 0.35 * torch.randn_like(z)
    elif base in TW_LIKE or base == "tac_consistency":
        z[:, :6] += 0.18 * torch.randn_like(z[:, :6])
        z[:, 6:] += 0.70 * torch.randn_like(z[:, 6:])
    return z


def bar_scale(budget: int, alpha: float, ref: int = 256):
    """Budget-annealed consistency scale.  alpha=0 -> 1 (original behaviour)."""
    if alpha == 0.0:
        return 1.0
    return (float(ref) / float(budget)) ** alpha


def train_one(seed, budget, base, alpha=0.0, epochs=90, diag=None, diag_every=5):
    set_seed(seed)
    x_cpu, y_cpu = make_domain(budget, seed * 100 + budget, "source")
    pair_cpu = paired_views(x_cpu, seed * 1000 + budget)
    x, y, pair = x_cpu.to(DEVICE), y_cpu.to(DEVICE), pair_cpu.to(DEVICE)
    model = PolicyMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    batch = min(256, budget)
    scale = bar_scale(budget, alpha)
    teacher = None
    if base == "tw_ema":
        teacher = copy.deepcopy(model).to(DEVICE)
        for p_ in teacher.parameters():
            p_.requires_grad_(False)
        teacher.eval()

    # fixed probe sets for diagnosis (target-like distribution, held-out)
    if diag is not None:
        xd, yd = make_domain(4096, seed + 700_000, "target")
        xd, yd = xd.to(DEVICE), yd.to(DEVICE)

    for epoch in range(epochs):
        if base == "tw_ema" and epoch == 10:
            # Freeze a teacher after warm-up: its target is non-constant, so the
            # degenerate "constant predictor" minimum no longer exists.
            teacher = copy.deepcopy(model).to(DEVICE)
            for p_ in teacher.parameters():
                p_.requires_grad_(False)
            teacher.eval()
        order = torch.randperm(budget, device=DEVICE)
        bce_acc, pen_acc, n_batches = 0.0, 0.0, 0
        for start in range(0, budget, batch):
            idx = order[start:start + batch]
            xb = augment(x[idx], base)
            logits, rep = model(xb, return_rep=True)
            loss_bce = bce(logits, y[idx])
            loss = loss_bce
            pen_val = 0.0
            if base == "task_weighted" or base == "tw_vr":
                nuisance_view = xb.clone()
                nuisance_view[:, 6:] += 0.9 * torch.randn_like(nuisance_view[:, 6:])
                pen = (model(nuisance_view) - logits.detach()).pow(2).mean()
                w = 0.12 * scale
                loss = loss + w * pen
                pen_val = (w * pen).item()
                if base == "tw_vr":
                    # Anti-collapse variance floor: forbid a near-constant predictor.
                    std_p = torch.sigmoid(logits).std()
                    vr = torch.relu(0.10 - std_p)
                    loss = loss + vr
                    pen_val += vr.item()
            elif base == "tw_ema":
                if teacher is not None and epoch >= 10:
                    with torch.no_grad():
                        t_logits = teacher(xb)
                    nuisance_view = xb.clone()
                    nuisance_view[:, 6:] += 0.9 * torch.randn_like(nuisance_view[:, 6:])
                    pen = (model(nuisance_view) - t_logits).pow(2).mean()
                    loss = loss + 0.12 * pen
                    pen_val = (0.12 * pen).item()
            elif base == "tac_consistency":
                pair_logits, pair_rep = model(pair[idx], return_rep=True)
                pen = (rep - pair_rep).pow(2).mean()
                w1 = 0.18 * scale
                w2 = 0.22 * scale
                loss = loss + w1 * pen + w2 * (torch.sigmoid(logits) - torch.sigmoid(pair_logits)).pow(2).mean()
                pen_val = (w1 * pen).item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            bce_acc += loss_bce.item(); pen_acc += pen_val; n_batches += 1

        if diag is not None and (epoch % diag_every == 0 or epoch == epochs - 1):
            with torch.inference_mode():
                p = torch.sigmoid(model(xd))
                pred = (p >= 0.5).float()
                acc = (pred == yd).float().mean().item()
                p_clip = p.clamp(1e-6, 1 - 1e-6)
                ent = (-(p_clip * p_clip.log() + (1 - p_clip) * (1 - p_clip).log())).mean().item()
                _, rr = model(xd, return_rep=True)
                rep_var = rr.var(dim=0).mean().item()
                pstd = p.std().item()
            diag.append({
                "budget": budget, "base": base, "alpha": alpha, "seed": seed,
                "epoch": epoch, "bce": bce_acc / max(n_batches, 1),
                "pen": pen_acc / max(n_batches, 1),
                "pen_over_bce": (pen_acc / max(n_batches, 1)) / max(bce_acc / max(n_batches, 1), 1e-9),
                "target_acc": acc, "pred_entropy": ent, "rep_var": rep_var, "pred_std": pstd,
            })
    return model


@torch.inference_mode()
def evaluate_raw(model, seed):
    x, y = make_domain(8192, seed + 900_000, "target")
    x, y = x.to(DEVICE), y.to(DEVICE)
    logits = model(x)
    prob = torch.sigmoid(logits)
    pred = (prob >= 0.5).float()
    acc = (pred == y).float().mean().item()
    hce = ((pred != y) & ((prob >= 0.8) | (prob <= 0.2))).float().mean().item()
    brier = ((prob - y) ** 2).mean().item()
    ece = expected_calibration_error(prob, y)
    return dict(acc=acc, hce=hce, brier=brier, ece=ece), logits, y


def expected_calibration_error(prob, y, n_bins=15):
    conf = torch.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).float()
    correct = (pred == y).float()
    edges = torch.linspace(0, 1, n_bins + 1, device=prob.device)
    ece = 0.0
    n = prob.numel()
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (conf > lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        conf_b = conf[m].mean().item()
        acc_b = correct[m].mean().item()
        ece += (m.sum().item() / n) * abs(acc_b - conf_b)
    return ece


@torch.inference_mode()
def fit_temperature(model, seed, grid=None):
    """Fit a single temperature on a held-out SOURCE split (not target)."""
    if grid is None:
        grid = [round(0.5 + 0.05 * i, 2) for i in range(51)]  # 0.5..3.0
    xc, yc = make_domain(4096, seed + 500_000, "source")
    xc, yc = xc.to(DEVICE), yc.to(DEVICE)
    logits = model(xc)
    best_t, best_nll = 1.0, float("inf")
    for t in grid:
        nll = nn.functional.binary_cross_entropy_with_logits(logits / t, yc).item()
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


@torch.inference_mode()
def evaluate_calibrated(logits, y, T):
    prob = torch.sigmoid(logits / T)
    pred = (prob >= 0.5).float()
    acc = (pred == y).float().mean().item()
    hce = ((pred != y) & ((prob >= 0.8) | (prob <= 0.2))).float().mean().item()
    brier = ((prob - y) ** 2).mean().item()
    ece = expected_calibration_error(prob, y)
    return dict(acc_cal=acc, hce_cal=hce, brier_cal=brier, ece_cal=ece)


def ci95(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    half = 2.776 * statistics.stdev(values) / math.sqrt(len(values))
    return mean, half


def run_main(seeds, out_dir):
    rows = []
    for budget in BUDGETS:
        for name in METHOD_ORDER:
            base, alpha, calib = METHODS[name]
            for seed in seeds:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                model = train_one(seed, budget, base, alpha=alpha)
                raw, logits, y = evaluate_raw(model, seed)
                row = {"budget": budget, "method": name, "alpha": alpha, "seed": seed, **raw}
                if calib:
                    T = fit_temperature(model, seed)
                    row.update(evaluate_calibrated(logits, y, T))
                    row["temperature"] = T
                else:
                    row.update({"acc_cal": None, "hce_cal": None, "brier_cal": None,
                                "ece_cal": None, "temperature": None})
                torch.cuda.synchronize()
                row["seconds"] = time.perf_counter() - t0
                row["gpu"] = torch.cuda.get_device_name(0)
                rows.append(row)
                print(json.dumps(row), flush=True)
    write_csv(out_dir / "v2_main.csv", rows)
    return rows


def run_sweep(seeds, out_dir):
    rows = []
    alphas = [0.33, 0.67, 1.0]
    for base in ["task_weighted", "tac_consistency"]:
        for alpha in alphas:
            for budget in [1024, 2048, 4096]:
                for seed in seeds:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    model = train_one(seed, budget, base, alpha=alpha)
                    raw, _, _ = evaluate_raw(model, seed)
                    rows.append({"base": base, "alpha": alpha, "budget": budget,
                                 "seed": seed, **raw,
                                 "seconds": time.perf_counter() - t0})
                    print(json.dumps(rows[-1]), flush=True)
    write_csv(out_dir / "v2_bar_sweep.csv", rows)
    return rows


def run_diagnosis(seeds, out_dir):
    rows = []
    for base in ["task_weighted", "tac_consistency"]:
        for budget in BUDGETS_ORIG + [4096]:
            for seed in seeds:
                diag = []
                train_one(seed, budget, base, alpha=0.0, diag=diag, diag_every=5)
                rows.extend(diag)
                print(json.dumps(diag[-1]), flush=True)
    write_csv(out_dir / "v2_diagnosis.csv", rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)


def aggregate_main(rows, out_dir):
    summ = []
    for budget in BUDGETS:
        for name in METHOD_ORDER:
            g = [r for r in rows if r["budget"] == budget and r["method"] == name]
            if not g:
                continue
            item = {"budget": budget, "method": name, "n": len(g)}
            for m in ["acc", "hce", "brier", "ece"]:
                mean, half = ci95([r[m] for r in g])
                item[f"{m}_mean"] = round(mean, 6); item[f"{m}_ci95"] = round(half, 6)
            item["acc_mean_pct"] = round(100 * item["acc_mean"], 3)
            item["hce_mean_pct"] = round(100 * item["hce_mean"], 3)
            summ.append(item)
    write_csv(out_dir / "v2_summary.csv", summ)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["main", "sweep", "diag", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU required")

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "python": platform.python_version(), "platform": platform.platform(),
        "seeds": args.seeds, "budgets": BUDGETS, "methods": METHOD_ORDER,
        "bar_default_alpha": 0.67,
        "claim_boundary": "Controlled synthetic domain-transfer mechanism test; not a real-robot experiment.",
    }
    (out / "v2_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    t0 = time.perf_counter()
    if args.stage in ("main", "all"):
        rows = run_main(args.seeds, out)
        aggregate_main(rows, out)
    if args.stage in ("sweep", "all"):
        run_sweep(args.seeds, out)
    if args.stage in ("diag", "all"):
        run_diagnosis(args.seeds, out)
    meta["total_wall_seconds"] = time.perf_counter() - t0
    (out / "v2_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()

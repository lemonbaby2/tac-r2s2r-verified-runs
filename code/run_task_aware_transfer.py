import csv
import argparse
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
SEEDS = [11, 23, 37, 53, 71]
BUDGETS = [256, 512, 1024, 2048]
METHODS = ["source_only", "uniform_dr", "task_weighted", "tac_consistency"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_domain(n: int, seed: int, domain: str):
    """Generate a controlled grasp-place state classification task.

    First six dimensions are task variables: position error, aperture,
    friction, mass, delay, and clearance. The remaining 24 are nuisance
    appearance variables. Success labels are determined only by task variables.
    """
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
    # A nonlinear, contact-inspired success margin. Noise represents sensing error.
    margin = (
        1.20 * rel[:, 0]
        - 1.05 * rel[:, 1]
        + 0.95 * rel[:, 2]
        - 0.65 * rel[:, 3]
        - 0.85 * rel[:, 4]
        + 0.75 * rel[:, 5]
        + 0.45 * rel[:, 0] * rel[:, 2]
        - 0.35 * rel[:, 1] * rel[:, 4]
    )
    margin += 0.30 * torch.randn(n, generator=gen)
    y = (margin > 0).float()
    return torch.cat([rel, nuisance], dim=1), y


def paired_views(x: torch.Tensor, seed: int):
    """Create paired sim/real-like views with shared task state and changed nuisance."""
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


def augment(x: torch.Tensor, method: str):
    z = x.clone()
    if method == "uniform_dr":
        z += 0.35 * torch.randn_like(z)
    elif method in {"task_weighted", "tac_consistency"}:
        z[:, :6] += 0.18 * torch.randn_like(z[:, :6])
        z[:, 6:] += 0.70 * torch.randn_like(z[:, 6:])
    return z


def train_one(seed: int, budget: int, method: str, epochs: int = 90):
    set_seed(seed)
    x_cpu, y_cpu = make_domain(budget, seed * 100 + budget, "source")
    pair_cpu = paired_views(x_cpu, seed * 1000 + budget)
    x, y, pair = x_cpu.to(DEVICE), y_cpu.to(DEVICE), pair_cpu.to(DEVICE)
    model = PolicyMLP().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    batch = min(256, budget)
    for epoch in range(epochs):
        order = torch.randperm(budget, device=DEVICE)
        for start in range(0, budget, batch):
            idx = order[start:start + batch]
            xb = augment(x[idx], method)
            logits, rep = model(xb, return_rep=True)
            loss = bce(logits, y[idx])
            if method == "task_weighted":
                # Penalize sensitivity to nuisance-only perturbations.
                nuisance_view = xb.clone()
                nuisance_view[:, 6:] += 0.9 * torch.randn_like(nuisance_view[:, 6:])
                loss = loss + 0.12 * (model(nuisance_view) - logits.detach()).pow(2).mean()
            elif method == "tac_consistency":
                pair_logits, pair_rep = model(pair[idx], return_rep=True)
                loss = loss + 0.18 * (rep - pair_rep).pow(2).mean()
                loss = loss + 0.22 * (torch.sigmoid(logits) - torch.sigmoid(pair_logits)).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


@torch.inference_mode()
def evaluate(model: nn.Module, seed: int):
    x, y = make_domain(8192, seed + 900_000, "target")
    x, y = x.to(DEVICE), y.to(DEVICE)
    prob = torch.sigmoid(model(x))
    pred = (prob >= 0.5).float()
    accuracy = (pred == y).float().mean().item()
    confident_wrong = ((pred != y) & ((prob >= 0.8) | (prob <= 0.2))).float().mean().item()
    brier = ((prob - y) ** 2).mean().item()
    return {
        "target_success_rate": accuracy,
        "unsafe_confident_error_rate": confident_wrong,
        "brier_score": brier,
    }


def ci95(values):
    mean = statistics.mean(values)
    if len(values) < 2:
        return mean, 0.0
    half = 2.776 * statistics.stdev(values) / math.sqrt(len(values))  # t(4), 95%
    return mean, half


def main():
    global DEVICE, SEEDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    DEVICE = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    SEEDS = args.seeds
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA GPU is required for this experiment")
    start_all = time.perf_counter()
    rows = []
    for budget in BUDGETS:
        for method in METHODS:
            for seed in SEEDS:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                model = train_one(seed, budget, method)
                metrics = evaluate(model, seed)
                torch.cuda.synchronize()
                rows.append({
                    "seed": seed,
                    "budget": budget,
                    "method": method,
                    **metrics,
                    "train_eval_seconds": time.perf_counter() - t0,
                    "gpu": torch.cuda.get_device_name(0),
                })
                print(rows[-1], flush=True)

    raw_path = ROOT / f"raw_results{args.suffix}.csv"
    with raw_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    summary = []
    for budget in BUDGETS:
        for method in METHODS:
            group = [r for r in rows if r["budget"] == budget and r["method"] == method]
            item = {"budget": budget, "method": method, "n_seeds": len(group)}
            for metric in ["target_success_rate", "unsafe_confident_error_rate", "brier_score", "train_eval_seconds"]:
                mean, half = ci95([r[metric] for r in group])
                item[f"{metric}_mean"] = mean
                item[f"{metric}_ci95_half"] = half
            summary.append(item)

    with (ROOT / f"summary{args.suffix}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)

    metadata = {
        "timestamp_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(DEVICE),
        "gpu": torch.cuda.get_device_name(DEVICE),
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(DEVICE).total_memory / 1024**3, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "seeds": SEEDS,
        "budgets": BUDGETS,
        "methods": METHODS,
        "target_test_samples_per_run": 8192,
        "total_runs": len(rows),
        "total_wall_seconds": time.perf_counter() - start_all,
        "claim_boundary": "Controlled synthetic domain-transfer mechanism test; not a real-robot experiment.",
    }
    (ROOT / f"run_metadata{args.suffix}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

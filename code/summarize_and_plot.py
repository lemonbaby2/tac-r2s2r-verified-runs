import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCES = [ROOT / "raw_results.csv", ROOT / "raw_results_gpu5060ti.csv"]
METRICS = ["target_success_rate", "unsafe_confident_error_rate", "brier_score"]
LABELS = {
    "source_only": "Source only",
    "uniform_dr": "Uniform DR",
    "task_weighted": "Task weighted",
    "tac_consistency": "Task + consistency",
}
COLORS = {
    "source_only": "#495867",
    "uniform_dr": "#a45a52",
    "task_weighted": "#6d9c50",
    "tac_consistency": "#2e6cb1",
}
T_CRIT_9 = 2.262157


def mean_ci(values):
    return statistics.mean(values), T_CRIT_9 * statistics.stdev(values) / math.sqrt(len(values))


def load():
    rows = []
    for path in SOURCES:
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for key in ["seed", "budget"]:
                    row[key] = int(row[key])
                for key in METRICS + ["train_eval_seconds"]:
                    row[key] = float(row[key])
                rows.append(row)
    if len(rows) != 160:
        raise RuntimeError(f"Expected 160 runs, got {len(rows)}")
    keys = [(r["budget"], r["method"], r["seed"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate run keys")
    return rows


def main():
    rows = load()
    summary = []
    for budget in [256, 512, 1024, 2048]:
        for method in LABELS:
            group = [r for r in rows if r["budget"] == budget and r["method"] == method]
            if len(group) != 10:
                raise RuntimeError((budget, method, len(group)))
            out = {"budget": budget, "method": method, "n": 10}
            for metric in METRICS:
                out[metric + "_mean"], out[metric + "_ci95_half"] = mean_ci([r[metric] for r in group])
            summary.append(out)
    with (ROOT / "combined_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

    # Paired contrasts at the largest source-domain budget.
    paired = []
    for baseline in ["source_only", "uniform_dr", "task_weighted"]:
        method_rows = {r["seed"]: r for r in rows if r["budget"] == 2048 and r["method"] == "tac_consistency"}
        base_rows = {r["seed"]: r for r in rows if r["budget"] == 2048 and r["method"] == baseline}
        for metric in METRICS:
            diffs = [method_rows[s][metric] - base_rows[s][metric] for s in sorted(method_rows)]
            mean, half = mean_ci(diffs)
            paired.append({"budget": 2048, "contrast": "tac_consistency minus " + baseline, "metric": metric, "mean": mean, "ci95_half": half})
    with (ROOT / "paired_contrasts.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    budgets = np.array([256, 512, 1024, 2048])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.65), constrained_layout=True)
    for method in LABELS:
        rr = [r for r in summary if r["method"] == method]
        for ax, metric, scale in [(axes[0], "target_success_rate", 100), (axes[1], "unsafe_confident_error_rate", 100)]:
            means = np.array([r[metric + "_mean"] for r in rr]) * scale
            err = np.array([r[metric + "_ci95_half"] for r in rr]) * scale
            ax.errorbar(budgets, means, yerr=err, marker="o", capsize=2.5, linewidth=1.6, color=COLORS[method], label=LABELS[method])
    axes[0].set(ylabel="Target-domain classification accuracy (%)", xlabel="Source training samples", ylim=(35, 100))
    axes[1].set(ylabel="Confident error rate (%)", xlabel="Source training samples", ylim=(0, 65))
    for ax in axes:
        ax.set_xticks(budgets, [str(v) for v in budgets]); ax.grid(alpha=.2)
    axes[0].legend(frameon=False, loc="lower right", fontsize=7)
    fig.savefig(ROOT / "figure7_real_gpu_accuracy_and_risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # An independent diagnostic figure for the consistency ablation.
    fig, ax = plt.subplots(figsize=(5.4, 3.4), constrained_layout=True)
    for method in ["task_weighted", "tac_consistency"]:
        rr = [r for r in summary if r["method"] == method]
        means = np.array([r["brier_score_mean"] for r in rr])
        err = np.array([r["brier_score_ci95_half"] for r in rr])
        ax.errorbar(budgets, means, yerr=err, marker="o", capsize=3, linewidth=2, color=COLORS[method], label=LABELS[method])
    ax.set(xlabel="Source training samples", ylabel="Brier score (lower is better)", ylim=(0, .7))
    ax.set_xticks(budgets, [str(v) for v in budgets]); ax.grid(alpha=.2); ax.legend(frameon=False)
    fig.savefig(ROOT / "figure8_real_gpu_consistency_ablation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Conceptual method diagram; all performance plots above use raw run logs.
    fig, ax = plt.subplots(figsize=(10.0, 2.15), constrained_layout=True)
    ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    blocks = [
        (0.1, "Source state\n+ task variables"),
        (2.1, "Paired target view\n+ appearance shift"),
        (4.1, "Shared encoder\n+ 96 → 48"),
        (6.1, "Representation +\n+ prediction consistency"),
        (8.1, "Target evaluation\n+ accuracy / risk"),
    ]
    fills = ["#e7f0fa", "#eaf4e5", "#e8ebf4", "#f6eee3", "#e9f2f2"]
    for i, (x, label) in enumerate(blocks):
        ax.add_patch(FancyBboxPatch((x, .5), 1.8, 1.0, boxstyle="round,pad=0.05,rounding_size=0.09", facecolor=fills[i], edgecolor="#465563", linewidth=1))
        ax.text(x + .9, 1.0, label, ha="center", va="center", fontsize=9)
        if i < len(blocks) - 1:
            ax.add_patch(FancyArrowPatch((x + 1.85, 1), (x + 2.05, 1), arrowstyle="-|>", mutation_scale=12, color="#465563"))
    fig.savefig(ROOT / "figure3_consistency_mechanism.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"runs": len(rows), "gpus": sorted(set(r["gpu"] for r in rows)), "summary": summary, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify every headline number of the manuscript directly against the released data.

Run:  python code/verify_numbers.py

Exit code 0 = all checks pass; 1 = at least one mismatch.
No GPU, no network and no third-party package is required (standard library only).
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

FAILED = []


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(("  PASS  " if ok else "  FAIL  ") + f"{label:<46} got={got!r:<22} paper={want!r}")
    if not ok:
        FAILED.append(label)


def rows(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


print("=" * 96)
print("1. Table IV source values (summary.csv, mean over 10 seeds, budget 2048)")
print("=" * 96)
summary = rows(os.path.join(DATA, "summary.csv"))
idx = {(int(r["budget"]), r["method"]): r for r in summary}

# Paper: TAC 93.80% +/- 0.32 pp, confident error 1.61%; source-only 85.90%;
# uniform randomization 76.91%; seed-paired TAC - source-only = 7.91 pp.
check("TAC target accuracy @2048", float(idx[(2048, "tac_consistency")]["target_success_rate"]), 0.9380, 5e-4)
check("TAC accuracy CI95 half-width @2048", float(idx[(2048, "tac_consistency")]["target_success_rate_ci95"]), 0.0032, 5e-4)
check("TAC confident error @2048", float(idx[(2048, "tac_consistency")]["unsafe_confident_error_rate"]), 0.0161, 5e-4)
check("Source-only accuracy @2048", float(idx[(2048, "source_only")]["target_success_rate"]), 0.8590, 5e-4)
check("Uniform-DR accuracy @2048", float(idx[(2048, "uniform_dr")]["target_success_rate"]), 0.7691, 5e-4)

paired = rows(os.path.join(DATA, "paired_contrasts.csv"))
p = {(r["budget"], r["contrast"], r["metric"]): float(r["mean"]) for r in paired}
check("TAC - source-only accuracy gap @2048",
      p[("2048", "tac_consistency minus source_only", "target_success_rate")], 0.0791, 5e-4)

print()
print("=" * 96)
print("2. Design of the released runs (run_metrics.csv)")
print("=" * 96)
runs = rows(os.path.join(DATA, "run_metrics.csv"))
methods = sorted({r["method"] for r in runs})
budgets = sorted({int(r["budget"]) for r in runs})
seeds = sorted({int(r["seed"]) for r in runs})
gpus = sorted({r["gpu"] for r in runs})
print(f"  runs={len(runs)}  methods={methods}")
print(f"  budgets={budgets}  seeds={seeds}")
print(f"  gpus={gpus}")
print(f"  unique GPU UUIDs={len({r['gpu_uuid'] for r in runs})}")
for label, got, want in [
    ("total runs = 4 methods x 4 budgets x 10 seeds", len(runs), 160),
    ("distinct methods", len(methods), 4),
    ("distinct budgets", len(budgets), 4),
    ("distinct seeds", len(seeds), 10),
    ("distinct physical GPUs", len(gpus), 2),
]:
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{label:<46} got={got} want={want}")
    if not ok:
        FAILED.append(label)

per_gpu = {g: sum(1 for r in runs if r["gpu"] == g) for g in gpus}
print(f"  runs per GPU: {per_gpu}")

print()
print("=" * 96)
print("3. Per-epoch training logs (training_history.csv)")
print("=" * 96)
hist = rows(os.path.join(DATA, "training_history.csv"))
epochs = sorted({int(r["epoch"]) for r in hist})
for label, got, want in [
    ("rows = 160 runs x 90 epochs", len(hist), 160 * 90),
    ("epochs per run", len(epochs), 90),
]:
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{label:<46} got={got} want={want}")
    if not ok:
        FAILED.append(label)

print()
print("=" * 96)
print("4. Structured snapshot (verified_runs.json)")
print("=" * 96)
with open(os.path.join(DATA, "verified_runs.json"), encoding="utf-8") as f:
    snap = json.load(f)
for label, got, want in [
    ("runs in JSON", len(snap["runs"]), 160),
    ("summary rows in JSON", len(snap["summary"]), 16),
    ("GPU metadata entries", len(snap["metadata"]), 2),
]:
    ok = got == want
    print(("  PASS  " if ok else "  FAIL  ") + f"{label:<46} got={got} want={want}")
    if not ok:
        FAILED.append(label)
print(f"  scope: {snap.get('scope')}")
print(f"  version: {snap.get('version')}")

print()
if FAILED:
    print(f"RESULT: {len(FAILED)} CHECK(S) FAILED -> {FAILED}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED - every headline number in the manuscript is reproducible "
      "from the released files.")

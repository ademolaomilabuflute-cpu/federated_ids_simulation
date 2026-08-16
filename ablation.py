"""
Ablation runner: sweeps client count and Dirichlet alpha, matching the
methodology behind the paper's Table 6 (client scaling) and Section 5.5
(extreme heterogeneity, alpha=0.1). No earlier version of this repository
ran or produced these numbers — this script actually generates them.

Usage:
    python ablation.py --data-path /path/to/EdgeIIoT_preprocessed.csv

Each configuration is run as a fresh subprocess call to simulate.py, and
results are collected into results/ablation_summary.csv.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys


CLIENT_COUNT_SWEEP = [5, 10, 20]
ALPHA_SWEEP = [0.5, 0.1]


def run_one(data_path: str, num_clients: int, alpha: float, num_rounds: int, out_dir: str) -> dict:
    run_dir = os.path.join(out_dir, f"clients{num_clients}_alpha{alpha}")
    os.makedirs(run_dir, exist_ok=True)
    cmd = [
        sys.executable, "simulate.py",
        "--data-path", data_path,
        "--num-clients", str(num_clients),
        "--alpha", str(alpha),
        "--num-rounds", str(num_rounds),
        "--out-dir", run_dir,
    ]
    print(f"\n### Running: clients={num_clients}, alpha={alpha} ###")
    subprocess.run(cmd, check=True)

    report_path = os.path.join(run_dir, "per_class_report.csv")
    weighted_f1 = None
    if os.path.exists(report_path):
        with open(report_path) as f:
            rows = list(csv.DictReader(f))
        supports = [int(r["support"]) for r in rows]
        f1s = [float(r["f1"]) for r in rows]
        total = sum(supports)
        weighted_f1 = sum(f1 * s for f1, s in zip(f1s, supports)) / total if total else None

    return {"num_clients": num_clients, "alpha": alpha, "weighted_f1": weighted_f1, "run_dir": run_dir}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--num-rounds", type=int, default=50)
    parser.add_argument("--out-dir", default="results/ablation")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []

    for n_clients in CLIENT_COUNT_SWEEP:
        summary.append(run_one(args.data_path, n_clients, 0.5, args.num_rounds, args.out_dir))

    for alpha in ALPHA_SWEEP:
        if alpha == 0.5:
            continue  # already covered by the client-count sweep at the baseline alpha
        summary.append(run_one(args.data_path, 5, alpha, args.num_rounds, args.out_dir))

    summary_path = os.path.join(args.out_dir, "ablation_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["num_clients", "alpha", "weighted_f1", "run_dir"])
        writer.writeheader()
        writer.writerows(summary)
    print(f"\n[DONE] Ablation summary saved to {summary_path}")


if __name__ == "__main__":
    main()

"""
run_distribution_shift.py
=========================
Compute MMD / Wasserstein / Mahalanobis distances between each patient
and the pooled training set, and correlate with per-fold AUC.

Usage:
    python scripts/run_distribution_shift.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src import distribution_shift, visualize


def main():
    per_fold_csv = config.TABLES_PATH / "loocv_per_fold.csv"
    df = None
    if per_fold_csv.exists():
        df = pd.read_csv(per_fold_csv)
        print(f"Loaded LOOCV results from {per_fold_csv}")
    else:
        print(f"WARNING: {per_fold_csv} not found.")
        print("Distribution shifts will be computed without AUC correlation.")

    df_shift = distribution_shift.compute_all_shifts(per_fold_results=df)
    out = config.TABLES_PATH / "distribution_shift.csv"
    df_shift.to_csv(out, index=False)
    print(f"Saved {out}")

    if df is not None:
        visualize.plot_distribution_shift(df_shift)


if __name__ == "__main__":
    main()

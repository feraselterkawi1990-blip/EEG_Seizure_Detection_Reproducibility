"""
run_statistical_tests.py
========================
Re-run Wilcoxon + bootstrap CI on existing LOOCV results without
re-running the full LOOCV. Useful after adding new models incrementally.

Usage:
    python scripts/run_statistical_tests.py
    python scripts/run_statistical_tests.py --metric sensitivity
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src import statistical_tests


def main():
    parser = argparse.ArgumentParser(
        description="Run statistical tests on LOOCV results."
    )
    parser.add_argument("--input", type=str,
                        default=str(config.TABLES_PATH / "loocv_per_fold.csv"),
                        help="Path to per-fold CSV")
    parser.add_argument("--metric", type=str, default="auc",
                        choices=["auc", "sensitivity", "specificity", "accuracy"],
                        help="Primary metric for Wilcoxon (bootstrap on all)")
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: input file not found: {args.input}")
        print("Run LOOCV first: python scripts/run_loocv.py")
        sys.exit(1)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows, {df['model'].nunique()} models, "
          f"{df['test_patient'].nunique()} patients")

    statistical_tests.run_all_tests(df, save_dir=config.TABLES_PATH)


if __name__ == "__main__":
    main()

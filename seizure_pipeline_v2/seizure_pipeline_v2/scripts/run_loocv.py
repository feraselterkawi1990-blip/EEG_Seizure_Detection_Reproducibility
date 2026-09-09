"""
run_loocv.py
============
Entry point: build per-patient cache, run LOOCV across models, generate
figures and statistical tests.

CHANGELOG (v2):
  - Now invokes statistical tests after LOOCV completes.
  - Now runs distribution-shift analysis when --shift flag set.
  - Defaults to RF + SVM + XGB.
"""

import argparse
import logging
import sys
from pathlib import Path

# Allow `python scripts/run_loocv.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from src import cache, loocv, visualize, statistical_tests, distribution_shift


def setup_logging():
    config.LOGS_PATH.mkdir(parents=True, exist_ok=True)
    log_file = config.LOGS_PATH / "run_loocv.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run LOOCV pipeline for EEG seizure detection."
    )
    parser.add_argument("--smoke", action="store_true",
                        help="Quick test on a single patient (chb01)")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache build (assumes cache is ready)")
    parser.add_argument("--force-cache", action="store_true",
                        help="Rebuild cache even if exists")
    parser.add_argument("--skip-loocv", action="store_true",
                        help="Skip LOOCV (re-runs only stats and figures from existing CSV)")
    parser.add_argument("--shift", action="store_true",
                        help="Also run distribution-shift analysis (slow, optional)")
    parser.add_argument("--no-stats", action="store_true",
                        help="Skip statistical tests (Wilcoxon + bootstrap)")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger("run_loocv")

    # Smoke test override
    if args.smoke:
        config.PATIENTS = ["chb01"]
        logger.info("[SMOKE] reduced to 1 patient")

    # Stage 1: Cache
    if not args.skip_cache and not args.skip_loocv:
        logger.info("Stage 1: building per-patient caches...")
        cache.build_all_caches(force=args.force_cache)

    # Stage 2: LOOCV
    if not args.skip_loocv:
        logger.info("Stage 2: running LOOCV...")
        df = loocv.run_loocv()
    else:
        per_fold_csv = config.TABLES_PATH / "loocv_per_fold.csv"
        if not per_fold_csv.exists():
            logger.error(f"--skip-loocv requires {per_fold_csv} to exist")
            sys.exit(1)
        df = pd.read_csv(per_fold_csv)
        logger.info(f"Loaded {len(df)} rows from {per_fold_csv}")

    # Stage 3: Statistical tests
    if not args.no_stats:
        logger.info("Stage 3: statistical tests (Wilcoxon + bootstrap CI)...")
        statistical_tests.run_all_tests(df, save_dir=config.TABLES_PATH)

    # Stage 4: Distribution shift (optional, slow)
    df_shift = None
    if args.shift:
        logger.info("Stage 4: distribution-shift analysis...")
        df_shift = distribution_shift.compute_all_shifts(per_fold_results=df)
        df_shift.to_csv(config.TABLES_PATH / "distribution_shift.csv", index=False)

    # Stage 5: Figures
    logger.info("Stage 5: generating figures...")
    visualize.generate_all_figures(df, df_shift=df_shift)

    logger.info("\nDone. Results in %s", config.OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()

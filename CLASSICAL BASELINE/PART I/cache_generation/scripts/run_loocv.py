"""
run_loocv.py
============
Main entry point: end-to-end pipeline runner.

Stages
------
  1. Build per-patient feature caches (one-time, slowest stage)
  2. Run LOOCV across all patients
  3. Generate summary tables and figures

Usage
-----
    # Full pipeline (default)
    python scripts/run_loocv.py

    # Re-run only LOOCV using existing caches
    python scripts/run_loocv.py --skip-cache

    # Force rebuild of caches
    python scripts/run_loocv.py --force-cache

    # Quick smoke test (cache one patient, dry run)
    python scripts/run_loocv.py --smoke
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src import cache, loocv, visualize


def setup_logging(level: str = "INFO"):
    log_format = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOGS_PATH / "run_loocv.log", mode="a"),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="EEG Seizure Detection LOOCV pipeline")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache building (use existing caches)")
    parser.add_argument("--force-cache", action="store_true",
                        help="Force rebuild of all patient caches")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test: cache only first patient, no LOOCV")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)
    logger = logging.getLogger("run_loocv")

    logger.info("=" * 70)
    logger.info("EEG Seizure Detection — LOOCV Pipeline")
    logger.info("=" * 70)
    logger.info(f"CHB-MIT path: {config.CHB_MIT_PATH}")
    logger.info(f"Output path:  {config.OUTPUT_PATH}")
    logger.info(f"Patients:     {len(config.PATIENTS)}  ({config.PATIENTS[0]}..{config.PATIENTS[-1]})")
    logger.info(f"Models:       {config.MODELS_TO_RUN}")

    if args.smoke:
        logger.info("\n[SMOKE TEST] Building cache for first patient only...")
        cache.build_patient_cache(config.PATIENTS[0], force=args.force_cache)
        logger.info("Smoke test complete.")
        return

    # ---------------- Stage 1: cache features ----------------
    if not args.skip_cache:
        logger.info("\n--- Stage 1: building per-patient feature caches ---")
        cache.build_all_caches(force=args.force_cache)

    # ---------------- Stage 2: LOOCV ----------------
    logger.info("\n--- Stage 2: running LOOCV ---")
    df = loocv.run_loocv()

    # ---------------- Stage 3: figures + tables ----------------
    logger.info("\n--- Stage 3: generating figures and summary tables ---")
    visualize.generate_all_figures(df)

    logger.info("\n--- DONE ---")
    logger.info(f"Tables: {config.TABLES_PATH}")
    logger.info(f"Figures: {config.FIGURES_PATH}")


if __name__ == "__main__":
    main()

"""
run_loocv_iso.py
================
Entry-point for the Stage 3-isolated experiment.

Pipeline:
  1. Build cache_iso/ for all 21 patients (per-segment z-scored features)
     - Skipped automatically if cache_iso/<patient>.npz already exists.
  2. Run LOOCV with RF only on the ISO-cached features.
  3. Write results/tables/loocv_per_fold_iso.csv
  4. Print comparison vs Part I baseline.

Usage
-----
From project root:

    python scripts/run_loocv_iso.py            # full LOOCV (21 folds)
    python scripts/run_loocv_iso.py --dry-run  # only chb01 + chb02 (~30 min)
    python scripts/run_loocv_iso.py --force    # rebuild cache_iso

Time estimates (Ryzen 9 7940HS, no GPU):
  - Cache build: 30-60 min/patient × 21 = 10-20 hours TOTAL
                 (one-time cost; subsequent runs use cached features)
  - LOOCV: 5-15 min/fold × 21 = 2-5 hours
  - Total first run: 12-25 hours
  - Subsequent reruns: 2-5 hours
"""

import argparse
import logging
import sys
from pathlib import Path

# Make src/ importable when running as a script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src import cache_iso
from src.loocv_iso import run_loocv_iso


def setup_logging():
    """Mirror the logging style used by run_loocv.py."""
    log_path = config.LOGS_PATH / "loocv_iso.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3-isolated LOOCV: RF + per-segment z-score on Part I config"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on chb01 + chb02 only (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild ISO cache even if it exists")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only build the ISO cache; skip LOOCV")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache build; assume caches already exist")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    # --- Determine patient list ---
    if args.dry_run:
        original_patients = config.PATIENTS.copy()
        config.PATIENTS = ["chb01", "chb02"]
        logger.info("=" * 70)
        logger.info("DRY RUN MODE — only chb01 and chb02")
        logger.info("=" * 70)

    logger.info(f"Patients: {len(config.PATIENTS)}  ({config.PATIENTS})")
    logger.info(f"Output:   {config.TABLES_PATH}")
    logger.info(f"Cache:    {cache_iso.CACHE_ISO_PATH}")

    # --- Phase 1: Build ISO cache ---
    if not args.skip_cache:
        logger.info("\n" + "=" * 70)
        logger.info("PHASE 1: Building ISO cache (per-segment z-score)")
        logger.info("=" * 70)
        for pid in config.PATIENTS:
            try:
                cache_iso.build_patient_cache_iso(pid, force=args.force)
            except Exception as e:
                logger.error(f"Failed to ISO-cache {pid}: {e}")

    if args.cache_only:
        logger.info("Cache build complete. Exiting (--cache-only).")
        return

    # --- Phase 2: Run LOOCV ---
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: LOOCV on ISO cache")
    logger.info("=" * 70)
    df = run_loocv_iso()

    logger.info("\n" + "=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)
    logger.info(f"Results: {config.TABLES_PATH / 'loocv_per_fold_iso.csv'}")
    logger.info(f"Log:     {config.LOGS_PATH / 'loocv_iso.log'}")


if __name__ == "__main__":
    main()

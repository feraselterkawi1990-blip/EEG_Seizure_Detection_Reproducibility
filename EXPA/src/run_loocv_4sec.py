"""
run_loocv_4sec.py
=================
Entry-point for EXP A: Part I + 4-second windows + StandardScaler.

This completes the 2x2 factorial design needed to decompose:
  - normalisation effect
  - window length effect
  - their interaction
  - architecture effect

DESIGN CHOICE: Default keeps SVM for methodological consistency with
Part I and iso experiments. Even if SVM gives weak results, removing
it would be cherry-picking. Scientific integrity > time savings.

Pipeline:
  1. Build cache_4sec/ for all 21 patients (4-sec windows, NO z-score)
  2. Run LOOCV with RF + SVM + XGB (matches Part I + iso)
  3. Write results/tables/loocv_per_fold_4sec_stdscaler.csv

Usage:
    python scripts/run_loocv_4sec.py                       # default: RF+SVM+XGB
    python scripts/run_loocv_4sec.py --dry-run             # chb01+chb02 only
    python scripts/run_loocv_4sec.py --models RF XGB       # skip SVM (faster)
    python scripts/run_loocv_4sec.py --cache-only          # just build cache
    python scripts/run_loocv_4sec.py --skip-cache          # use existing cache

Time estimates (Ryzen 9 7940HS):
  - Cache build (4-sec, 21 patients): ~10 hours
  - LOOCV (RF + SVM + XGB):           ~9 hours (SVM dominates)
  - LOOCV (RF + XGB only):            ~15 min
  - Total with SVM:                   ~18-19 hours
  - Total without SVM:                ~10-11 hours
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src import cache_4sec
from src.loocv_4sec import run_loocv_4sec


def setup_logging():
    log_path = config.LOGS_PATH / "loocv_4sec.log"
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
        description="EXP A: Part I + 4-sec + StandardScaler LOOCV"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Run on chb01 + chb02 only (smoke test)")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild 4-sec cache even if it exists")
    parser.add_argument("--cache-only", action="store_true",
                        help="Only build the 4-sec cache; skip LOOCV")
    parser.add_argument("--skip-cache", action="store_true",
                        help="Skip cache build; assume caches already exist")
    parser.add_argument("--models", nargs="+", default=["RF", "SVM", "XGB"],
                        choices=["RF", "SVM", "XGB"],
                        help="Models to run. Default: RF+SVM+XGB (matches Part I).")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    if args.dry_run:
        config.PATIENTS = ["chb01", "chb02"]
        logger.info("=" * 78)
        logger.info("DRY RUN MODE - only chb01 and chb02")
        logger.info("=" * 78)

    logger.info("=" * 78)
    logger.info("EXP A - Part I + 4-second windows + StandardScaler")
    logger.info(f"Patients: {len(config.PATIENTS)}  ({config.PATIENTS})")
    logger.info(f"Models:   {args.models}")
    logger.info(f"Cache:    {cache_4sec.CACHE_4SEC_PATH}")
    logger.info(f"Output:   {config.TABLES_PATH / 'loocv_per_fold_4sec_stdscaler.csv'}")
    logger.info("=" * 78)

    if not args.skip_cache:
        logger.info("\n" + "=" * 78)
        logger.info("PHASE 1: Building 4-sec cache (NO z-score)")
        logger.info("=" * 78)
        for pid in config.PATIENTS:
            try:
                cache_4sec.build_patient_cache_4sec(pid, force=args.force)
            except Exception as e:
                logger.error(f"Failed to 4-sec-cache {pid}: {e}")

    if args.cache_only:
        logger.info("Cache build complete. Exiting (--cache-only).")
        return

    logger.info("\n" + "=" * 78)
    logger.info("PHASE 2: LOOCV on 4-sec cache with StandardScaler")
    logger.info("=" * 78)
    df = run_loocv_4sec(models_to_run=args.models)

    logger.info("\n" + "=" * 78)
    logger.info("DONE")
    logger.info("=" * 78)
    logger.info(f"Results: {config.TABLES_PATH / 'loocv_per_fold_4sec_stdscaler.csv'}")
    logger.info(f"Log:     {config.LOGS_PATH / 'loocv_4sec.log'}")


if __name__ == "__main__":
    main()

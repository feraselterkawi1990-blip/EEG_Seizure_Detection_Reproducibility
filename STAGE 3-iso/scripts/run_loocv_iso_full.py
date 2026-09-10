"""
run_loocv_iso_full.py
=====================
Run Stage 3-isolated LOOCV with RF, SVM, and XGBoost on the existing
cache_iso/ (built by run_loocv_iso.py).

Usage:
    python scripts/run_loocv_iso_full.py
    python scripts/run_loocv_iso_full.py --models RF XGB     # skip SVM (slow)
    python scripts/run_loocv_iso_full.py --models XGB        # XGB only

Time estimates (Ryzen 9 7940HS):
    RF:  ~10 min total (already done — but re-runs with MCC)
    SVM: ~1.5 hours total
    XGB: ~30 min total
    Default RF+SVM+XGB: ~2-2.5 hours
    RF+XGB only:        ~40 min
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import config
from src.loocv_iso_full import run_loocv_iso_full


def setup_logging():
    log_path = config.LOGS_PATH / "loocv_iso_full.log"
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
        description="Stage 3-isolated full LOOCV (RF + SVM + XGB)"
    )
    parser.add_argument("--models", nargs="+", default=["RF", "SVM", "XGB"],
                        choices=["RF", "SVM", "XGB"],
                        help="Models to run. Default: all three.")
    args = parser.parse_args()

    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 78)
    logger.info("STAGE 3-ISOLATED FULL LOOCV")
    logger.info(f"Models: {args.models}")
    logger.info(f"Cache:  results\\cache_iso (must already exist)")
    logger.info(f"Output: results\\tables\\loocv_per_fold_iso_full.csv")
    logger.info("=" * 78)

    df = run_loocv_iso_full(models_to_run=args.models)

    logger.info("\n" + "=" * 78)
    logger.info("DONE")
    logger.info("=" * 78)


if __name__ == "__main__":
    main()

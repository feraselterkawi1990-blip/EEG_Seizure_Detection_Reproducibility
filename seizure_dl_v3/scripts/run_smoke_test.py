"""
Quick smoke test — train EEGNet on 2 patients (chb01 train, chb02 test).

Should complete in ~10-15 minutes on RTX 4070.
Confirms the pipeline works end-to-end before launching the 12-hour LOOCV.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Override config for smoke test
import config
config.MAX_EPOCHS = 5
config.EARLY_STOPPING_PATIENCE = 3
config.PATIENTS = ["chb01", "chb02"]

from run_dl_loocv import run_fold, set_seed
from config import RANDOM_SEED, RESULTS_DIR


def main():
    set_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("SMOKE TEST — EEGNet on chb01→chb02")
    print("=" * 60)

    metrics = run_fold(
        model_name="eegnet",
        test_patient="chb02",
        train_patients=["chb01"],
    )

    print()
    print("=" * 60)
    print("Smoke test PASSED. Pipeline is functional.")
    print("=" * 60)
    print(f"Test AUC: {metrics['auc']:.3f}")
    print(f"Test MCC: {metrics['mcc']:+.3f}")
    print(f"Fold time: {metrics['fold_total_sec']:.1f}s")
    print()
    print("If AUC > 0.5 and no errors, you can launch the full LOOCV:")
    print("  python scripts/run_dl_loocv.py --model eegnet")


if __name__ == "__main__":
    main()

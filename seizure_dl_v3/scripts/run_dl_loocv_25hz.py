"""
Stage 1.5 LOOCV runner — uses 25 Hz cache, otherwise identical to Stage 1.

Outputs to results/per_fold_eegnet_25hz.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PATIENTS, RESULTS_DIR, LOG_DIR,
    N_CHANNELS, WINDOW_SAMPLES, RANDOM_SEED,
    EEGNET_CONFIG,
)
from models import build_model, count_parameters
from training import (
    DEVICE, assemble_train_pool, assemble_test, split_train_val,
    train_one_fold, predict_proba, youden_threshold, compute_metrics,
)

# Use the 25Hz cache instead of the default
CACHE_25HZ = Path(r"D:\seizure_dl_v3\cache_raw_25hz")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fold(test_patient, train_patients):
    t0 = time.time()
    print(f"\n=== EEGNET 25Hz | held-out: {test_patient} ===")
    print("  Loading training data (25Hz cache)...")
    X_pool, y_pool = assemble_train_pool(train_patients, CACHE_25HZ)
    print(f"    Train pool: {len(X_pool)} segments, "
          f"{(y_pool==1).sum()} pos, {(y_pool==0).sum()} neg")

    X_tr, y_tr, X_val, y_val = split_train_val(X_pool, y_pool)
    print(f"    Train: {len(X_tr)} | Val: {len(X_val)}")

    model = build_model("eegnet", N_CHANNELS, WINDOW_SAMPLES, **EEGNET_CONFIG)
    n_params = count_parameters(model)
    print(f"    Model: eegnet ({n_params:,} params)")

    print("  Training...")
    fit_t0 = time.time()
    model, history, _, _ = train_one_fold(model, X_tr, y_tr, X_val, y_val)
    fit_time = time.time() - fit_t0
    print(f"    Fit time: {fit_time:.1f}s ({len(history)} epochs)")

    val_proba = predict_proba(model, X_val)
    threshold = youden_threshold(y_val, val_proba)
    print(f"    Threshold: {threshold:.2f}")

    print("  Evaluating...")
    X_test, y_test = assemble_test(test_patient, CACHE_25HZ)
    test_proba = predict_proba(model, X_test)
    metrics = compute_metrics(y_test, test_proba, threshold)
    metrics.update({
        "test_patient": test_patient,
        "model": "eegnet_25hz",
        "n_test": int(len(y_test)),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "fit_time_sec": fit_time,
        "n_params": n_params,
        "fold_total_sec": time.time() - t0,
    })

    proba_dir = RESULTS_DIR / "probabilities"
    proba_dir.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        proba_dir / f"eegnet_25hz_{test_patient}.npz",
        y_true=y_test, y_proba=test_proba, threshold=threshold,
    )

    print(f"  Done in {metrics['fold_total_sec']:.1f}s | "
          f"AUC={metrics['auc']:.3f} MCC={metrics['mcc']:+.3f} "
          f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f}")
    return metrics


def main():
    set_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Cache: {CACHE_25HZ}")

    out_csv = RESULTS_DIR / "per_fold_eegnet_25hz.csv"
    fieldnames = [
        "test_patient", "model", "accuracy", "sensitivity", "specificity",
        "auc", "mcc", "f1", "tp", "tn", "fp", "fn",
        "threshold", "n_test", "n_pos_test", "n_neg_test",
        "fit_time_sec", "fold_total_sec", "n_params",
    ]

    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for test_patient in PATIENTS:
            train_patients = [p for p in PATIENTS if p != test_patient]
            try:
                metrics = run_fold(test_patient, train_patients)
                row = {k: metrics.get(k) for k in fieldnames}
                writer.writerow(row)
                f.flush()
            except Exception as e:
                print(f"  [ERROR] {test_patient}: {e}")
                import traceback
                traceback.print_exc()

    print(f"\n=== Done. Results saved to {out_csv} ===")


if __name__ == "__main__":
    main()

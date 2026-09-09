"""
Main LOOCV runner — trains and evaluates a deep model across the 21 patients.

Usage:
    python scripts/run_dl_loocv.py --model eegnet
    python scripts/run_dl_loocv.py --model shallowconvnet
    python scripts/run_dl_loocv.py --model cnnlstm
    python scripts/run_dl_loocv.py --model eegnet --patients chb01,chb02

Output:
    results/per_fold_<model>.csv with one row per held-out patient.
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
    PATIENTS, RAW_CACHE_DIR, RESULTS_DIR, LOG_DIR,
    N_CHANNELS, WINDOW_SAMPLES, RANDOM_SEED,
    EEGNET_CONFIG, SHALLOWCONVNET_CONFIG, CNNLSTM_CONFIG,
)
from models import build_model, count_parameters
from training import (
    DEVICE,
    assemble_train_pool, assemble_test, split_train_val,
    train_one_fold, predict_proba, youden_threshold, compute_metrics,
)


MODEL_CONFIGS = {
    "eegnet": EEGNET_CONFIG,
    "shallowconvnet": SHALLOWCONVNET_CONFIG,
    "cnnlstm": CNNLSTM_CONFIG,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fold(model_name, test_patient, train_patients):
    """One LOOCV fold. Returns metrics dict."""
    t0 = time.time()
    print(f"\n=== {model_name.upper()} | held-out: {test_patient} ===")

    # 1. Assemble training pool
    print("  Loading training data...")
    X_pool, y_pool = assemble_train_pool(train_patients, RAW_CACHE_DIR)
    print(f"    Train pool: {len(X_pool)} segments, "
          f"{(y_pool==1).sum()} pos, {(y_pool==0).sum()} neg")

    # 2. Train/val split
    X_tr, y_tr, X_val, y_val = split_train_val(X_pool, y_pool)
    print(f"    Train: {len(X_tr)} | Val: {len(X_val)}")

    # 3. Build model
    cfg = MODEL_CONFIGS[model_name]
    model = build_model(model_name, N_CHANNELS, WINDOW_SAMPLES, **cfg)
    n_params = count_parameters(model)
    print(f"    Model: {model_name} ({n_params:,} params)")

    # 4. Train with early stopping
    print("  Training...")
    fit_t0 = time.time()
    model, history, mean, std = train_one_fold(
        model, X_tr, y_tr, X_val, y_val,
        log_path=LOG_DIR / f"{model_name}_{test_patient}.json",
    )
    fit_time = time.time() - fit_t0
    print(f"    Fit time: {fit_time:.1f}s ({len(history)} epochs)")

    # 5. Pick threshold on validation set
    val_proba = predict_proba(model, X_val, mean, std)
    threshold = youden_threshold(y_val, val_proba)
    print(f"    Threshold: {threshold:.2f}")

    # 6. Evaluate on held-out patient
    print("  Evaluating...")
    X_test, y_test = assemble_test(test_patient, RAW_CACHE_DIR)
    test_proba = predict_proba(model, X_test, mean, std)
    metrics = compute_metrics(y_test, test_proba, threshold)
    metrics["test_patient"] = test_patient
    metrics["model"] = model_name
    metrics["n_test"] = int(len(y_test))
    metrics["n_pos_test"] = int((y_test == 1).sum())
    metrics["n_neg_test"] = int((y_test == 0).sum())
    metrics["fit_time_sec"] = fit_time
    metrics["n_params"] = n_params
    metrics["fold_total_sec"] = time.time() - t0

    # Save raw probabilities for downstream calibration analysis
    proba_dir = RESULTS_DIR / "probabilities"
    proba_dir.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        proba_dir / f"{model_name}_{test_patient}.npz",
        y_true=y_test, y_proba=test_proba, threshold=threshold,
    )

    print(f"  Done in {metrics['fold_total_sec']:.1f}s | "
          f"AUC={metrics['auc']:.3f} MCC={metrics['mcc']:+.3f} "
          f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        choices=["eegnet", "shallowconvnet", "cnnlstm"])
    parser.add_argument("--patients", default=None,
                        help="Comma-separated patient list (default: all 21)")
    args = parser.parse_args()

    set_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    test_list = args.patients.split(",") if args.patients else PATIENTS

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM available: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    out_csv = RESULTS_DIR / f"per_fold_{args.model}.csv"

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

        for test_patient in test_list:
            train_patients = [p for p in PATIENTS if p != test_patient]
            try:
                metrics = run_fold(args.model, test_patient, train_patients)
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

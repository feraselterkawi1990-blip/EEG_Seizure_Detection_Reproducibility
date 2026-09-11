"""
loocv.py
========
Leave-One-Patient-Out Cross-Validation.

For each held-out patient k:
    - Pool all segments from the remaining 21 patients
    - Train each model on the pooled training set
    - Evaluate each model on a class-balanced test set drawn from patient k
    - Record metrics

Aggregate across folds: report mean ± std for each metric.

Critical correctness properties:
- No segment from the held-out patient ever appears in training
- Class balancing is applied AFTER the patient split (no leakage)
- Random seeds are set deterministically per fold
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import config
from . import cache, models, metrics

logger = logging.getLogger(__name__)


# =====================================================================
# Class balancing helpers
# =====================================================================
def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return indices for a class-balanced subset.

    Undersamples the majority class to match the minority class size.
    """
    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    n = min(len(pos_idx), len(neg_idx))
    if n == 0:
        return np.array([], dtype=int)
    pos_sel = rng.choice(pos_idx, size=n, replace=False)
    neg_sel = rng.choice(neg_idx, size=n, replace=False)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


def pool_training_data(
    train_patients: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate cached features from all training patients.

    Returns
    -------
    F_train, X_train_raw, y_train, patient_origin
    """
    F_list, X_list, y_list, origin = [], [], [], []
    for i, pid in enumerate(train_patients):
        d = cache.load_patient_cache(pid)
        F_list.append(d["F"])
        X_list.append(d["X_raw"])
        y_list.append(d["y"])
        origin.append(np.full(len(d["y"]), i, dtype=np.int16))
    F = np.concatenate(F_list, axis=0)
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    o = np.concatenate(origin, axis=0)
    return F, X, y, o


# =====================================================================
# Per-fold evaluation
# =====================================================================
def evaluate_fold(test_patient: str,
                  train_patients: List[str],
                  fold_idx: int) -> Dict[str, dict]:
    """
    Run one LOOCV fold: train on `train_patients`, test on `test_patient`.

    Returns
    -------
    Dict mapping model_name -> metrics dict.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + fold_idx)
    fold_results = {}

    # --- Load test data ---
    test_data = cache.load_patient_cache(test_patient)
    F_test_full = test_data["F"]
    X_test_full = test_data["X_raw"]
    y_test_full = test_data["y"]

    # Skip patient if they have no seizures (can't evaluate)
    if y_test_full.sum() == 0:
        logger.warning(f"  {test_patient} has no ictal segments; skipping fold")
        return {}

    # --- Build balanced test set ---
    if config.TEST_BALANCE:
        test_idx = balanced_indices(y_test_full, rng)
    else:
        test_idx = np.arange(len(y_test_full))
    F_test = F_test_full[test_idx]
    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]

    # --- Load + pool training data ---
    F_train, X_train, y_train, _ = pool_training_data(train_patients)

    # Optionally balance training set
    if config.TRAIN_BALANCE_STRATEGY == "undersample":
        train_idx = balanced_indices(y_train, rng)
        F_train = F_train[train_idx]
        X_train = X_train[train_idx]
        y_train = y_train[train_idx]

    logger.info(
        f"  train: {len(y_train)} segs ({int(y_train.sum())} ictal), "
        f"test: {len(y_test)} segs ({int(y_test.sum())} ictal)"
    )

    # =================================================================
    # Random Forest
    # =================================================================
    if "RF" in config.MODELS_TO_RUN:
        t0 = time.time()
        rf = models.build_rf()
        rf.fit(F_train, y_train)
        y_pred = rf.predict(F_test)
        y_score = rf.predict_proba(F_test)[:, 1]
        m = metrics.compute_metrics(y_test, y_pred, y_score)
        m["fit_time_sec"] = time.time() - t0
        fold_results["RF"] = m
        logger.info(f"    RF  acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                    f"spec={m['specificity']:.3f} auc={m['auc']:.3f}")

    # =================================================================
    # SVM
    # =================================================================
    if "SVM" in config.MODELS_TO_RUN:
        t0 = time.time()
        svm = models.build_svm()
        # SVM is slow on huge feature matrices; subsample if needed
        max_svm_train = 20000
        if len(y_train) > max_svm_train:
            idx_sub = rng.choice(len(y_train), size=max_svm_train, replace=False)
            svm.fit(F_train[idx_sub], y_train[idx_sub])
        else:
            svm.fit(F_train, y_train)
        y_pred = svm.predict(F_test)
        y_score = svm.predict_proba(F_test)[:, 1]
        m = metrics.compute_metrics(y_test, y_pred, y_score)
        m["fit_time_sec"] = time.time() - t0
        fold_results["SVM"] = m
        logger.info(f"    SVM acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                    f"spec={m['specificity']:.3f} auc={m['auc']:.3f}")

    # =================================================================
    # LSTM
    # =================================================================
    if "LSTM" in config.MODELS_TO_RUN:
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(config.RANDOM_SEED + fold_idx)

            # Reshape to (n_samples, n_timesteps, n_channels)
            X_tr = np.transpose(X_train, (0, 2, 1))
            X_te = np.transpose(X_test, (0, 2, 1))
            X_tr = models.normalize_sequences(X_tr)
            X_te = models.normalize_sequences(X_te)

            t0 = time.time()
            lstm = models.build_lstm(input_shape=X_tr.shape[1:])

            # Hold out 15% of training as validation for early stopping
            n_val = max(1, int(0.15 * len(y_train)))
            val_idx = rng.choice(len(y_train), size=n_val, replace=False)
            train_idx = np.setdiff1d(np.arange(len(y_train)), val_idx)

            models.train_keras_model(
                lstm,
                X_tr[train_idx], y_train[train_idx],
                X_tr[val_idx], y_train[val_idx],
                params=config.LSTM_PARAMS,
            )
            y_score = lstm.predict(X_te, verbose=0)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            m = metrics.compute_metrics(y_test, y_pred, y_score)
            m["fit_time_sec"] = time.time() - t0
            fold_results["LSTM"] = m
            logger.info(f"    LSTM acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                        f"spec={m['specificity']:.3f} auc={m['auc']:.3f}")
        except Exception as e:
            logger.error(f"    LSTM failed: {e}")
            fold_results["LSTM"] = {"error": str(e)}

    # =================================================================
    # CNN
    # =================================================================
    if "CNN" in config.MODELS_TO_RUN:
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(config.RANDOM_SEED + fold_idx)

            X_tr = np.transpose(X_train, (0, 2, 1))
            X_te = np.transpose(X_test, (0, 2, 1))
            X_tr = models.normalize_sequences(X_tr)
            X_te = models.normalize_sequences(X_te)

            t0 = time.time()
            cnn = models.build_cnn(input_shape=X_tr.shape[1:])

            n_val = max(1, int(0.15 * len(y_train)))
            val_idx = rng.choice(len(y_train), size=n_val, replace=False)
            train_idx = np.setdiff1d(np.arange(len(y_train)), val_idx)

            models.train_keras_model(
                cnn,
                X_tr[train_idx], y_train[train_idx],
                X_tr[val_idx], y_train[val_idx],
                params=config.CNN_PARAMS,
            )
            y_score = cnn.predict(X_te, verbose=0)[:, 1]
            y_pred = (y_score >= 0.5).astype(int)
            m = metrics.compute_metrics(y_test, y_pred, y_score)
            m["fit_time_sec"] = time.time() - t0
            fold_results["CNN"] = m
            logger.info(f"    CNN acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                        f"spec={m['specificity']:.3f} auc={m['auc']:.3f}")
        except Exception as e:
            logger.error(f"    CNN failed: {e}")
            fold_results["CNN"] = {"error": str(e)}

    # Free memory between folds
    del F_train, X_train, y_train
    return fold_results


# =====================================================================
# Full LOOCV orchestrator
# =====================================================================
def run_loocv() -> pd.DataFrame:
    """
    Run LOOCV across all patients in config.PATIENTS.

    Returns the full per-fold results table and writes it to TABLES_PATH.
    Also computes and writes the summary mean ± std table.
    """
    all_results = []

    for fold_idx, test_patient in enumerate(config.PATIENTS):
        train_patients = [p for p in config.PATIENTS if p != test_patient]

        logger.info(f"\n=== Fold {fold_idx+1}/{len(config.PATIENTS)}: "
                    f"holding out {test_patient} ===")

        try:
            fold_metrics = evaluate_fold(test_patient, train_patients, fold_idx)
        except Exception as e:
            logger.error(f"Fold {test_patient} failed: {e}")
            continue

        for model_name, m in fold_metrics.items():
            row = {"fold": fold_idx, "test_patient": test_patient,
                   "model": model_name, **m}
            all_results.append(row)

    df = pd.DataFrame(all_results)
    per_fold_path = config.TABLES_PATH / "loocv_per_fold.csv"
    df.to_csv(per_fold_path, index=False)
    logger.info(f"Per-fold results written to {per_fold_path}")

    # Summary: mean ± std across folds, per model
    summary = (
        df.groupby("model")[["accuracy", "sensitivity", "specificity", "auc"]]
          .agg(["mean", "std", "count"])
    )
    summary_path = config.TABLES_PATH / "loocv_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Summary written to {summary_path}")

    print("\n" + "=" * 70)
    print("LOOCV SUMMARY (mean ± std across folds)")
    print("=" * 70)
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        print(f"\n{model} (n_folds = {len(sub)}):")
        for col in ["accuracy", "sensitivity", "specificity", "auc"]:
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f"  {col:>12}: {vals.mean():.3f} ± {vals.std():.3f}")

    return df

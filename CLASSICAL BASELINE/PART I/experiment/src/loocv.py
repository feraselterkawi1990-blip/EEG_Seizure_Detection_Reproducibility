"""
loocv.py
========
Leave-One-Patient-Out Cross-Validation.

For each held-out patient k:
    - Pool all segments from the remaining 20 patients
    - Train each model on the pooled training set
    - Evaluate each model on a class-balanced test set drawn from patient k
    - Record metrics

Aggregate across folds: report mean +/- std for each metric.

Critical correctness properties:
- No segment from the held-out patient ever appears in training
- Class balancing is applied AFTER the patient split (no leakage)
- Random seeds are set deterministically per fold
- ALL classifiers (RF, SVM, XGB, LSTM, CNN) use the SAME held-out
  validation set for threshold tuning via Youden's J — fair comparison

CHANGELOG (v2 - May 2026):
  - FIX #1: LSTM and CNN now use Youden's J threshold tuning on validation,
            same as RF/SVM. Previously they used a fixed 0.5 threshold,
            making the cross-model comparison unfair.
  - FIX #2: All classifiers now use the SAME validation split derived from
            the SAME train_mask. Previously LSTM/CNN drew an independent
            15% split that overlapped with RF/SVM training data.
  - FIX #4: All numerical parameters now read from config.py. Previously
            VARIANCE_THRESHOLD, MAX_TRAIN_NEG, SVM_TRAIN_SUBSAMPLE,
            VAL_SPLIT_RATIO, VAL_SPLIT_FLOOR, THRESHOLD_RANGE, and
            THRESHOLD_N_STEPS were hard-coded here.
  - NEW:    XGBoost block (issue #20).
  - NEW:    Per-fold preprocessing diagnostics (n_features_kept) propagated
            to the per-fold results CSV for downstream analysis.
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

import config
from . import cache, models, metrics

logger = logging.getLogger(__name__)


# =====================================================================
# Class balancing helpers
# =====================================================================
def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Return indices for a class-balanced subset.

    All minority-class samples are kept (they are reordered randomly via
    rng.choice with size equal to the full minority count). The majority
    class is randomly subsampled without replacement to match.

    For CHB-MIT test sets this means: keep ALL ictal segments + sample an
    equal number of non-ictal segments (matches thesis §3.5.2).
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
    load_raw: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Concatenate cached features from all training patients.

    Parameters
    ----------
    train_patients : list of patient IDs to pool
    load_raw : if False, skip loading X_raw (saves ~10 GB RAM when only
               classical models RF/SVM/XGB are used).

    Returns
    -------
    F_train, X_train_raw, y_train, patient_origin
    (X_train_raw is None when load_raw=False)
    """
    F_list, X_list, y_list, origin = [], [], [], []
    for i, pid in enumerate(train_patients):
        d = cache.load_patient_cache(pid)
        F_list.append(d["F"])
        if load_raw:
            X_list.append(d["X_raw"])
        y_list.append(d["y"])
        origin.append(np.full(len(d["y"]), i, dtype=np.int16))
    F = np.concatenate(F_list, axis=0)
    X = np.concatenate(X_list, axis=0) if load_raw else None
    y = np.concatenate(y_list, axis=0)
    o = np.concatenate(origin, axis=0)
    return F, X, y, o


# =====================================================================
# Threshold tuning (shared by ALL classifiers in v2)
# =====================================================================
def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Find threshold maximising Youden's J = sensitivity + specificity - 1
    on a held-out validation set.

    Search space: config.THRESHOLD_N_STEPS evenly-spaced thresholds in
    config.THRESHOLD_RANGE.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    lo, hi = config.THRESHOLD_RANGE
    thresholds = np.linspace(lo, hi, config.THRESHOLD_N_STEPS)
    best_j, best_t = -np.inf, 0.5
    for t in thresholds:
        preds = (scores >= t).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        j = sens + spec - 1
        if j > best_j:
            best_j, best_t = j, t
    return float(best_t)


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
    # Only load raw signals if a deep model (LSTM/CNN) is in the run list.
    # This saves ~10 GB RAM when only RF/SVM/XGB are used.
    needs_raw = bool(set(config.MODELS_TO_RUN) & {"LSTM", "CNN"})
    F_train, X_train, y_train, _ = pool_training_data(
        train_patients, load_raw=needs_raw
    )

    # Optionally additionally balance training set 1:1 BEFORE the cap.
    # Default config sets TRAIN_BALANCE_STRATEGY = "class_weight" which
    # SKIPS this block — see config.py docstring.
    if config.TRAIN_BALANCE_STRATEGY == "undersample":
        train_idx_bal = balanced_indices(y_train, rng)
        F_train = F_train[train_idx_bal]
        if X_train is not None:
            X_train = X_train[train_idx_bal]
        y_train = y_train[train_idx_bal]

    # ----------------------------------------------------------------
    # Stratified subsampling for tractability.
    # Pooled training data can exceed 5-6 million segments, which is
    # impractical for sklearn's RF (and intractable for SVM). Keep ALL
    # ictal samples (rare and valuable) and subsample non-ictal to
    # config.MAX_TRAIN_NEG. This produces ~1:5 ratio (Thesis §3.5.3).
    # ----------------------------------------------------------------
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(neg_idx) > config.MAX_TRAIN_NEG:
        neg_sub = rng.choice(neg_idx, size=config.MAX_TRAIN_NEG, replace=False)
        keep = np.concatenate([pos_idx, neg_sub])
        rng.shuffle(keep)
        F_train = F_train[keep]
        if X_train is not None:
            X_train = X_train[keep]
        y_train = y_train[keep]

    logger.info(
        f"  train: {len(y_train)} segs ({int(y_train.sum())} ictal), "
        f"test: {len(y_test)} segs ({int(y_test.sum())} ictal)"
    )

    # ----------------------------------------------------------------
    # Hold out a validation slice for threshold tuning. ALL classifiers
    # use the SAME val_idx so cross-model comparisons are fair (FIX #2).
    # ----------------------------------------------------------------
    n_total = len(y_train)
    n_val = max(int(config.VAL_SPLIT_RATIO * n_total), config.VAL_SPLIT_FLOOR)
    n_val = min(n_val, n_total - 1)  # safety
    val_idx = rng.choice(n_total, size=n_val, replace=False)
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)

    # Feature views (RF/SVM/XGB)
    F_tr, y_tr = F_train[train_idx], y_train[train_idx]
    F_val, y_val = F_train[val_idx], y_train[val_idx]

    # Raw views (LSTM/CNN), if loaded
    if needs_raw and X_train is not None:
        X_tr_raw = X_train[train_idx]
        X_val_raw = X_train[val_idx]
    else:
        X_tr_raw = X_val_raw = None

    # ----------------------------------------------------------------
    # Feature preprocessing for classical models.
    #   1) Replace inf/NaN values from numerical issues
    #   2) Drop near-constant features (var < config.VARIANCE_THRESHOLD)
    #   3) Standardize remaining features to zero mean / unit variance
    # All transformations are FIT on training data only and APPLIED to
    # validation and test data — no leakage.
    # ----------------------------------------------------------------
    F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    F_val = np.nan_to_num(F_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    var_filter = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    F_tr = var_filter.fit_transform(F_tr)
    F_val = var_filter.transform(F_val)
    F_test = var_filter.transform(F_test)

    scaler = StandardScaler()
    F_tr = scaler.fit_transform(F_tr).astype(np.float32)
    F_val = scaler.transform(F_val).astype(np.float32)
    F_test = scaler.transform(F_test).astype(np.float32)

    n_features_kept = F_tr.shape[1]
    logger.info(
        f"  features: {n_features_kept}/{config.N_TOTAL_FEATURES} "
        f"kept after variance filter, scaled"
    )

    # =================================================================
    # Random Forest
    # =================================================================
    if "RF" in config.MODELS_TO_RUN:
        t0 = time.time()
        rf = models.build_rf()
        rf.fit(F_tr, y_tr)
        val_scores = rf.predict_proba(F_val)[:, 1]
        thr = tune_threshold(val_scores, y_val)
        y_score = rf.predict_proba(F_test)[:, 1]
        y_pred = (y_score >= thr).astype(int)
        m = metrics.compute_metrics(y_test, y_pred, y_score)
        m["fit_time_sec"] = time.time() - t0
        m["threshold"] = thr
        m["n_features_kept"] = n_features_kept
        fold_results["RF"] = m
        logger.info(f"    RF  acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                    f"spec={m['specificity']:.3f} auc={m['auc']:.3f} thr={thr:.2f}")

    # =================================================================
    # SVM
    # =================================================================
    if "SVM" in config.MODELS_TO_RUN:
        t0 = time.time()
        svm = models.build_svm()
        if len(y_tr) > config.SVM_TRAIN_SUBSAMPLE:
            idx_sub = rng.choice(
                len(y_tr), size=config.SVM_TRAIN_SUBSAMPLE, replace=False
            )
            svm.fit(F_tr[idx_sub], y_tr[idx_sub])
        else:
            svm.fit(F_tr, y_tr)
        val_scores = svm.predict_proba(F_val)[:, 1]
        thr = tune_threshold(val_scores, y_val)
        y_score = svm.predict_proba(F_test)[:, 1]
        y_pred = (y_score >= thr).astype(int)
        m = metrics.compute_metrics(y_test, y_pred, y_score)
        m["fit_time_sec"] = time.time() - t0
        m["threshold"] = thr
        m["n_features_kept"] = n_features_kept
        fold_results["SVM"] = m
        logger.info(f"    SVM acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                    f"spec={m['specificity']:.3f} auc={m['auc']:.3f} thr={thr:.2f}")

    # =================================================================
    # XGBoost  (NEW in v2)
    # =================================================================
    if "XGB" in config.MODELS_TO_RUN:
        t0 = time.time()
        xgb = models.build_xgb()
        # XGBoost handles its own internal validation if we pass eval_set.
        # We don't use early stopping here to keep the training pool
        # comparable to RF (which doesn't have early stopping either).
        xgb.fit(F_tr, y_tr)
        val_scores = xgb.predict_proba(F_val)[:, 1]
        thr = tune_threshold(val_scores, y_val)
        y_score = xgb.predict_proba(F_test)[:, 1]
        y_pred = (y_score >= thr).astype(int)
        m = metrics.compute_metrics(y_test, y_pred, y_score)
        m["fit_time_sec"] = time.time() - t0
        m["threshold"] = thr
        m["n_features_kept"] = n_features_kept
        fold_results["XGB"] = m
        logger.info(f"    XGB acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                    f"spec={m['specificity']:.3f} auc={m['auc']:.3f} thr={thr:.2f}")

    # =================================================================
    # LSTM
    # =================================================================
    if "LSTM" in config.MODELS_TO_RUN and X_tr_raw is not None:
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(config.RANDOM_SEED + fold_idx)

            # Reshape (n, n_ch, n_t) -> (n, n_t, n_ch) for Keras RNN/Conv1D
            X_tr_seq = np.transpose(X_tr_raw, (0, 2, 1))
            X_val_seq = np.transpose(X_val_raw, (0, 2, 1))
            X_te_seq = np.transpose(X_test, (0, 2, 1))
            X_tr_seq = models.normalize_sequences(X_tr_seq)
            X_val_seq = models.normalize_sequences(X_val_seq)
            X_te_seq = models.normalize_sequences(X_te_seq)

            t0 = time.time()
            lstm = models.build_lstm(input_shape=X_tr_seq.shape[1:])
            models.train_keras_model(
                lstm,
                X_tr_seq, y_tr,
                X_val_seq, y_val,
                params=config.LSTM_PARAMS,
            )
            # FIX #1: tune threshold on validation, then apply to test
            val_scores = lstm.predict(X_val_seq, verbose=0)[:, 1]
            thr = tune_threshold(val_scores, y_val)
            y_score = lstm.predict(X_te_seq, verbose=0)[:, 1]
            y_pred = (y_score >= thr).astype(int)
            m = metrics.compute_metrics(y_test, y_pred, y_score)
            m["fit_time_sec"] = time.time() - t0
            m["threshold"] = thr
            m["n_features_kept"] = n_features_kept
            fold_results["LSTM"] = m
            logger.info(f"    LSTM acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                        f"spec={m['specificity']:.3f} auc={m['auc']:.3f} thr={thr:.2f}")
        except Exception as e:
            logger.error(f"    LSTM failed: {e}")
            fold_results["LSTM"] = {"error": str(e)}

    # =================================================================
    # CNN
    # =================================================================
    if "CNN" in config.MODELS_TO_RUN and X_tr_raw is not None:
        try:
            import tensorflow as tf
            tf.keras.utils.set_random_seed(config.RANDOM_SEED + fold_idx)

            X_tr_seq = np.transpose(X_tr_raw, (0, 2, 1))
            X_val_seq = np.transpose(X_val_raw, (0, 2, 1))
            X_te_seq = np.transpose(X_test, (0, 2, 1))
            X_tr_seq = models.normalize_sequences(X_tr_seq)
            X_val_seq = models.normalize_sequences(X_val_seq)
            X_te_seq = models.normalize_sequences(X_te_seq)

            t0 = time.time()
            cnn = models.build_cnn(input_shape=X_tr_seq.shape[1:])
            models.train_keras_model(
                cnn,
                X_tr_seq, y_tr,
                X_val_seq, y_val,
                params=config.CNN_PARAMS,
            )
            # FIX #1: tune threshold on validation, then apply to test
            val_scores = cnn.predict(X_val_seq, verbose=0)[:, 1]
            thr = tune_threshold(val_scores, y_val)
            y_score = cnn.predict(X_te_seq, verbose=0)[:, 1]
            y_pred = (y_score >= thr).astype(int)
            m = metrics.compute_metrics(y_test, y_pred, y_score)
            m["fit_time_sec"] = time.time() - t0
            m["threshold"] = thr
            m["n_features_kept"] = n_features_kept
            fold_results["CNN"] = m
            logger.info(f"    CNN acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                        f"spec={m['specificity']:.3f} auc={m['auc']:.3f} thr={thr:.2f}")
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
    Also computes and writes the summary mean +/- std table.
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

    # Summary: mean +/- std across folds, per model
    summary = (
        df.groupby("model")[["accuracy", "sensitivity", "specificity", "auc"]]
          .agg(["mean", "std", "count"])
    )
    summary_path = config.TABLES_PATH / "loocv_summary.csv"
    summary.to_csv(summary_path)
    logger.info(f"Summary written to {summary_path}")

    print("\n" + "=" * 70)
    print("LOOCV SUMMARY (mean +/- std across folds)")
    print("=" * 70)
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        print(f"\n{model} (n_folds = {len(sub)}):")
        for col in ["accuracy", "sensitivity", "specificity", "auc"]:
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f"  {col:>12}: {vals.mean():.3f} +/- {vals.std():.3f}")

    return df

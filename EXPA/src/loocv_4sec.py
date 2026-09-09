"""
loocv_4sec.py
=============
Stage 3-EXP-A LOOCV: Part I + 4-second windows + StandardScaler.

This completes the 2x2 factorial design:

                    | 1-sec               | 4-sec
   ----------------|---------------------|---------------------
   StandardScaler  | Part I (0.575)      | EXP A ← THIS
                   |                     |
   per-seg z       | iso (0.589)         | Stage 3 (0.866)

KEY CONFIGURATION:
  - 23 native channels (Part I)
  - 4-second windows (Part II) — THE INTENDED CHANGE
  - 0.5-30 Hz bandpass (Part I)
  - 207 features → variance filter → ~92 (Part I logic)
  - StandardScaler over training pool (Part I, NOT per-seg z)
  - RF + XGB classifiers (skip SVM to save 8 hours; iso showed it's unstable)
  - 21 patient LOOCV
  - MAX_TRAIN_NEG = 100,000

Output: results/tables/loocv_per_fold_4sec_stdscaler.csv

Cache used: results/cache_4sec/ (built by cache_4sec.py)
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler  # ← Part I logic, NOT per-seg z

import config
from . import cache_4sec, models, metrics

logger = logging.getLogger(__name__)


def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """50/50 class-balanced subsample."""
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


def pool_training_data_4sec(train_patients: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Pool 4-sec cached features."""
    F_list, y_list = [], []
    for pid in train_patients:
        d = cache_4sec.load_patient_cache_4sec(pid)
        F_list.append(d["F"])
        y_list.append(d["y"])
    return np.concatenate(F_list, axis=0), np.concatenate(y_list, axis=0)


def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Youden's J threshold tuning."""
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


def compute_mcc(tp: int, tn: int, fp: int, fn: int) -> float:
    """Matthews Correlation Coefficient."""
    num = (tp * tn) - (fp * fn)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float(num / denom)


def evaluate_fold_4sec(test_patient: str,
                       train_patients: List[str],
                       fold_idx: int,
                       models_to_run: List[str]) -> Dict[str, dict]:
    """
    Run one LOOCV fold for EXP A: Part I + 4-sec + StandardScaler.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + fold_idx)
    fold_results = {}

    # --- Load test data ---
    test_data = cache_4sec.load_patient_cache_4sec(test_patient)
    F_test_full = test_data["F"]
    y_test_full = test_data["y"]

    if y_test_full.sum() == 0:
        logger.warning(f"  {test_patient} has no ictal segments; skipping fold")
        return {}

    # --- Balanced test set ---
    if config.TEST_BALANCE:
        test_idx = balanced_indices(y_test_full, rng)
    else:
        test_idx = np.arange(len(y_test_full))
    F_test = F_test_full[test_idx]
    y_test = y_test_full[test_idx]

    # --- Pool training data ---
    F_train, y_train = pool_training_data_4sec(train_patients)

    if config.TRAIN_BALANCE_STRATEGY == "undersample":
        train_idx_bal = balanced_indices(y_train, rng)
        F_train = F_train[train_idx_bal]
        y_train = y_train[train_idx_bal]

    # MAX_TRAIN_NEG subsampling
    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)
    if len(neg_idx) > config.MAX_TRAIN_NEG:
        neg_sub = rng.choice(neg_idx, size=config.MAX_TRAIN_NEG, replace=False)
        keep = np.concatenate([pos_idx, neg_sub])
        rng.shuffle(keep)
        F_train = F_train[keep]
        y_train = y_train[keep]

    logger.info(
        f"  train: {len(y_train)} segs ({int(y_train.sum())} ictal), "
        f"test: {len(y_test)} segs ({int(y_test.sum())} ictal)"
    )

    # --- Validation slice ---
    n_total = len(y_train)
    n_val = max(int(config.VAL_SPLIT_RATIO * n_total), config.VAL_SPLIT_FLOOR)
    n_val = min(n_val, n_total - 1)
    val_idx = rng.choice(n_total, size=n_val, replace=False)
    train_mask = np.ones(n_total, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.flatnonzero(train_mask)

    F_tr, y_tr = F_train[train_idx], y_train[train_idx]
    F_val, y_val = F_train[val_idx], y_train[val_idx]

    # ----------------------------------------------------------------
    # Feature preprocessing — PART I LOGIC:
    #   1) Replace inf/NaN
    #   2) Variance filter
    #   3) StandardScaler over training pool
    # ----------------------------------------------------------------
    F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    F_val = np.nan_to_num(F_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    F_test_clean = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    var_filter = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    F_tr = var_filter.fit_transform(F_tr)
    F_val = var_filter.transform(F_val)
    F_test_clean = var_filter.transform(F_test_clean)

    # *** PART I LOGIC: StandardScaler ***
    scaler = StandardScaler()
    F_tr = scaler.fit_transform(F_tr)
    F_val = scaler.transform(F_val)
    F_test_clean = scaler.transform(F_test_clean)

    n_features_kept = F_tr.shape[1]
    logger.info(
        f"  features: {n_features_kept}/{config.N_TOTAL_FEATURES} kept (with StandardScaler)"
    )

    # =================================================================
    # Run each requested model
    # =================================================================
    for model_name in models_to_run:
        t0 = time.time()
        try:
            if model_name == "RF":
                clf = models.build_rf()
                X_tr_use, y_tr_use = F_tr, y_tr
            elif model_name == "SVM":
                if len(F_tr) > config.SVM_TRAIN_SUBSAMPLE:
                    sub_idx = rng.choice(len(F_tr), size=config.SVM_TRAIN_SUBSAMPLE,
                                          replace=False)
                    X_tr_use = F_tr[sub_idx]
                    y_tr_use = y_tr[sub_idx]
                else:
                    X_tr_use, y_tr_use = F_tr, y_tr
                clf = models.build_svm()
            elif model_name == "XGB":
                clf = models.build_xgb()
                X_tr_use, y_tr_use = F_tr, y_tr
            else:
                logger.warning(f"  Unknown model: {model_name}, skipping")
                continue

            clf.fit(X_tr_use, y_tr_use)
            val_scores = clf.predict_proba(F_val)[:, 1]
            thr = tune_threshold(val_scores, y_val)
            y_score = clf.predict_proba(F_test_clean)[:, 1]
            y_pred = (y_score >= thr).astype(int)
            m = metrics.compute_metrics(y_test, y_pred, y_score)
            m["mcc"] = compute_mcc(m["tp"], m["tn"], m["fp"], m["fn"])
            m["fit_time_sec"] = time.time() - t0
            m["threshold"] = thr
            m["n_features_kept"] = n_features_kept
            fold_results[model_name] = m
            logger.info(
                f"    {model_name:>3}  acc={m['accuracy']:.3f} sens={m['sensitivity']:.3f} "
                f"spec={m['specificity']:.3f} auc={m['auc']:.3f} mcc={m['mcc']:+.3f} "
                f"thr={thr:.2f}  ({m['fit_time_sec']:.0f}s)"
            )
        except Exception as e:
            logger.error(f"    {model_name} failed: {e}")
            continue

    return fold_results


def run_loocv_4sec(models_to_run: List[str] = None) -> pd.DataFrame:
    """
    Run EXP A LOOCV: Part I config + 4-sec windows + StandardScaler.
    Default models: RF + SVM + XGB (matches Part I + iso for consistency).
    Removing SVM would be cherry-picking; we keep it even though it's slow.
    """
    if models_to_run is None:
        models_to_run = ["RF", "SVM", "XGB"]

    logger.info(f"Models to run: {models_to_run}")

    all_results = []

    for fold_idx, test_patient in enumerate(config.PATIENTS):
        train_patients = [p for p in config.PATIENTS if p != test_patient]

        logger.info(f"\n=== Fold {fold_idx+1}/{len(config.PATIENTS)}: "
                    f"holding out {test_patient} ===")

        try:
            fold_metrics = evaluate_fold_4sec(test_patient, train_patients,
                                              fold_idx, models_to_run)
        except Exception as e:
            logger.error(f"Fold {test_patient} failed: {e}")
            continue

        for model_name, m in fold_metrics.items():
            row = {"fold": fold_idx, "test_patient": test_patient,
                   "model": model_name, **m}
            all_results.append(row)

    df = pd.DataFrame(all_results)
    out_path = config.TABLES_PATH / "loocv_per_fold_4sec_stdscaler.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"\nEXP A results written to {out_path}")

    # Summary
    print("\n" + "=" * 78)
    print("EXP A SUMMARY: Part I + 4-sec windows + StandardScaler")
    print("=" * 78)
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        print(f"\n{model} (n_folds = {len(sub)}):")
        for col in ["accuracy", "sensitivity", "specificity", "auc", "mcc"]:
            if col in sub.columns:
                vals = sub[col].dropna()
                if len(vals) > 0:
                    print(f"  {col:>12}: {vals.mean():+.3f} +/- {vals.std():.3f}")

    # Compare to Part I + iso
    part1_path = config.TABLES_PATH / "loocv_per_fold.csv"
    iso_path = config.TABLES_PATH / "loocv_per_fold_iso_full.csv"

    print("\n" + "=" * 78)
    print("2x2 FACTORIAL DESIGN — Aggregate AUCs")
    print("=" * 78)
    print(f"\n  Part I (1-sec, StdScaler):     0.5745")
    if iso_path.exists():
        df_iso = pd.read_csv(iso_path)
        df_iso_rf = df_iso[df_iso["model"] == "RF"]
        if not df_iso_rf.empty:
            print(f"  iso    (1-sec, per-seg z):     {df_iso_rf['auc'].mean():.4f}")
    df_4sec_rf = df[df["model"] == "RF"]
    if not df_4sec_rf.empty:
        print(f"  EXP A  (4-sec, StdScaler):     {df_4sec_rf['auc'].mean():.4f}  ← NEW")
    print(f"  Stage 3 (4-sec, per-seg z):    0.8660")

    return df

"""
loocv_iso.py
============
Stage 3-isolated LOOCV — RF baseline with per-segment z-score normalisation
applied to the raw EEG BEFORE feature extraction, instead of per-feature
StandardScaler over the training pool.

This module mirrors src/loocv.py exactly, with TWO changes:
  1. Builds a NEW cache (cache_iso/) where features are extracted from
     per-segment z-scored signals. This isolates the normalisation effect.
  2. SKIPS the StandardScaler step in the LOOCV (variance filter still
     applied to drop dead features). Per-segment z-score IS the normalisation.

Everything else IDENTICAL to Part I:
  - 23 native channels
  - 1-second windows, 50% overlap
  - Bandpass 0.5-30 Hz
  - 9 features × 23 = 207 → variance filter → 92
  - Random Forest n_estimators=200, class_weight='balanced'
  - MAX_TRAIN_NEG = 100,000
  - 10% validation slice with floor 1000
  - Youden-J threshold tuning
  - Random seed 42 + fold_idx
  - LOOCV across 21 patients
  - Balanced 50/50 test set per held-out patient

Usage
-----
After preparing the iso cache (see scripts/build_iso_cache.py):

    from src.loocv_iso import run_loocv_iso
    df = run_loocv_iso()

Or via the entry-point script:

    python scripts/run_loocv_iso.py
"""

import logging
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
# Note: NO StandardScaler import — per-segment z-score replaces it.

import config
from . import cache_iso, models, metrics

logger = logging.getLogger(__name__)


# =====================================================================
# Class balancing helpers (identical to loocv.py)
# =====================================================================
def balanced_indices(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Return indices for a class-balanced subset (50/50)."""
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


def pool_training_data_iso(train_patients: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Pool ISO-cached features from training patients.

    Loads only F (no X_raw needed for RF) from cache_iso/.
    """
    F_list, y_list, origin = [], [], []
    for i, pid in enumerate(train_patients):
        d = cache_iso.load_patient_cache_iso(pid)
        F_list.append(d["F"])
        y_list.append(d["y"])
        origin.append(np.full(len(d["y"]), i, dtype=np.int16))
    F = np.concatenate(F_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    o = np.concatenate(origin, axis=0)
    return F, y, o


# =====================================================================
# Threshold tuning (identical to loocv.py)
# =====================================================================
def tune_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Find threshold maximising Youden's J on validation."""
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
# Per-fold evaluation — RF only
# =====================================================================
def evaluate_fold_iso(test_patient: str,
                      train_patients: List[str],
                      fold_idx: int) -> Dict[str, dict]:
    """
    Run one LOOCV fold for Stage 3-isolated.

    Differs from loocv.evaluate_fold ONLY in:
      - Loads from cache_iso (per-segment z-scored features)
      - SKIPS StandardScaler (per-segment z-score IS the normalisation)
      - Runs RF only (the experiment is about RF, not SVM/XGB)
    """
    rng = np.random.default_rng(config.RANDOM_SEED + fold_idx)
    fold_results = {}

    # --- Load test data from ISO cache ---
    test_data = cache_iso.load_patient_cache_iso(test_patient)
    F_test_full = test_data["F"]
    y_test_full = test_data["y"]

    if y_test_full.sum() == 0:
        logger.warning(f"  {test_patient} has no ictal segments; skipping fold")
        return {}

    # --- Build balanced test set ---
    if config.TEST_BALANCE:
        test_idx = balanced_indices(y_test_full, rng)
    else:
        test_idx = np.arange(len(y_test_full))
    F_test = F_test_full[test_idx]
    y_test = y_test_full[test_idx]

    # --- Pool ISO-cached training data ---
    F_train, y_train, _ = pool_training_data_iso(train_patients)

    # Optional 1:1 undersampling (only if config requests it; default skips)
    if config.TRAIN_BALANCE_STRATEGY == "undersample":
        train_idx_bal = balanced_indices(y_train, rng)
        F_train = F_train[train_idx_bal]
        y_train = y_train[train_idx_bal]

    # Stratified subsampling for tractability (identical to loocv.py)
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

    # --- Validation slice for Youden-J tuning ---
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
    # Feature preprocessing — DIFFERS FROM loocv.py:
    #   1) Replace inf/NaN values from numerical issues
    #   2) Drop near-constant features (variance filter)
    #   3) NO StandardScaler — per-segment z-score in the cache IS the
    #      normalisation. Adding StandardScaler on top would be
    #      double-normalisation and would defeat the isolation goal.
    # ----------------------------------------------------------------
    F_tr = np.nan_to_num(F_tr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    F_val = np.nan_to_num(F_val, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    var_filter = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    F_tr = var_filter.fit_transform(F_tr)
    F_val = var_filter.transform(F_val)
    F_test = var_filter.transform(F_test)

    n_features_kept = F_tr.shape[1]
    logger.info(
        f"  features: {n_features_kept}/{config.N_TOTAL_FEATURES} "
        f"kept after variance filter (NO StandardScaler — per-segment z-score)"
    )

    # =================================================================
    # Random Forest — same hyperparameters as Part I
    # =================================================================
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

    # Free memory between folds
    del F_train, y_train
    return fold_results


# =====================================================================
# Full LOOCV orchestrator
# =====================================================================
def run_loocv_iso() -> pd.DataFrame:
    """
    Run LOOCV across all patients with per-segment z-score normalisation.

    Returns the full per-fold results table and writes it to TABLES_PATH.
    Output filename: loocv_per_fold_iso.csv (distinct from Part I baseline).
    """
    all_results = []

    for fold_idx, test_patient in enumerate(config.PATIENTS):
        train_patients = [p for p in config.PATIENTS if p != test_patient]

        logger.info(f"\n=== Fold {fold_idx+1}/{len(config.PATIENTS)}: "
                    f"holding out {test_patient} ===")

        try:
            fold_metrics = evaluate_fold_iso(test_patient, train_patients, fold_idx)
        except Exception as e:
            logger.error(f"Fold {test_patient} failed: {e}")
            continue

        for model_name, m in fold_metrics.items():
            row = {"fold": fold_idx, "test_patient": test_patient,
                   "model": model_name, **m}
            all_results.append(row)

    df = pd.DataFrame(all_results)
    per_fold_path = config.TABLES_PATH / "loocv_per_fold_iso.csv"
    df.to_csv(per_fold_path, index=False)
    logger.info(f"Per-fold ISO results written to {per_fold_path}")

    # Summary
    summary_path = config.TABLES_PATH / "loocv_summary_iso.csv"
    if not df.empty:
        summary = (
            df.groupby("model")[["accuracy", "sensitivity", "specificity", "auc"]]
              .agg(["mean", "std", "count"])
        )
        summary.to_csv(summary_path)

    print("\n" + "=" * 70)
    print("STAGE 3-ISOLATED LOOCV SUMMARY (mean +/- std across folds)")
    print("=" * 70)
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        print(f"\n{model} (n_folds = {len(sub)}):")
        for col in ["accuracy", "sensitivity", "specificity", "auc"]:
            vals = sub[col].dropna()
            if len(vals) > 0:
                print(f"  {col:>12}: {vals.mean():.3f} +/- {vals.std():.3f}")

    # Compare to Part I if available
    part1_path = config.TABLES_PATH / "loocv_per_fold.csv"
    if part1_path.exists():
        print("\n" + "=" * 70)
        print("COMPARISON: Stage 3-ISOLATED vs Part I baseline RF")
        print("=" * 70)
        df_part1 = pd.read_csv(part1_path)
        df_part1_rf = df_part1[df_part1["model"] == "RF"]
        df_iso_rf = df[df["model"] == "RF"]
        merged = df_part1_rf[["test_patient", "auc"]].merge(
            df_iso_rf[["test_patient", "auc"]],
            on="test_patient",
            suffixes=("_part1", "_iso"),
        )
        merged["delta"] = merged["auc_iso"] - merged["auc_part1"]
        print(f"\nPart I RF mean AUC: {df_part1_rf['auc'].mean():.4f}")
        print(f"Stage 3-iso  AUC:    {df_iso_rf['auc'].mean():.4f}")
        print(f"Mean delta:          {merged['delta'].mean():+.4f}")
        print(f"Patients improved:   {(merged['delta'] > 0).sum()}/{len(merged)}")
        print(f"Patients worsened:   {(merged['delta'] < 0).sum()}/{len(merged)}")

    return df

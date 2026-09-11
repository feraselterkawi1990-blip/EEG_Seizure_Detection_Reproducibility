"""
distribution_shift.py
=====================
Quantitative distribution-shift analysis for catastrophic-collapse folds
(NEW in v2).

Addresses thesis review issue #23 (mechanistic analysis of fold collapse).

For each held-out patient i, computes three distance metrics between that
patient's feature distribution and the pooled training distribution from
the other 20 patients:

  1. Maximum Mean Discrepancy (MMD) with RBF kernel
       Sample-efficient kernel-based two-sample distance.
  2. Wasserstein-1 (Earth Mover's) distance
       Per-feature 1-D distance, averaged across the 92 retained features.
  3. Mahalanobis distance to the training-set centroid
       Computed using a regularised pseudo-inverse of the training
       covariance matrix.

The hypothesis (§5.2.3 of thesis): patients with high distribution shift
are the ones that collapse catastrophically. If Spearman correlation
between any of these metrics and per-fold AUC is significantly negative,
we have a quantitative mechanism for the collapse.

Usage
-----
After cache build is complete (every patient has a .npz cache file):

    from src.distribution_shift import compute_all_shifts
    df_shift = compute_all_shifts(per_fold_results=auc_df)
    df_shift.to_csv("results/tables/distribution_shift.csv")
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

import config
from . import cache

logger = logging.getLogger(__name__)


# =====================================================================
# Distance metrics
# =====================================================================
def mmd_rbf(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: float = None,
    n_subsample: int = 2000,
    random_state: int = 42,
) -> float:
    """
    Unbiased Maximum Mean Discrepancy with RBF kernel between samples X and Y.

    For tractability we subsample each set to n_subsample (matching by class
    is not enforced here — caller should pass already-stratified samples).

    Parameters
    ----------
    X : (n, d) source samples.
    Y : (m, d) target samples.
    gamma : RBF bandwidth. If None, set by median heuristic on combined sample.
    n_subsample : maximum samples per group to keep MMD tractable.
    """
    rng = np.random.default_rng(random_state)
    if len(X) > n_subsample:
        X = X[rng.choice(len(X), n_subsample, replace=False)]
    if len(Y) > n_subsample:
        Y = Y[rng.choice(len(Y), n_subsample, replace=False)]

    if gamma is None:
        # Median-heuristic on a small slice of the combined sample
        Z = np.vstack([X, Y])
        m = min(500, len(Z))
        Z_s = Z[rng.choice(len(Z), m, replace=False)]
        # Pairwise sq distances
        sq = np.sum((Z_s[:, None, :] - Z_s[None, :, :]) ** 2, axis=-1)
        med = np.median(sq[sq > 0]) if (sq > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-12)

    def _k(A, B):
        sq = np.sum((A[:, None, :] - B[None, :, :]) ** 2, axis=-1)
        return np.exp(-gamma * sq)

    Kxx = _k(X, X)
    Kyy = _k(Y, Y)
    Kxy = _k(X, Y)
    n, m = len(X), len(Y)
    # Unbiased estimator (Gretton et al. 2012)
    sum_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    sum_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    sum_xy = Kxy.sum() / (n * m)
    return float(sum_xx + sum_yy - 2 * sum_xy)


def wasserstein_avg(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Mean per-feature Wasserstein-1 distance.

    Each of the d features is treated as a 1-D distribution; the
    Wasserstein-1 distance between the marginals is computed and averaged.
    """
    d = X.shape[1]
    dists = np.zeros(d, dtype=float)
    for j in range(d):
        dists[j] = stats.wasserstein_distance(X[:, j], Y[:, j])
    return float(dists.mean())


def mahalanobis_to_centroid(
    X: np.ndarray,
    train_X: np.ndarray,
    ridge: float = 1e-3,
) -> float:
    """
    Mahalanobis distance from the centroid of `X` to the centroid of `train_X`,
    using the regularised covariance of `train_X`.
    """
    mu_train = train_X.mean(axis=0)
    mu_test = X.mean(axis=0)
    cov = np.cov(train_X, rowvar=False)
    # Regularise to ensure invertibility
    cov_reg = cov + ridge * np.eye(cov.shape[0])
    try:
        inv = np.linalg.pinv(cov_reg)
    except np.linalg.LinAlgError:
        return float("nan")
    diff = mu_test - mu_train
    d2 = float(diff @ inv @ diff)
    return float(np.sqrt(max(d2, 0.0)))


# =====================================================================
# Per-patient shift computation
# =====================================================================
def _preprocess_pair(
    F_train: np.ndarray, F_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the same preprocessing as loocv.py (variance filter + standardise),
    fit on train, applied to test.
    """
    F_train = np.nan_to_num(F_train, nan=0.0, posinf=0.0, neginf=0.0)
    F_test = np.nan_to_num(F_test, nan=0.0, posinf=0.0, neginf=0.0)
    vt = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    F_train = vt.fit_transform(F_train)
    F_test = vt.transform(F_test)
    sc = StandardScaler()
    F_train = sc.fit_transform(F_train)
    F_test = sc.transform(F_test)
    return F_train.astype(np.float32), F_test.astype(np.float32)


def compute_all_shifts(
    per_fold_results: pd.DataFrame = None,
    n_subsample_train: int = 50_000,
    n_subsample_test: int = 5_000,
) -> pd.DataFrame:
    """
    For every patient in config.PATIENTS, hold them out, pool the others,
    compute the three distribution-shift metrics, and (optionally) join
    per-fold AUCs to enable correlation analysis.

    Parameters
    ----------
    per_fold_results : optional DataFrame from run_loocv() with columns
        {test_patient, model, auc, sensitivity, ...}. If supplied, AUCs
        are joined and Spearman correlations are reported at the bottom.
    n_subsample_train : max training segments fed to MMD/Mahalanobis
        (otherwise the n^2 kernel computation is intractable).
    n_subsample_test : max test segments.

    Returns
    -------
    DataFrame with columns:
      test_patient, n_test, n_train, mmd_rbf, wasserstein_avg, mahalanobis,
      [optional] auc_RF, auc_SVM, auc_XGB, ...
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows = []

    for test_patient in config.PATIENTS:
        train_patients = [p for p in config.PATIENTS if p != test_patient]
        try:
            test_data = cache.load_patient_cache(test_patient)
        except FileNotFoundError as e:
            logger.warning(f"  {test_patient}: cache missing, skipping ({e})")
            continue
        F_test = test_data["F"]

        # Pool training (features only — saves memory)
        F_list = []
        for p in train_patients:
            try:
                d = cache.load_patient_cache(p)
                F_list.append(d["F"])
            except FileNotFoundError:
                continue
        if not F_list:
            continue
        F_train = np.concatenate(F_list, axis=0)

        # Apply same preprocessing as the LOOCV (variance + scaler)
        F_train_p, F_test_p = _preprocess_pair(F_train, F_test)

        # Subsample for MMD tractability
        if len(F_train_p) > n_subsample_train:
            F_train_p = F_train_p[
                rng.choice(len(F_train_p), n_subsample_train, replace=False)
            ]
        if len(F_test_p) > n_subsample_test:
            F_test_p = F_test_p[
                rng.choice(len(F_test_p), n_subsample_test, replace=False)
            ]

        logger.info(f"  {test_patient}: computing shifts on "
                    f"{len(F_train_p)} train / {len(F_test_p)} test")

        mmd = mmd_rbf(F_train_p, F_test_p, n_subsample=2000)
        wd = wasserstein_avg(F_train_p, F_test_p)
        md = mahalanobis_to_centroid(F_test_p, F_train_p)

        rows.append({
            "test_patient": test_patient,
            "n_test": len(F_test_p),
            "n_train": len(F_train_p),
            "mmd_rbf": mmd,
            "wasserstein_avg": wd,
            "mahalanobis": md,
        })

    df = pd.DataFrame(rows)

    # Optional: join per-fold AUCs and compute Spearman correlations
    if per_fold_results is not None and not per_fold_results.empty:
        for model in sorted(per_fold_results["model"].unique()):
            sub = per_fold_results[per_fold_results["model"] == model]
            df = df.merge(
                sub[["test_patient", "auc"]].rename(columns={"auc": f"auc_{model}"}),
                on="test_patient",
                how="left",
            )
        # Print Spearman correlations at the end
        print("\n" + "=" * 70)
        print("DISTRIBUTION SHIFT vs AUC (Spearman rank correlation)")
        print("=" * 70)
        for model in sorted(per_fold_results["model"].unique()):
            col = f"auc_{model}"
            if col not in df.columns:
                continue
            for shift_metric in ["mmd_rbf", "wasserstein_avg", "mahalanobis"]:
                paired = df[[col, shift_metric]].dropna()
                if len(paired) < 5:
                    continue
                rho, p = stats.spearmanr(paired[shift_metric], paired[col])
                interpretation = (
                    "[*] significant" if p < 0.05 else "  ns "
                )
                print(f"  {model:>4} AUC vs {shift_metric:<18}: "
                      f"rho={rho:+.3f}  p={p:.4f}  {interpretation}")
    return df

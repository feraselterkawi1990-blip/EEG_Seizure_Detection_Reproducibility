"""
metrics.py
==========
Performance metrics: accuracy, sensitivity, specificity, AUC,
confusion matrix, Wilson 95% confidence intervals.
"""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson 95% confidence interval for a binomial proportion.

    More accurate than normal approximation, especially for small n.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (center - half, center + half)


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_score: np.ndarray = None) -> Dict[str, float]:
    """
    Compute classification metrics for binary seizure detection.

    Returns dict with: accuracy, sensitivity, specificity, AUC,
    TP, TN, FP, FN, accuracy_ci_low, accuracy_ci_high.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = accuracy_score(y_true, y_pred)

    auc = np.nan
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = np.nan

    n = len(y_true)
    correct = int((y_true == y_pred).sum())
    ci_low, ci_high = wilson_interval(correct, n)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "n_test": n,
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
    }


def compute_roc(y_true: np.ndarray, y_score: np.ndarray):
    """Return (fpr, tpr, thresholds) for ROC plotting."""
    return roc_curve(y_true, y_score)

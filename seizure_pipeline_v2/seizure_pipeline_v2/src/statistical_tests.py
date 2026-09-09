"""
statistical_tests.py
====================
Statistical comparison utilities for LOOCV results (NEW in v2).

Addresses thesis review issues:
  #21 Wilcoxon signed-rank test for paired model comparisons.
  #22 Bootstrap 95% confidence intervals for aggregate metrics.

The thesis (§5.1) currently mentions a "paired t-test" for AUC differences,
but with 21 paired AUCs whose distribution is non-Gaussian (heavy-tailed
because of catastrophic-collapse folds), Wilcoxon signed-rank is the
appropriate non-parametric alternative. This module computes Wilcoxon
across all model pairs with Holm-Bonferroni correction and reports an
effect size (matched-pairs rank biserial correlation).

Bootstrap CIs on the aggregate AUC mean (§4.1, Table 4.1) are computed
by resampling the n_folds per-fold AUCs with replacement.

Usage
-----
After run_loocv() has produced loocv_per_fold.csv:

    from src.statistical_tests import (
        wilcoxon_paired_models,
        bootstrap_ci_per_model,
        run_all_tests,
    )
    df = pd.read_csv("results/tables/loocv_per_fold.csv")
    results = run_all_tests(df)
    print(results)

Or invoke as a script:

    python scripts/run_statistical_tests.py
"""

import logging
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# =====================================================================
# Wilcoxon signed-rank
# =====================================================================
def wilcoxon_paired_models(
    df: pd.DataFrame,
    metric: str = "auc",
    correction: str = "holm",
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests across models on `metric`.

    For each pair of models (a, b), pair their per-fold values by
    test_patient and compute the Wilcoxon statistic on (a - b).

    Parameters
    ----------
    df : per-fold results, columns must include {test_patient, model, metric}.
    metric : column to compare (default "auc").
    correction : multiple-testing correction, "holm" or "bonferroni" or None.

    Returns
    -------
    DataFrame with columns:
      model_a, model_b, n_pairs, median_diff,
      wilcoxon_W, p_raw, p_corrected, effect_size_rrb,
      significant_05, interpretation
    """
    models = sorted(df["model"].unique())
    pivot = df.pivot_table(index="test_patient", columns="model", values=metric)

    rows = []
    for a, b in combinations(models, 2):
        if a not in pivot.columns or b not in pivot.columns:
            continue
        paired = pivot[[a, b]].dropna()
        if len(paired) < 5:
            logger.warning(f"  Skipping {a} vs {b}: only {len(paired)} pairs")
            continue
        diff = paired[a].values - paired[b].values
        # Wilcoxon raises if all diffs are zero; handle gracefully
        if np.allclose(diff, 0):
            W, p = 0.0, 1.0
        else:
            res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
            W, p = float(res.statistic), float(res.pvalue)
        # Matched-pairs rank biserial correlation as effect size:
        #   r_rb = (W_pos - W_neg) / (W_pos + W_neg)
        # Equivalent to (2*W - n*(n+1)/2) / (n*(n+1)/2) for the standard
        # signed-rank statistic.
        n = len(diff)
        T_max = n * (n + 1) / 2
        # scipy returns W = min(W_pos, W_neg) by default — recompute both.
        ranks = stats.rankdata(np.abs(diff[diff != 0]))
        signs = np.sign(diff[diff != 0])
        W_pos = float(ranks[signs > 0].sum())
        W_neg = float(ranks[signs < 0].sum())
        rrb = (W_pos - W_neg) / (W_pos + W_neg) if (W_pos + W_neg) > 0 else 0.0

        rows.append({
            "model_a": a,
            "model_b": b,
            "n_pairs": n,
            "median_diff": float(np.median(diff)),
            "wilcoxon_W": W,
            "p_raw": p,
            "effect_size_rrb": rrb,
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    # Multiple-testing correction
    p_raw = out["p_raw"].values
    if correction == "holm":
        p_corr = _holm_correction(p_raw)
    elif correction == "bonferroni":
        p_corr = np.minimum(p_raw * len(p_raw), 1.0)
    else:
        p_corr = p_raw
    out["p_corrected"] = p_corr
    out["significant_05"] = out["p_corrected"] < 0.05

    def _interpret(row):
        if not row["significant_05"]:
            return "no significant difference"
        winner = row["model_a"] if row["median_diff"] > 0 else row["model_b"]
        loser = row["model_b"] if row["median_diff"] > 0 else row["model_a"]
        return f"{winner} > {loser}"
    out["interpretation"] = out.apply(_interpret, axis=1)

    return out.sort_values("p_corrected").reset_index(drop=True)


def _holm_correction(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni step-down correction."""
    n = len(p_values)
    order = np.argsort(p_values)
    p_sorted = p_values[order]
    p_adj_sorted = np.minimum.accumulate(p_sorted * (n - np.arange(n)))
    p_adj_sorted = np.minimum(p_adj_sorted, 1.0)
    p_adj = np.empty_like(p_adj_sorted)
    p_adj[order] = p_adj_sorted
    return p_adj


# =====================================================================
# Bootstrap CIs on aggregate metrics
# =====================================================================
def bootstrap_ci(
    values: np.ndarray,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Percentile bootstrap CI on the mean of `values`.

    Returns (mean, ci_low, ci_high).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = values[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo = float(np.quantile(boot_means, alpha))
    hi = float(np.quantile(boot_means, 1 - alpha))
    return (float(values.mean()), lo, hi)


def bootstrap_ci_per_model(
    df: pd.DataFrame,
    metric: str = "auc",
    n_resamples: int = 10_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Compute bootstrap CI on `metric` mean for each model in df.
    """
    rows = []
    for model in sorted(df["model"].unique()):
        vals = df.loc[df["model"] == model, metric].dropna().values
        mean, lo, hi = bootstrap_ci(vals, n_resamples=n_resamples,
                                     confidence=confidence)
        rows.append({
            "model": model,
            "metric": metric,
            "n_folds": len(vals),
            "mean": mean,
            "std": float(vals.std()) if len(vals) > 0 else np.nan,
            "ci_low": lo,
            "ci_high": hi,
            "ci_label": f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
                        if not np.isnan(mean) else "—",
        })
    return pd.DataFrame(rows)


# =====================================================================
# Top-level convenience
# =====================================================================
def run_all_tests(
    df: pd.DataFrame,
    metrics_to_test: List[str] = ("auc", "sensitivity", "specificity"),
    save_dir: "pathlib.Path" = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run all statistical tests and return a dict of result tables.

    If save_dir is provided, writes each table as CSV.
    """
    out = {}

    # Bootstrap CIs
    for m in metrics_to_test:
        ci_table = bootstrap_ci_per_model(df, metric=m)
        out[f"bootstrap_ci_{m}"] = ci_table
        if save_dir is not None:
            ci_table.to_csv(save_dir / f"bootstrap_ci_{m}.csv", index=False)

    # Wilcoxon (only on AUC, the headline metric)
    wilcox_auc = wilcoxon_paired_models(df, metric="auc", correction="holm")
    out["wilcoxon_auc"] = wilcox_auc
    if save_dir is not None and not wilcox_auc.empty:
        wilcox_auc.to_csv(save_dir / "wilcoxon_auc_holm.csv", index=False)

    # Pretty-print
    print("\n" + "=" * 70)
    print("BOOTSTRAP 95% CI ON AGGREGATE AUC")
    print("=" * 70)
    print(out["bootstrap_ci_auc"].to_string(index=False))

    print("\n" + "=" * 70)
    print("WILCOXON SIGNED-RANK on AUC (Holm-corrected)")
    print("=" * 70)
    if wilcox_auc.empty:
        print("(no model pairs available)")
    else:
        print(wilcox_auc.to_string(index=False))

    return out

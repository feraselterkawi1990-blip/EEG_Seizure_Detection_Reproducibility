"""
Analyse DL LOOCV results and compare to thesis classical-ML baselines.

Produces:
- summary table (mean ± SD per model)
- per-patient comparison table
- statistical tests (Wilcoxon Holm-corrected)
- figures: boxplots, per-patient bars, universal-collapse focus
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PATIENTS, RESULTS_DIR,
    UNIVERSAL_COLLAPSE, CALIBRATION_COLLAPSE, HEALTHY, BORDERLINE,
)


# ============================================================
# Thesis baselines (Table 4.1) — for comparison
# ============================================================
THESIS_BASELINES = {
    "RF":  {"auc": 0.575, "auc_ci": (0.509, 0.631), "sens": 0.137, "spec": 0.912, "mcc": 0.064},
    "SVM": {"auc": 0.540, "auc_ci": (0.474, 0.605), "sens": 0.259, "spec": 0.794, "mcc": 0.050},
    "XGB": {"auc": 0.503, "auc_ci": (0.440, 0.560), "sens": 0.170, "spec": 0.899, "mcc": 0.082},
}

# Per-patient AUC from thesis Table 4.2
THESIS_PER_PATIENT_RF_AUC = {
    "chb01": 0.721, "chb02": 0.720, "chb03": 0.568, "chb04": 0.609, "chb05": 0.673,
    "chb06": 0.402, "chb07": 0.547, "chb08": 0.630, "chb09": 0.587, "chb10": 0.762,
    "chb11": 0.154, "chb12": 0.501, "chb13": 0.425, "chb14": 0.555, "chb15": 0.598,
    "chb16": 0.355, "chb17": 0.576, "chb18": 0.695, "chb19": 0.553, "chb20": 0.685,
    "chb22": 0.749,
}


def bootstrap_ci(values, n_iter=10000, alpha=0.05, seed=42):
    """95% bootstrap CI for the mean."""
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    boots = np.array([values[rng.randint(0, n, size=n)].mean() for _ in range(n_iter)])
    return float(np.percentile(boots, 100 * alpha / 2)), \
           float(np.percentile(boots, 100 * (1 - alpha / 2)))


def wilcoxon_holm(x, y):
    """Wilcoxon signed-rank with effect size, returns (p, W, r_rb)."""
    from scipy.stats import wilcoxon
    res = wilcoxon(x, y, alternative="two-sided", zero_method="wilcox")
    diffs = np.array(x) - np.array(y)
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    pos = (nonzero > 0).sum()
    neg = (nonzero < 0).sum()
    r_rb = (pos - neg) / max(n, 1)
    return float(res.pvalue), float(res.statistic), float(r_rb)


def holm_correct(p_values):
    """Holm-Bonferroni correction."""
    p = np.asarray(p_values)
    order = np.argsort(p)
    n = len(p)
    corrected = np.empty_like(p)
    running_max = 0.0
    for rank, idx in enumerate(order):
        c = min(p[idx] * (n - rank), 1.0)
        running_max = max(running_max, c)
        corrected[idx] = running_max
    return corrected


def load_dl_results():
    """Load all per_fold_*.csv files."""
    out = {}
    for model in ["eegnet", "shallowconvnet", "cnnlstm"]:
        path = RESULTS_DIR / f"per_fold_{model}.csv"
        if path.exists():
            out[model] = pd.read_csv(path)
            print(f"  Loaded {model}: {len(out[model])} folds")
        else:
            print(f"  [skip] {model}: not found")
    return out


def make_summary_table(dl_results):
    """Mean ± SD with 95% bootstrap CI per model."""
    rows = []
    # DL models
    for name, df in dl_results.items():
        for metric in ["accuracy", "sensitivity", "specificity", "auc", "mcc", "f1"]:
            vals = df[metric].values
            mean = vals.mean()
            sd = vals.std()
            lo, hi = bootstrap_ci(vals)
            rows.append({
                "model": name, "metric": metric,
                "mean": mean, "sd": sd, "ci_low": lo, "ci_high": hi,
            })
    # Thesis classical (point estimates only)
    for name, b in THESIS_BASELINES.items():
        rows.append({
            "model": name + "_thesis", "metric": "auc",
            "mean": b["auc"], "sd": np.nan,
            "ci_low": b["auc_ci"][0], "ci_high": b["auc_ci"][1],
        })

    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "summary.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return df


def statistical_comparison(dl_results):
    """Pairwise Wilcoxon between every (DL, classical) and (DL, DL)."""
    # Build per-patient AUC dict for everything
    per_patient = {}
    for name, df in dl_results.items():
        per_patient[name] = {row["test_patient"]: row["auc"] for _, row in df.iterrows()}
    per_patient["RF_thesis"] = THESIS_PER_PATIENT_RF_AUC

    # Build pairwise tests (only on patients present in both)
    pairs = []
    names = list(per_patient.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            common = sorted(set(per_patient[a]) & set(per_patient[b]))
            if len(common) < 5:
                continue
            x = [per_patient[a][p] for p in common]
            y = [per_patient[b][p] for p in common]
            try:
                p, w, r = wilcoxon_holm(x, y)
                pairs.append({
                    "comparison": f"{a} vs {b}",
                    "n": len(common),
                    "median_diff": float(np.median(np.array(x) - np.array(y))),
                    "W": w,
                    "p_raw": p,
                    "effect_r_rb": r,
                })
            except ValueError:
                pass

    df = pd.DataFrame(pairs)
    df["p_holm"] = holm_correct(df["p_raw"].values)
    out = RESULTS_DIR / "statistical_comparison.csv"
    df.to_csv(out, index=False)
    print(f"  Wrote {out}")
    return df


def plot_boxplots(dl_results):
    """AUC box-plots: DL models alongside thesis RF."""
    data = []
    labels = []
    for name in ["eegnet", "shallowconvnet", "cnnlstm"]:
        if name in dl_results:
            data.append(dl_results[name]["auc"].values)
            labels.append(name.upper())
    # Thesis RF — reconstruct from the per-patient dict
    rf_auc = list(THESIS_PER_PATIENT_RF_AUC.values())
    data.append(rf_auc)
    labels.append("RF (thesis)")

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#999999"]
    for patch, c in zip(bp["boxes"], colors[:len(bp["boxes"])]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_ylabel("AUC (per-patient LOOCV)")
    ax.set_title("Cross-patient AUC: deep learning vs thesis classical baseline")
    ax.set_ylim(0, 1)
    ax.legend()

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "dl_vs_classical_boxplots.png", dpi=200)
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'dl_vs_classical_boxplots.png'}")


def plot_universal_collapse_focus(dl_results):
    """Did DL rescue the universal-collapse subset?"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    bar_width = 0.2
    x = np.arange(len(UNIVERSAL_COLLAPSE))

    for ax_idx, model in enumerate(["eegnet", "shallowconvnet", "cnnlstm"]):
        ax = axes[ax_idx]
        if model not in dl_results:
            ax.set_title(f"{model.upper()} (no data)")
            continue
        df = dl_results[model]
        per_pat = {row["test_patient"]: row["auc"] for _, row in df.iterrows()}
        dl_aucs = [per_pat.get(p, np.nan) for p in UNIVERSAL_COLLAPSE]
        rf_aucs = [THESIS_PER_PATIENT_RF_AUC.get(p, np.nan) for p in UNIVERSAL_COLLAPSE]

        ax.bar(x - bar_width / 2, rf_aucs, bar_width, label="RF (thesis)", color="#999999")
        ax.bar(x + bar_width / 2, dl_aucs, bar_width, label=model.upper(), color="#4C72B0")
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(UNIVERSAL_COLLAPSE, rotation=45)
        ax.set_title(model.upper())
        ax.set_ylim(0, 1)
        if ax_idx == 0:
            ax.set_ylabel("AUC")
            ax.legend()

    fig.suptitle("Universal-collapse subset: did deep learning rescue these patients?")
    fig.tight_layout()
    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    fig.savefig(fig_dir / "universal_collapse_dl_check.png", dpi=200)
    plt.close(fig)
    print(f"  Wrote {fig_dir / 'universal_collapse_dl_check.png'}")


def main():
    print("Loading DL results...")
    dl_results = load_dl_results()
    if not dl_results:
        print("No DL results found. Run run_dl_loocv.py first.")
        return

    print("\nSummary table...")
    make_summary_table(dl_results)

    print("\nStatistical comparison (Wilcoxon Holm)...")
    df_stat = statistical_comparison(dl_results)
    print(df_stat.to_string(index=False))

    print("\nFigures...")
    plot_boxplots(dl_results)
    plot_universal_collapse_focus(dl_results)

    print("\nDone.")


if __name__ == "__main__":
    main()

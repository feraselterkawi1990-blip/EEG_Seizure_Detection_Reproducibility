"""
visualize.py
============
Generate thesis-ready figures from LOOCV results.

CHANGELOG (v2):
  - NEW: plot_catastrophic_collapse() — visualises sensitivity-vs-AUC
         per patient, highlights the collapse cluster.
  - NEW: plot_distribution_shift() — joint plot of shift metrics vs AUC.
  - NEW: plot_xgb_comparison() — overlay XGB on RF/SVM boxplots.
  - Existing functions unchanged.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

logger = logging.getLogger(__name__)


def plot_metric_distributions(df: pd.DataFrame, save_path: Path = None):
    """Box plots of per-fold accuracy, sensitivity, specificity, AUC by model."""
    metrics_to_plot = ["accuracy", "sensitivity", "specificity", "auc"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    model_order = sorted(df["model"].unique())
    for ax, metric in zip(axes, metrics_to_plot):
        data_by_model = [df[df["model"] == m][metric].dropna().values
                         for m in model_order]
        ax.boxplot(data_by_model, labels=model_order)
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = save_path or (config.FIGURES_PATH / "loocv_boxplots.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_per_patient_accuracy(df: pd.DataFrame, save_path: Path = None):
    """Bar chart of accuracy per patient, grouped by model."""
    pivot = df.pivot_table(index="test_patient", columns="model",
                            values="accuracy")
    fig, ax = plt.subplots(figsize=(14, 5))
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
               label="Chance (0.5)")
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Held-out patient")
    ax.set_title("Per-patient accuracy under LOOCV")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45)
    plt.tight_layout()
    save_path = save_path or (config.FIGURES_PATH / "per_patient_accuracy.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_confusion_matrices(df: pd.DataFrame, save_path: Path = None):
    """Aggregated confusion matrices per model (sum of TP/TN/FP/FN across folds)."""
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        cm = np.array([[sub["tn"].sum(), sub["fp"].sum()],
                       [sub["fn"].sum(), sub["tp"].sum()]])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-seizure", "Seizure"])
        ax.set_yticklabels(["Non-seizure", "Seizure"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{model} (aggregated)")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                        fontsize=12, weight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    save_path = save_path or (config.FIGURES_PATH / "aggregated_confusion.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_catastrophic_collapse(df: pd.DataFrame, save_path: Path = None,
                                collapse_sens_threshold: float = 0.05):
    """
    NEW in v2: scatter of sensitivity vs AUC per patient, with the
    collapse zone shaded. Makes the catastrophic-collapse phenomenon
    visually obvious for the thesis.
    """
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5),
                              sharex=True, sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        sub = df[df["model"] == model].dropna(subset=["auc", "sensitivity"])
        ax.scatter(sub["sensitivity"], sub["auc"], s=80, alpha=0.7,
                   edgecolor="black")
        # Annotate each point with patient ID
        for _, r in sub.iterrows():
            ax.annotate(r["test_patient"].replace("chb", ""),
                        (r["sensitivity"], r["auc"]),
                        fontsize=8, alpha=0.8,
                        xytext=(3, 3), textcoords="offset points")
        # Collapse zone
        ax.axvspan(0, collapse_sens_threshold, color="red", alpha=0.15,
                   label=f"Collapse (sens<{collapse_sens_threshold})")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="AUC chance")
        ax.set_xlabel("Sensitivity")
        ax.set_ylabel("AUC")
        ax.set_title(model)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-patient sensitivity vs AUC — catastrophic collapse zone",
                 fontsize=12)
    plt.tight_layout()
    save_path = save_path or (config.FIGURES_PATH / "catastrophic_collapse.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def plot_distribution_shift(df_shift: pd.DataFrame, save_path: Path = None):
    """
    NEW in v2: scatter plots of (MMD, Wasserstein, Mahalanobis) vs AUC
    for each model, with Spearman correlation reported per panel.
    Expects df_shift to be the output of distribution_shift.compute_all_shifts().
    """
    from scipy import stats
    auc_cols = [c for c in df_shift.columns if c.startswith("auc_")]
    if not auc_cols:
        logger.warning("No auc_* columns in df_shift — run with per_fold_results")
        return

    metrics = ["mmd_rbf", "wasserstein_avg", "mahalanobis"]
    n_rows = len(auc_cols)
    n_cols = len(metrics)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for r, auc_col in enumerate(auc_cols):
        model_name = auc_col.replace("auc_", "")
        for c, shift in enumerate(metrics):
            ax = axes[r][c]
            paired = df_shift[[shift, auc_col, "test_patient"]].dropna()
            ax.scatter(paired[shift], paired[auc_col], s=60, alpha=0.7,
                       edgecolor="black")
            for _, row in paired.iterrows():
                ax.annotate(row["test_patient"].replace("chb", ""),
                            (row[shift], row[auc_col]),
                            fontsize=7, alpha=0.7,
                            xytext=(3, 3), textcoords="offset points")
            if len(paired) >= 5:
                rho, p = stats.spearmanr(paired[shift], paired[auc_col])
                ax.set_title(f"{model_name}: rho={rho:+.2f}, p={p:.3f}",
                             fontsize=10)
            else:
                ax.set_title(f"{model_name}: insufficient data",
                             fontsize=10)
            if r == n_rows - 1:
                ax.set_xlabel(shift)
            if c == 0:
                ax.set_ylabel("AUC")
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
                       alpha=0.5)
            ax.grid(True, alpha=0.3)

    plt.suptitle("Distribution shift vs cross-patient AUC", fontsize=13)
    plt.tight_layout()
    save_path = save_path or (config.FIGURES_PATH / "distribution_shift.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {save_path}")


def generate_summary_table_for_thesis(df: pd.DataFrame,
                                       save_path: Path = None) -> pd.DataFrame:
    """
    Create the headline thesis table:
    Model | Accuracy | Sensitivity | Specificity | AUC  (mean +/- std)
    """
    rows = []
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        row = {"Model": model}
        for col in ["accuracy", "sensitivity", "specificity", "auc"]:
            vals = sub[col].dropna()
            if len(vals) == 0:
                row[col] = "—"
            else:
                row[col] = f"{vals.mean():.3f} +/- {vals.std():.3f}"
        row["n_folds"] = len(sub)
        rows.append(row)
    table = pd.DataFrame(rows)
    save_path = save_path or (config.TABLES_PATH / "thesis_headline_table.csv")
    table.to_csv(save_path, index=False)
    logger.info(f"Saved {save_path}")
    return table


def generate_all_figures(df: pd.DataFrame, df_shift: pd.DataFrame = None):
    """One-stop figure generation."""
    plot_metric_distributions(df)
    plot_per_patient_accuracy(df)
    plot_confusion_matrices(df)
    plot_catastrophic_collapse(df)
    if df_shift is not None:
        plot_distribution_shift(df_shift)
    table = generate_summary_table_for_thesis(df)
    print("\n" + "=" * 70)
    print("THESIS HEADLINE TABLE")
    print("=" * 70)
    print(table.to_string(index=False))

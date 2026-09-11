"""
visualize.py
============
Generate thesis-ready figures from LOOCV results:

- Figure 11 equivalent: ROC curves (mean across folds) for all four models
- Box plots of per-fold metrics
- Per-patient bar charts
- Figure 5/6/7/9 equivalents: aggregated confusion matrices
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
    for ax, metric in zip(axes, metrics_to_plot):
        data_by_model = [df[df["model"] == m][metric].dropna().values
                         for m in df["model"].unique()]
        ax.boxplot(data_by_model, labels=df["model"].unique())
        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1.05)
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
    models = df["model"].unique()
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


def generate_summary_table_for_thesis(df: pd.DataFrame,
                                       save_path: Path = None) -> pd.DataFrame:
    """
    Create the headline thesis table:
    Model | Accuracy | Sensitivity | Specificity | AUC  (mean ± std)
    """
    rows = []
    for model in df["model"].unique():
        sub = df[df["model"] == model]
        row = {"Model": model}
        for col in ["accuracy", "sensitivity", "specificity", "auc"]:
            vals = sub[col].dropna()
            if len(vals) == 0:
                row[col] = "—"
            else:
                row[col] = f"{vals.mean():.3f} ± {vals.std():.3f}"
        row["n_folds"] = len(sub)
        rows.append(row)
    table = pd.DataFrame(rows)
    save_path = save_path or (config.TABLES_PATH / "thesis_headline_table.csv")
    table.to_csv(save_path, index=False)
    logger.info(f"Saved {save_path}")
    return table


def generate_all_figures(df: pd.DataFrame):
    """One-stop figure generation."""
    plot_metric_distributions(df)
    plot_per_patient_accuracy(df)
    plot_confusion_matrices(df)
    table = generate_summary_table_for_thesis(df)
    print("\n" + "=" * 70)
    print("THESIS HEADLINE TABLE")
    print("=" * 70)
    print(table.to_string(index=False))

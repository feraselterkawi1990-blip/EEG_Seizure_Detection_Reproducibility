"""
Cross-stage analysis — compares Stages 1, 1.5, 2, 3 to localise the
source of EEGNet's apparent advantage.

Run after each stage completes. Skips stages whose CSV is missing.

Usage:
    python scripts/analyze_stages.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import RESULTS_DIR, UNIVERSAL_COLLAPSE, CALIBRATION_COLLAPSE, HEALTHY, BORDERLINE


def bootstrap_ci(values, n_iter=10000, seed=42):
    rng = np.random.RandomState(seed)
    values = np.asarray(values)
    n = len(values)
    boots = np.array([values[rng.randint(0, n, size=n)].mean() for _ in range(n_iter)])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def summarise(name, df, metric="auc"):
    if df is None or len(df) == 0:
        return None
    vals = df[metric].dropna().values
    if len(vals) == 0:
        return None
    mean = vals.mean()
    sd = vals.std(ddof=1)
    lo, hi = bootstrap_ci(vals)
    return {
        "stage": name, "n": len(vals),
        "mean": mean, "sd": sd, "ci_low": lo, "ci_high": hi,
    }


def print_overall(stages):
    print("\n" + "=" * 80)
    print("OVERALL AUC BY STAGE")
    print("=" * 80)
    print(f"{'Stage':<35} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>22}")
    print("-" * 80)
    for s in stages:
        if s is None:
            continue
        ci = f"[{s['ci_low']:.3f}, {s['ci_high']:.3f}]"
        print(f"{s['stage']:<35} {s['n']:>4} {s['mean']:>8.3f} {s['sd']:>8.3f} {ci:>22}")


def print_by_regime(loaded):
    """For each loaded stage, print mean AUC by regime."""
    print("\n" + "=" * 80)
    print("MEAN AUC BY REGIME × STAGE")
    print("=" * 80)
    regimes = [
        ("Universal collapse", UNIVERSAL_COLLAPSE),
        ("Calibration collapse", CALIBRATION_COLLAPSE),
        ("Healthy",            HEALTHY),
        ("Borderline",         BORDERLINE),
    ]
    header = f"{'Regime':<22}"
    for stage_name in loaded.keys():
        header += f"{stage_name:>20}"
    print(header)
    print("-" * 80)
    for regime_name, patients in regimes:
        row = f"{regime_name:<22}"
        for stage_name, df in loaded.items():
            if df is None or len(df) == 0:
                row += f"{'—':>20}"
                continue
            sub = df[df["test_patient"].isin(patients)]
            if len(sub) == 0:
                row += f"{'—':>20}"
            else:
                row += f"{sub['auc'].mean():>20.3f}"
        print(row)


def print_per_patient(loaded):
    """Print per-patient AUC across all stages, side by side."""
    print("\n" + "=" * 90)
    print("PER-PATIENT AUC × STAGE")
    print("=" * 90)
    pats = sorted({p for df in loaded.values() if df is not None for p in df["test_patient"]})
    header = f"{'Patient':<8}"
    for stage_name in loaded.keys():
        header += f"{stage_name:>18}"
    print(header)
    print("-" * 90)
    for p in pats:
        row = f"{p:<8}"
        for stage_name, df in loaded.items():
            if df is None or len(df) == 0:
                row += f"{'—':>18}"
                continue
            sub = df[df["test_patient"] == p]
            if len(sub) == 0:
                row += f"{'—':>18}"
            else:
                row += f"{sub['auc'].iloc[0]:>18.3f}"
        print(row)


def interpret(loaded):
    """Generate a textual interpretation based on which stages are loaded."""
    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)

    s1 = loaded.get("Stage 1 (50Hz, per-seg)")
    s15 = loaded.get("Stage 1.5 (25Hz, per-seg)")
    s2 = loaded.get("Stage 2 (50Hz, per-rec)")
    s3 = loaded.get("Stage 3 (RF, per-seg)")

    def mean_or_none(df):
        return df["auc"].mean() if df is not None and len(df) > 0 else None

    m1 = mean_or_none(s1)
    m15 = mean_or_none(s15)
    m2 = mean_or_none(s2)
    m3 = mean_or_none(s3)
    rf_thesis = 0.575

    print(f"Reference: Thesis classical RF = {rf_thesis}")
    if m1 is not None:
        print(f"Stage 1   = {m1:.3f}  (Δ vs thesis = {m1-rf_thesis:+.3f})")
    if m15 is not None:
        print(f"Stage 1.5 = {m15:.3f}  (Δ vs Stage 1  = {m15-m1:+.3f})" if m1 is not None else f"Stage 1.5 = {m15:.3f}")
    if m2 is not None:
        print(f"Stage 2   = {m2:.3f}  (Δ vs Stage 1  = {m2-m1:+.3f})" if m1 is not None else f"Stage 2 = {m2:.3f}")
    if m3 is not None:
        print(f"Stage 3   = {m3:.3f}  (Δ vs thesis  = {m3-rf_thesis:+.3f})")

    print("\nDIAGNOSTIC:")
    if m15 is not None and m1 is not None:
        drop_emg = m1 - m15
        if drop_emg > 0.10:
            print(f"  EMG/gamma hypothesis: SUPPORTED. Stage 1.5 drops by {drop_emg:+.3f}.")
            print("  → Conclusion: a substantial part of Stage 1 AUC came from 25-50 Hz band")
            print("    (likely EMG contamination during ictal periods).")
        elif drop_emg > 0.03:
            print(f"  EMG/gamma hypothesis: PARTIAL. Stage 1.5 drops by {drop_emg:+.3f}.")
            print("  → Conclusion: EMG contributes but is not the dominant signal.")
        else:
            print(f"  EMG/gamma hypothesis: REJECTED. Stage 1.5 changes by {drop_emg:+.3f} only.")
            print("  → Conclusion: The signal is in the 0.5-25 Hz band (delta/theta/alpha/beta),")
            print("    consistent with genuine seizure morphology.")

    if m2 is not None and m1 is not None:
        drop_amp = m1 - m2
        if drop_amp > 0.15:
            print(f"\n  Amplitude-bias hypothesis: SUPPORTED. Stage 2 drops by {drop_amp:+.3f}.")
            print("  → Conclusion: Per-segment z-score preserved a discriminative")
            print("    amplitude envelope that EEGNet exploited. Per-recording z-score")
            print("    eliminates this and recovers the true cross-patient difficulty.")
        elif drop_amp > 0.05:
            print(f"\n  Amplitude-bias hypothesis: PARTIAL. Stage 2 drops by {drop_amp:+.3f}.")
            print("  → Conclusion: Amplitude contributes but is not the only discriminative cue.")
        else:
            print(f"\n  Amplitude-bias hypothesis: REJECTED. Stage 2 changes by {drop_amp:+.3f} only.")
            print("  → Conclusion: EEGNet learns a morphological pattern beyond amplitude.")

    if m3 is not None:
        rise_classical = m3 - rf_thesis
        if rise_classical > 0.10:
            print(f"\n  Methodology hypothesis: SUPPORTED. Classical+per-seg lifts by {rise_classical:+.3f}.")
            print("  → Conclusion: A substantial part of the deep-vs-classical gap comes from")
            print("    normalisation choice, not architecture. The thesis's StandardScaler-over-")
            print("    training-pool was unnecessarily strict.")
        elif rise_classical > 0.03:
            print(f"\n  Methodology hypothesis: PARTIAL. Classical+per-seg lifts by {rise_classical:+.3f}.")
            print("  → Conclusion: Per-segment normalisation helps classical features modestly,")
            print("    but a real architectural advantage remains for deep learning.")
        else:
            print(f"\n  Methodology hypothesis: REJECTED. Classical+per-seg unchanged ({rise_classical:+.3f}).")
            print("  → Conclusion: The deep-vs-classical gap is genuine architecture-level,")
            print("    not a normalisation artefact.")


def main():
    files = {
        "Stage 1 (50Hz, per-seg)":  RESULTS_DIR / "per_fold_eegnet.csv",
        "Stage 1.5 (25Hz, per-seg)": RESULTS_DIR / "per_fold_eegnet_25hz.csv",
        "Stage 2 (50Hz, per-rec)":  RESULTS_DIR / "per_fold_eegnet_perrec.csv",
        "Stage 3 (RF, per-seg)":     RESULTS_DIR / "per_fold_classical_perseg.csv",
    }

    loaded = {}
    summaries = []
    for name, path in files.items():
        if path.exists():
            df = pd.read_csv(path)
            loaded[name] = df
            print(f"  Loaded {name}: {len(df)} folds")
            summaries.append(summarise(name, df))
        else:
            loaded[name] = None
            print(f"  [skip] {name}: not yet run")

    if not loaded:
        print("\nNo stage results found. Run at least Stage 1 first.")
        return

    print_overall(summaries)
    print_by_regime(loaded)
    print_per_patient(loaded)
    interpret(loaded)

    print("\n" + "=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()

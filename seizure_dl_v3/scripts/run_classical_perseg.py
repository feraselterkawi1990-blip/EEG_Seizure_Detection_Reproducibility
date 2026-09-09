"""
Stage 3 — Classical Random Forest with per-segment z-score.

Question: is the apparent advantage of EEGNet over the thesis classical
pipeline (RF AUC = 0.575) due to (a) deep architecture or (b) the
normalisation strategy?

To control for normalisation, we extract the same kind of features the
thesis pipeline uses (or simpler equivalents) FROM THE RAW SEGMENTS THAT
HAVE BEEN PER-SEGMENT Z-SCORED, and re-run RF/SVM/XGB on those.

The simpler approach used here: extract a small feature set computable
from already-z-scored raw segments — band powers, Hjorth parameters,
line-length, statistical moments. These are then used by RF.

If RF + per-segment z-score still gives AUC near 0.575, the deep gain
is real architecture. If it rises substantially toward 0.7-0.8, the
gain was largely the normalisation choice.

Outputs results/per_fold_classical_perseg.csv
"""
import sys
import csv
import time
from pathlib import Path
import numpy as np
from scipy.signal import welch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from math import sqrt

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PATIENTS, RAW_CACHE_DIR, RESULTS_DIR,
    N_CHANNELS, WINDOW_SAMPLES, TARGET_FS, RANDOM_SEED,
    NEG_TO_POS_RATIO, THRESHOLD_RANGE,
)


# Frequency bands (same as thesis classical pipeline)
BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


def per_segment_zscore_np(X):
    """Per-segment per-channel z-score on numpy array. X: (N, C, T)."""
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + 1e-6
    return (X - mean) / std


def hjorth_params(x):
    """Return (activity, mobility, complexity) for 1D signal x."""
    var0 = np.var(x)
    diff1 = np.diff(x)
    var1 = np.var(diff1)
    diff2 = np.diff(diff1)
    var2 = np.var(diff2)
    activity = var0
    mobility = np.sqrt(var1 / max(var0, 1e-12))
    complexity = np.sqrt(var2 / max(var1, 1e-12)) / max(mobility, 1e-12)
    return activity, mobility, complexity


def extract_features(X):
    """
    Extract classical features per segment, per channel.
    X: (N, C, T) → (N, F) where F = C * (5 bands + 3 Hjorth + 1 line-length + 2 moments)
    """
    n, c, t = X.shape
    fs = TARGET_FS
    n_features_per_ch = 5 + 3 + 1 + 2  # 11 per channel
    feats = np.zeros((n, c * n_features_per_ch), dtype=np.float32)

    for i in range(n):
        f_idx = 0
        for ch in range(c):
            sig = X[i, ch, :]

            # 1. Welch PSD → 5 band powers (relative)
            freqs, psd = welch(sig, fs=fs, nperseg=min(256, len(sig)))
            total_power = psd.sum() + 1e-12
            for (lo, hi) in BANDS.values():
                mask = (freqs >= lo) & (freqs < hi)
                feats[i, f_idx] = psd[mask].sum() / total_power
                f_idx += 1

            # 2. Hjorth parameters
            act, mob, comp = hjorth_params(sig)
            feats[i, f_idx] = act; f_idx += 1
            feats[i, f_idx] = mob; f_idx += 1
            feats[i, f_idx] = comp; f_idx += 1

            # 3. Line length
            ll = np.sum(np.abs(np.diff(sig)))
            feats[i, f_idx] = ll; f_idx += 1

            # 4. Statistical moments
            feats[i, f_idx] = np.mean(np.abs(sig)); f_idx += 1
            feats[i, f_idx] = np.std(sig); f_idx += 1

    return feats


def assemble_pool(patients, cache_dir):
    rng = np.random.RandomState(RANDOM_SEED)
    X_pos_all, X_neg_all = [], []
    for p in patients:
        npz = np.load(cache_dir / f"{p}_raw.npz")
        X_pos_all.append(npz["X_seizure"])
        X_neg_all.append(npz["X_baseline"])
    X_pos = np.concatenate(X_pos_all, axis=0)
    X_neg = np.concatenate(X_neg_all, axis=0)
    n_pos = len(X_pos)
    n_neg_target = min(n_pos * NEG_TO_POS_RATIO, len(X_neg))
    neg_idx = rng.choice(len(X_neg), size=n_neg_target, replace=False)
    X_neg = X_neg[neg_idx]
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos), dtype=np.int8),
                        np.zeros(len(X_neg), dtype=np.int8)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def assemble_test(patient, cache_dir):
    npz = np.load(cache_dir / f"{patient}_raw.npz")
    X_pos = npz["X_seizure"]
    X_neg = npz["X_baseline"]
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos), dtype=np.int8),
                        np.zeros(len(X_neg), dtype=np.int8)])
    return X, y


def youden_threshold(y_true, y_proba, candidates=THRESHOLD_RANGE):
    best_t, best_j = 0.5, -np.inf
    for t in candidates:
        y_pred = (y_proba >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        sens = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        j = sens + spec - 1
        if j > best_j:
            best_j = j
            best_t = t
    return best_t


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn
    acc = (tp + tn) / max(n, 1)
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    try:
        auc = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc = float("nan")
    mcc_denom = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / max(mcc_denom, 1.0)
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return {"accuracy": acc, "sensitivity": sens, "specificity": spec,
            "auc": auc, "mcc": mcc, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "threshold": threshold}


def run_fold(test_patient, train_patients):
    t0 = time.time()
    print(f"\n=== RF (per-seg z-score) | held-out: {test_patient} ===")
    print("  Loading training raw data...")
    X_train_raw, y_train = assemble_pool(train_patients, RAW_CACHE_DIR)
    print(f"    Train pool: {len(X_train_raw)} segments")

    print("  Per-segment z-score...")
    X_train_z = per_segment_zscore_np(X_train_raw)

    print("  Extracting features...")
    feat_t0 = time.time()
    X_train_feat = extract_features(X_train_z)
    print(f"    Train features: {X_train_feat.shape} in {time.time()-feat_t0:.1f}s")

    # Validation split (last 10%)
    n_val = max(int(len(X_train_feat) * 0.10), 1)
    X_val_feat = X_train_feat[-n_val:]
    y_val = y_train[-n_val:]
    X_train_feat = X_train_feat[:-n_val]
    y_train = y_train[:-n_val]

    print(f"  Training RF (300 trees)...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        n_jobs=-1, random_state=RANDOM_SEED, class_weight="balanced",
    )
    fit_t0 = time.time()
    rf.fit(X_train_feat, y_train)
    fit_time = time.time() - fit_t0
    print(f"    Fit time: {fit_time:.1f}s")

    val_proba = rf.predict_proba(X_val_feat)[:, 1]
    threshold = youden_threshold(y_val, val_proba)
    print(f"    Threshold: {threshold:.2f}")

    print("  Loading test raw data...")
    X_test_raw, y_test = assemble_test(test_patient, RAW_CACHE_DIR)

    print("  Per-segment z-score on test...")
    X_test_z = per_segment_zscore_np(X_test_raw)

    print("  Extracting test features...")
    X_test_feat = extract_features(X_test_z)

    test_proba = rf.predict_proba(X_test_feat)[:, 1]
    metrics = compute_metrics(y_test, test_proba, threshold)
    metrics.update({
        "test_patient": test_patient,
        "model": "rf_perseg",
        "n_test": int(len(y_test)),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "fit_time_sec": fit_time,
        "fold_total_sec": time.time() - t0,
    })
    print(f"  Done in {metrics['fold_total_sec']:.1f}s | "
          f"AUC={metrics['auc']:.3f} MCC={metrics['mcc']:+.3f} "
          f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f}")
    return metrics


def main():
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    out_csv = RESULTS_DIR / "per_fold_classical_perseg.csv"
    fieldnames = [
        "test_patient", "model", "accuracy", "sensitivity", "specificity",
        "auc", "mcc", "f1", "tp", "tn", "fp", "fn",
        "threshold", "n_test", "n_pos_test", "n_neg_test",
        "fit_time_sec", "fold_total_sec",
    ]

    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for test_patient in PATIENTS:
            train_patients = [p for p in PATIENTS if p != test_patient]
            try:
                metrics = run_fold(test_patient, train_patients)
                row = {k: metrics.get(k) for k in fieldnames}
                writer.writerow(row)
                f.flush()
            except Exception as e:
                print(f"  [ERROR] {test_patient}: {e}")
                import traceback; traceback.print_exc()

    print(f"\n=== Done. Results: {out_csv} ===")


if __name__ == "__main__":
    main()

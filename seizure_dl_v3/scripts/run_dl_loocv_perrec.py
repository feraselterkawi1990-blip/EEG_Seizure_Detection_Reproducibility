"""
Stage 2 LOOCV runner — per-recording (global) z-score normalisation.

Same data (50 Hz cache from Stage 1), but normalisation strategy changes:
- Stage 1: each segment z-scored independently (preserves amplitude differences
  between ictal/interictal as a discriminative cue)
- Stage 2: z-score statistics computed across the ENTIRE training pool, then
  applied uniformly to all segments (training and test). This destroys
  the inter-segment amplitude variability and is the stricter protocol used
  in Saab et al. 2020.

If Stage 2 AUC drops substantially below Stage 1, the EEGNet advantage
on Stage 1 was largely amplitude-driven.

Outputs to results/per_fold_eegnet_perrec.csv
"""
import argparse
import csv
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    PATIENTS, RAW_CACHE_DIR, RESULTS_DIR, LOG_DIR,
    N_CHANNELS, WINDOW_SAMPLES, RANDOM_SEED,
    EEGNET_CONFIG, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, VAL_SPLIT_RATIO,
    THRESHOLD_RANGE, NEG_TO_POS_RATIO,
)
from models import build_model, count_parameters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = (DEVICE == "cuda")
NUM_WORKERS = 0


# ============================================================
# DATASET — Stage 2 uses GLOBAL z-score, not per-segment
# ============================================================
class EEGSegmentDatasetGlobal(Dataset):
    """Per-recording z-score: one mean/std for the entire pool, applied uniformly."""

    def __init__(self, X, y, mean=None, std=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).long()

        if mean is None or std is None:
            # Compute statistics on this dataset's pool (training)
            self.mean = X.mean(dim=(0, 2), keepdim=True)  # (1, C, 1)
            self.std = X.std(dim=(0, 2), keepdim=True) + 1e-6  # (1, C, 1)
        else:
            self.mean = mean
            self.std = std

        self.X = (X - self.mean) / self.std
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def assemble_train_pool(train_patients, cache_dir):
    rng = np.random.RandomState(RANDOM_SEED)
    X_pos_all, X_neg_all = [], []
    for p in train_patients:
        npz = np.load(cache_dir / f"{p}_raw.npz")
        X_pos_all.append(npz["X_seizure"])
        X_neg_all.append(npz["X_baseline"])
    X_pos = np.concatenate(X_pos_all, axis=0) if X_pos_all else np.zeros((0,))
    X_neg = np.concatenate(X_neg_all, axis=0)
    n_pos = len(X_pos)
    n_neg_target = min(n_pos * NEG_TO_POS_RATIO, len(X_neg))
    neg_idx = rng.choice(len(X_neg), size=n_neg_target, replace=False)
    X_neg = X_neg[neg_idx]
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([
        np.ones(len(X_pos), dtype=np.int8),
        np.zeros(len(X_neg), dtype=np.int8),
    ])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def assemble_test(patient, cache_dir):
    npz = np.load(cache_dir / f"{patient}_raw.npz")
    X_pos = npz["X_seizure"]
    X_neg = npz["X_baseline"]
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([
        np.ones(len(X_pos), dtype=np.int8),
        np.zeros(len(X_neg), dtype=np.int8),
    ])
    return X, y


def split_train_val(X, y, val_ratio=VAL_SPLIT_RATIO):
    rng = np.random.RandomState(RANDOM_SEED)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    n_pos_val = max(int(len(pos_idx) * val_ratio), 1)
    n_neg_val = max(int(len(neg_idx) * val_ratio), 1)
    val_idx = np.concatenate([pos_idx[:n_pos_val], neg_idx[:n_neg_val]])
    train_idx = np.concatenate([pos_idx[n_pos_val:], neg_idx[n_neg_val:]])
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def train_one_fold_global(model, X_tr, y_tr, X_val, y_val):
    """Train using GLOBAL z-score: stats from training set, applied to validation."""
    train_ds = EEGSegmentDatasetGlobal(X_tr, y_tr)
    # Validation uses training stats
    val_ds = EEGSegmentDatasetGlobal(X_val, y_val, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = model.to(DEVICE)
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience_left = EARLY_STOPPING_PATIENCE
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss_total, n_batches = 0.0, 0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.float().to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item()
            n_batches += 1
        avg_train = train_loss_total / max(n_batches, 1)

        model.eval()
        val_loss_total, n_val_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE, non_blocking=True)
                yb = yb.float().to(DEVICE, non_blocking=True)
                logits = model(xb).squeeze(-1)
                loss = criterion(logits, yb)
                val_loss_total += loss.item()
                n_val_batches += 1
        avg_val = val_loss_total / max(n_val_batches, 1)
        history.append({"epoch": epoch, "train_loss": avg_train, "val_loss": avg_val})

        if avg_val < best_val_loss - 1e-4:
            best_val_loss = avg_val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = EARLY_STOPPING_PATIENCE
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, train_ds.mean, train_ds.std


def predict_proba_global(model, X, mean, std):
    model.eval()
    ds = EEGSegmentDatasetGlobal(X, np.zeros(len(X), dtype=np.int8), mean=mean, std=std)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


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
    from math import sqrt
    from sklearn.metrics import roc_auc_score
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


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_fold(test_patient, train_patients):
    t0 = time.time()
    print(f"\n=== EEGNET (global z-score) | held-out: {test_patient} ===")
    print("  Loading training data...")
    X_pool, y_pool = assemble_train_pool(train_patients, RAW_CACHE_DIR)
    print(f"    Train pool: {len(X_pool)} segments, "
          f"{(y_pool==1).sum()} pos, {(y_pool==0).sum()} neg")
    X_tr, y_tr, X_val, y_val = split_train_val(X_pool, y_pool)
    print(f"    Train: {len(X_tr)} | Val: {len(X_val)}")

    model = build_model("eegnet", N_CHANNELS, WINDOW_SAMPLES, **EEGNET_CONFIG)
    n_params = count_parameters(model)
    print(f"    Model: eegnet ({n_params:,} params)")

    print("  Training (global z-score)...")
    fit_t0 = time.time()
    model, history, mean, std = train_one_fold_global(model, X_tr, y_tr, X_val, y_val)
    fit_time = time.time() - fit_t0
    print(f"    Fit time: {fit_time:.1f}s ({len(history)} epochs)")

    val_proba = predict_proba_global(model, X_val, mean, std)
    threshold = youden_threshold(y_val, val_proba)
    print(f"    Threshold: {threshold:.2f}")

    print("  Evaluating (global stats applied to test)...")
    X_test, y_test = assemble_test(test_patient, RAW_CACHE_DIR)
    test_proba = predict_proba_global(model, X_test, mean, std)
    metrics = compute_metrics(y_test, test_proba, threshold)
    metrics.update({
        "test_patient": test_patient,
        "model": "eegnet_perrec",
        "n_test": int(len(y_test)),
        "n_pos_test": int((y_test == 1).sum()),
        "n_neg_test": int((y_test == 0).sum()),
        "fit_time_sec": fit_time,
        "n_params": n_params,
        "fold_total_sec": time.time() - t0,
    })
    print(f"  Done in {metrics['fold_total_sec']:.1f}s | "
          f"AUC={metrics['auc']:.3f} MCC={metrics['mcc']:+.3f} "
          f"Sens={metrics['sensitivity']:.3f} Spec={metrics['specificity']:.3f}")
    return metrics


def main():
    set_seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    LOG_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    out_csv = RESULTS_DIR / "per_fold_eegnet_perrec.csv"
    fieldnames = [
        "test_patient", "model", "accuracy", "sensitivity", "specificity",
        "auc", "mcc", "f1", "tp", "tn", "fp", "fn",
        "threshold", "n_test", "n_pos_test", "n_neg_test",
        "fit_time_sec", "fold_total_sec", "n_params",
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
                import traceback
                traceback.print_exc()

    print(f"\n=== Done. Results saved to {out_csv} ===")


if __name__ == "__main__":
    main()

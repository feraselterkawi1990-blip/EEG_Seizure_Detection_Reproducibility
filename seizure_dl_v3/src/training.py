"""
Training utilities — FIXED with per-segment z-score normalisation.

Critical fix from v1:
- Normalisation is now per-segment (each window standardised individually)
- This prevents the model from learning trivial amplitude differences
  between ictal and interictal segments (which generalise across patients
  for non-physiological reasons).
- This is the standard normalisation used in the seminal EEG-DL papers
  (Schirrmeister 2017, Lawhern 2018).
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    MAX_EPOCHS, EARLY_STOPPING_PATIENCE, VAL_SPLIT_RATIO,
    THRESHOLD_RANGE, NEG_TO_POS_RATIO, RANDOM_SEED,
    NUM_WORKERS,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = (DEVICE == "cuda")


def per_segment_zscore(X):
    """
    Z-score each (channel, segment) independently.
    X: (N, C, T) → output: (N, C, T) standardised.
    This forces the model to learn shape, not amplitude.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).float()
    # Per-segment, per-channel mean/std
    mean = X.mean(dim=2, keepdim=True)  # (N, C, 1)
    std = X.std(dim=2, keepdim=True) + 1e-6  # (N, C, 1)
    return (X - mean) / std


class EEGSegmentDataset(Dataset):
    """Stores (X, y) with per-segment z-score normalisation."""

    def __init__(self, X, y):
        self.X = per_segment_zscore(X)
        self.y = torch.from_numpy(y).long() if isinstance(y, np.ndarray) else y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def assemble_train_pool(train_patients, cache_dir):
    """Load all training patients, balance to 1:NEG_TO_POS_RATIO ratio, shuffle."""
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
    """Load held-out patient → test data with full ratio (no subsampling)."""
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
    """Stratified split for early stopping."""
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


def train_one_fold(model, X_tr, y_tr, X_val, y_val, log_path=None):
    """Train with early stopping. Returns (model, history, None, None).

    NOTE: returns None, None for mean/std because normalisation is now
    per-segment (no global statistics needed). Kept return signature for
    backward compatibility with run_dl_loocv.py.
    """
    train_ds = EEGSegmentDataset(X_tr, y_tr)
    val_ds = EEGSegmentDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    model = model.to(DEVICE)

    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32, device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

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

    return model, history, None, None


def predict_proba(model, X, mean=None, std=None, batch_size=BATCH_SIZE):
    """Return (n,) probabilities. mean/std args kept for backward compat (ignored)."""
    model.eval()
    ds = EEGSegmentDataset(X, np.zeros(len(X), dtype=np.int8))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            logits = model(xb).squeeze(-1)
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
    return np.concatenate(probs)


def youden_threshold(y_true, y_proba, candidates=THRESHOLD_RANGE):
    best_t = 0.5
    best_j = -np.inf
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

    return {
        "accuracy": acc, "sensitivity": sens, "specificity": spec,
        "auc": auc, "mcc": mcc, "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "threshold": threshold,
    }

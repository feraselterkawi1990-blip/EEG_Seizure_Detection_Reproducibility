"""Diagnostic v3 - three controls to localise the source of leakage."""
import sys
sys.path.insert(0, 'src')
sys.path.insert(0, 'scripts')

import numpy as np
import torch
from pathlib import Path

import config
config.MAX_EPOCHS = 20
config.EARLY_STOPPING_PATIENCE = 5

from config import RAW_CACHE_DIR, RANDOM_SEED, N_CHANNELS, WINDOW_SAMPLES, EEGNET_CONFIG
from models import build_model
from training import (
    DEVICE, assemble_train_pool, assemble_test, split_train_val,
    train_one_fold, predict_proba, compute_metrics, per_segment_zscore,
)

print("=" * 65)
print("DIAGNOSTIC v3 - three controls")
print("=" * 65)

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ============================================================
# Test A: Train on real data with RANDOMISED labels
# If model still hits high AUC, the leakage is structural
# ============================================================
print("\n[Test A] Train with RANDOMISED labels (control)")
X_pool, y_pool = assemble_train_pool(
    ["chb01", "chb02", "chb03", "chb04", "chb05"], RAW_CACHE_DIR
)
print(f"  Train pool: {len(X_pool)} segments")

y_random = np.random.permutation(y_pool)
print(f"  Original y[:10]: {y_pool[:10]}")
print(f"  Random   y[:10]: {y_random[:10]}")

X_tr, y_tr, X_val, y_val = split_train_val(X_pool, y_random)
model = build_model("eegnet", N_CHANNELS, WINDOW_SAMPLES, **EEGNET_CONFIG)
print(f"  Training EEGNet 20 epochs on RANDOM labels...")
model, _, _, _ = train_one_fold(model, X_tr, y_tr, X_val, y_val)

X_test, y_test = assemble_test("chb11", RAW_CACHE_DIR)
proba = predict_proba(model, X_test)
m = compute_metrics(y_test, proba, 0.5)
auc_a = m['auc']
print(f"  Random-labels AUC on chb11: {auc_a:.3f}")
print(f"  -> Should be ~0.5 if no leakage. Higher = structural problem.")

# ============================================================
# Test B: Are positives just LOUDER than negatives?
# ============================================================
print("\n[Test B] Raw amplitude of pos vs neg windows")
chb01 = np.load(RAW_CACHE_DIR / "chb01_raw.npz")
X_pos = chb01["X_seizure"]
X_neg = chb01["X_baseline"]
amp_pos = float(np.abs(X_pos).mean())
amp_neg = float(np.abs(X_neg).mean())
print(f"  chb01 pos windows: {len(X_pos)}, mean abs amp = {amp_pos:.6e}")
print(f"  chb01 neg windows: {len(X_neg)}, mean abs amp = {amp_neg:.6e}")
print(f"  Ratio pos/neg: {amp_pos / amp_neg:.2f}")

print(f"\n  After per-segment z-score:")
X_pos_z = per_segment_zscore(X_pos[:50])
X_neg_z = per_segment_zscore(X_neg[:50])
amp_pos_z = float(X_pos_z.abs().mean().item())
amp_neg_z = float(X_neg_z.abs().mean().item())
print(f"  pos z-scored mean abs: {amp_pos_z:.6f}")
print(f"  neg z-scored mean abs: {amp_neg_z:.6f}")
print(f"  Ratio: {amp_pos_z / amp_neg_z:.4f}  (should be ~1.0)")

# ============================================================
# Test C: chb11 amplitude check
# ============================================================
print("\n[Test C] chb11 amplitude check")
chb11 = np.load(RAW_CACHE_DIR / "chb11_raw.npz")
X11_pos = chb11["X_seizure"]
X11_neg = chb11["X_baseline"]
amp11_pos = float(np.abs(X11_pos).mean())
amp11_neg = float(np.abs(X11_neg).mean())
print(f"  chb11 pos: {len(X11_pos)}, mean abs amp = {amp11_pos:.6e}")
print(f"  chb11 neg: {len(X11_neg)}, mean abs amp = {amp11_neg:.6e}")
print(f"  Ratio pos/neg: {amp11_pos / amp11_neg:.2f}")

# ============================================================
# INTERPRETATION
# ============================================================
print("\n" + "=" * 65)
print("INTERPRETATION:")
print("=" * 65)
if auc_a > 0.7:
    print(f"X Test A AUC = {auc_a:.3f} > 0.7")
    print("  STRUCTURAL LEAKAGE confirmed.")
    print("  Even random labels give high AUC -> there is a bug.")
elif auc_a > 0.55:
    print(f"? Test A AUC = {auc_a:.3f} - mild structural correlation")
    print("  Possibly partial leakage or small dataset effect.")
else:
    print(f"OK Test A AUC = {auc_a:.3f} - no structural leakage")
    print("  The earlier high AUC was real cross-patient learning.")

if amp11_pos / amp11_neg > 2.0:
    print(f"\n! chb11 pos amplitude is {amp11_pos / amp11_neg:.1f}x neg")
    print("  Amplitude alone could be discriminative.")

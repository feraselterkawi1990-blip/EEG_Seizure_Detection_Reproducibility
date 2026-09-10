"""
Configuration for EEG Seizure Detection Pipeline (LOOCV)
=========================================================
Edit this file to match your local environment.

CHANGELOG (v2 - May 2026):
  - FIX #3: Corrected patient documentation (chb21 = re-recording of chb01,
            NOT chb24). Removed false claim about chb24.
  - FIX #4: All numerical parameters previously hard-coded in loocv.py are
            now declared here, restoring the §3.6.3 "single source of truth"
            promise.
  - FIX #5: Removed unused ICA_VARIANCE_THRESHOLD. Replaced with
            ICA_FRONTAL_ASYMMETRY_THRESHOLD and ICA_FRONTAL_PREFIXES which
            match what preprocessing.py actually does.
  - NEW:    XGB_PARAMS for XGBoost LOOCV.
  - NEW:    SVM cache_size raised to 2 GB for ~2x speedup on CPU.
  - NEW:    XGBOOST added to MODELS_TO_RUN by default.
"""

from pathlib import Path
import os

# =====================================================================
# PATHS - EDIT THESE
# =====================================================================
# Root directory of the CHB-MIT dataset (downloaded from PhysioNet).
# Should contain subfolders: chb01/, chb02/, ..., chb23/, etc.
# Override via environment variable for portability:
#     CHBMIT_PATH=/your/path  python scripts/run_loocv.py
CHB_MIT_PATH = Path(os.environ.get(
    "CHBMIT_PATH",
    r"C:\Users\ASUS\Desktop\EEG_DATA\chbmit"
))

# Where intermediate features and final results are stored.
OUTPUT_PATH = Path("./results")
CACHE_PATH = OUTPUT_PATH / "cache"     # Cached preprocessed features per patient
TABLES_PATH = OUTPUT_PATH / "tables"   # CSV tables for the thesis
FIGURES_PATH = OUTPUT_PATH / "figures" # PNG figures for the thesis
LOGS_PATH = OUTPUT_PATH / "logs"

# =====================================================================
# PATIENTS
# =====================================================================
# CHB-MIT contains 24 case folders (chb01-chb24).
# Per Shoeb (2009), chb21 is a re-recording of chb01 obtained ~1.5 years
# later under different conditions. To avoid subject overlap we exclude
# chb21. We also exclude chb23 and chb24 from this study to match the
# scope of the original thesis (Shoeb 2009 reported 22 patients; we use
# 21 of those 22 by additionally dropping chb21).
#
# Result: 21 patients in {chb01..chb20, chb22}.
PATIENTS = [f"chb{i:02d}" for i in range(1, 23) if i != 21]  # 21 patients

# =====================================================================
# SIGNAL PROCESSING
# =====================================================================
SAMPLING_RATE = 256          # Hz (CHB-MIT native rate)
LOWCUT = 0.5                 # Hz - bandpass low cutoff
HIGHCUT = 30.0               # Hz - bandpass high cutoff (excludes muscle gamma)
FILTER_LENGTH = "auto"       # FIR filter length (let MNE choose)

# ICA parameters
ICA_N_COMPONENTS = 20        # Number of components to extract
ICA_RANDOM_STATE = 97
ICA_METHOD = "fastica"

# --- ICA artifact rejection ---
# Two criteria, applied OR-wise:
#   1. Excess kurtosis (transient artifacts: blinks, jerks, electrode pops)
#   2. Frontal-asymmetry ratio (ocular components dominated by frontal channels)
ICA_KURTOSIS_THRESHOLD = 5.0
ICA_FRONTAL_ASYMMETRY_THRESHOLD = 0.6
# Channel-name prefixes considered "frontal" for the asymmetry test.
# CHB-MIT bipolar derivations starting with FP1- or FP2- are the canonical
# scalp ocular references. (Thesis text §3.2.3 also lists F3/F4/F7/F8;
# this prefix-based implementation captures derivations like FP1-F3,
# FP1-F7, FP2-F4, FP2-F8 which include those electrodes as references.)
ICA_FRONTAL_PREFIXES = ("FP1", "FP2")

# Segmentation
SEGMENT_DURATION_SEC = 1.0    # Window length
OVERLAP_RATIO = 0.5           # 50% overlap between consecutive windows
SAMPLES_PER_SEGMENT = int(SAMPLING_RATE * SEGMENT_DURATION_SEC)  # 256

# =====================================================================
# FEATURE EXTRACTION
# =====================================================================
# Frequency bands (Hz)
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
# Time-domain features per channel:
# mean, std, max, min, peak-to-peak (= 5 features)
# Spectral features per channel: 4 band powers
# 23 channels x 9 features = 207 features per epoch (matches thesis)
N_CHANNELS = 23
N_FEATURES_PER_CHANNEL = 9
N_TOTAL_FEATURES = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 207

# Variance-threshold feature filter (§3.3.4 of thesis Table 3.1).
# Empirically removes 115/207 features on every fold, retaining 92.
VARIANCE_THRESHOLD = 1e-10

# =====================================================================
# CLASS BALANCING
# =====================================================================
# Strategy for handling imbalance in TRAINING set
# Options:
#   "class_weight" : default. Stratified subsample non-ictal to MAX_TRAIN_NEG,
#                    keep all ictal, then use class_weight="balanced" in classifier.
#                    Matches thesis §3.5.3 (~1:5 ictal:non-ictal ratio).
#   "undersample"  : additional pre-step that balances 1:1 BEFORE the MAX_TRAIN_NEG
#                    cap. Useful for ablations comparing 1:1 vs 1:5 ratios.
TRAIN_BALANCE_STRATEGY = "class_weight"

# Test set is always balanced (50/50) per held-out patient (§3.5.2).
TEST_BALANCE = True

# Maximum non-ictal segments retained in the training pool after pooling
# across the other 20 patients. All ictal segments are kept.
# Thesis Table 3.1: 100,000.
MAX_TRAIN_NEG = 100_000

# =====================================================================
# RANDOM SEEDS (for reproducibility)
# =====================================================================
RANDOM_SEED = 42

# =====================================================================
# MODEL HYPERPARAMETERS
# =====================================================================
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": None,
    "min_samples_split": 2,
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "class_weight": "balanced",
}

SVM_PARAMS = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,        # Required for AUC
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "cache_size": 2000,         # MB; default 200 is too small for 40k samples
}
# SVM training subsample (Thesis Table 3.1: 40,000)
SVM_TRAIN_SUBSAMPLE = 40_000

# XGBoost parameters - new in v2.
# Conservative defaults that train in ~5-10 min/fold on CPU with hist method.
XGB_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",      # CPU-efficient histogram method
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    # Note: scale_pos_weight=1 because we already balance via subsampling
    # and class_weight is not directly supported in XGBClassifier — we
    # achieve the balance through MAX_TRAIN_NEG ratio.
}

LSTM_PARAMS = {
    "lstm_units_1": 64,
    "lstm_units_2": 32,
    "dropout": 0.4,
    "dense_units": 16,
    "batch_size": 64,
    "epochs": 60,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
}

CNN_PARAMS = {
    "conv1_filters": 64,
    "conv1_kernel": 3,
    "conv2_filters": 32,
    "conv2_kernel": 3,
    "pool_size": 2,
    "dropout": 0.4,
    "dense_units": 32,
    "batch_size": 64,
    "epochs": 60,
    "learning_rate": 1e-3,
    "early_stopping_patience": 10,
}

# =====================================================================
# LOOCV PROTOCOL PARAMETERS
# =====================================================================
# Internal validation split for threshold tuning (§3.5.4 of thesis).
# A fraction of the (post-MAX_TRAIN_NEG) training pool is held out as
# validation, used ONLY to pick the operating threshold via Youden's J.
# Floor of 1000 ensures the validation set is large enough to give a
# stable threshold estimate even on smallest training pools.
VAL_SPLIT_RATIO = 0.10
VAL_SPLIT_FLOOR = 1000

# Threshold search (§3.5.4 of thesis Table 3.1).
THRESHOLD_RANGE = (0.05, 0.95)
THRESHOLD_N_STEPS = 19   # 19 evenly spaced thresholds in [0.05, 0.95]

# =====================================================================
# MODELS TO RUN
# =====================================================================
# Default: include XGBoost (added in v2).
# To run only sklearn classics: ["RF", "SVM", "XGB"]
# To include deep models (CPU 8-15 hours): ["RF", "SVM", "XGB", "CNN", "LSTM"]
MODELS_TO_RUN = ["RF", "SVM", "XGB"]

# Save per-patient diagnostic plots (training curves, confusion matrices)
SAVE_DIAGNOSTICS = True

# =====================================================================
# Create output directories on import
# =====================================================================
for _p in [OUTPUT_PATH, CACHE_PATH, TABLES_PATH, FIGURES_PATH, LOGS_PATH]:
    _p.mkdir(parents=True, exist_ok=True)

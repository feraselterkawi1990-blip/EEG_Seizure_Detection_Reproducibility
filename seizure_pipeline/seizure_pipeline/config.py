"""
Configuration for EEG Seizure Detection Pipeline (LOOCV)
=========================================================
Edit this file to match your local environment.
"""

from pathlib import Path

# =====================================================================
# PATHS - EDIT THESE
# =====================================================================
# Root directory of the CHB-MIT dataset (downloaded from PhysioNet).
# Should contain subfolders: chb01/, chb02/, ..., chb22/, etc.
CHB_MIT_PATH = Path(os.environ.get(
    "CHBMIT_PATH",
    r"C:\Users\ASUS\Desktop\THESIS WORK\chbmit"
))
# Where intermediate features and final results are stored.
OUTPUT_PATH = Path("./results")
CACHE_PATH = Path(r"D:\THESIS_CACHE")     # Cached preprocessed features per patient
TABLES_PATH = OUTPUT_PATH / "tables"   # CSV tables for the thesis
FIGURES_PATH = OUTPUT_PATH / "figures" # PNG figures for the thesis
LOGS_PATH = OUTPUT_PATH / "logs"

# =====================================================================
# PATIENTS
# =====================================================================
# CHB-MIT contains 23 case folders (chb01-chb24, chb21 was re-recorded as chb24).
# We use the 22 primary patient folders matching the thesis scope.
PATIENTS = [f"chb{i:02d}" for i in range(1, 23) if i != 21]  # 21 patients
# Note: chb21 is the re-recording of chb01 with different equipment configuration;
# excluding it avoids subject overlap. To use all 23 cases, set:
# PATIENTS = [f"chb{i:02d}" for i in range(1, 24) if i != 21]

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
# Artifact rejection thresholds
ICA_KURTOSIS_THRESHOLD = 5.0    # Reject components with kurtosis > 5 (likely artifact)
ICA_VARIANCE_THRESHOLD = 0.95   # Reject components explaining >95% variance (likely line noise / huge artifact)

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
# 23 channels × 9 features = 207 features per epoch (matches thesis)
N_CHANNELS = 23
N_FEATURES_PER_CHANNEL = 9
N_TOTAL_FEATURES = N_CHANNELS * N_FEATURES_PER_CHANNEL  # 207

# =====================================================================
# CLASS BALANCING
# =====================================================================
# Strategy for handling imbalance in TRAINING set
# Options: "undersample" (drop non-seizure to match seizure count)
#          "class_weight" (keep all data, weight classes inversely)
TRAIN_BALANCE_STRATEGY = "class_weight"

# Test set is always balanced (random undersampling to match seizure count)
# This matches the thesis methodology and avoids inflated accuracy from imbalance.
TEST_BALANCE = True

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
# LOOCV
# =====================================================================
# Models to evaluate. Set to subset for faster runs during debugging.
# Default is RF + SVM only (fastest, no GPU needed).
# To add LSTM and CNN, change to: ["RF", "SVM", "LSTM", "CNN"]
MODELS_TO_RUN = ["RF", "SVM"]

# Save per-patient diagnostic plots (training curves, confusion matrices)
SAVE_DIAGNOSTICS = True

# =====================================================================
# Create output directories on import
# =====================================================================
for _p in [OUTPUT_PATH, CACHE_PATH, TABLES_PATH, FIGURES_PATH, LOGS_PATH]:
    _p.mkdir(parents=True, exist_ok=True)

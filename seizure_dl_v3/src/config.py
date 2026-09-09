"""
Configuration for the deep-learning seizure detection extension.
All hyperparameters are kept here for reproducibility.

Aligned with the thesis classical-ML protocol where applicable.
"""
from pathlib import Path

# ============================================================
# PATHS — adjust these to your machine
# ============================================================
DATA_ROOT = Path(r"C:\Users\ASUS\Desktop\EEG_DATA\chbmit")
PROJECT_ROOT = Path(r"C:\Users\ASUS\Desktop\seizure_dl_v3\seizure_dl_v3")

# Existing cache is stored on D:
RAW_CACHE_DIR = Path(r"D:\seizure_dl_v3\cache_raw")

RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = RESULTS_DIR / "training_logs"
# ============================================================
# COHORT — identical to thesis
# ============================================================
PATIENTS = [
    "chb01", "chb02", "chb03", "chb04", "chb05",
    "chb06", "chb07", "chb08", "chb09", "chb10",
    "chb11", "chb12", "chb13", "chb14", "chb15",
    "chb16", "chb17", "chb18", "chb19", "chb20",
    "chb22"
]
# Note: chb21 (re-recording of chb01), chb23, chb24 excluded — same as thesis

# Cohort markers from thesis Table 5.0a (for reporting/analysis)
UNIVERSAL_COLLAPSE = ["chb03", "chb06", "chb07", "chb11", "chb13", "chb16"]
CALIBRATION_COLLAPSE = ["chb04", "chb05", "chb22"]
HEALTHY = ["chb08", "chb09", "chb10", "chb17", "chb18", "chb20"]
BORDERLINE = ["chb01", "chb02", "chb12", "chb14", "chb15", "chb19"]

# ============================================================
# SIGNAL PROCESSING — identical to thesis where possible
# ============================================================
TARGET_FS = 256                # CHB-MIT native sampling rate
WINDOW_SEC = 4.0               # 4-second windows
OVERLAP = 0.5                  # 50% overlap (stride = 2.0s)
WINDOW_SAMPLES = int(TARGET_FS * WINDOW_SEC)  # 1024
STRIDE_SAMPLES = int(TARGET_FS * WINDOW_SEC * (1 - OVERLAP))  # 512

# Bandpass filter: same as thesis FIR (0.5-50 Hz)
BANDPASS_LOW = 0.5
BANDPASS_HIGH = 50.0
NOTCH_FREQ = 60.0  # CHB-MIT was recorded in the US

# Channels — common subset across all patients (intersection)
# These 18 bipolar channels are present in every CHB-MIT EDF
COMMON_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]
N_CHANNELS = len(COMMON_CHANNELS)  # 18

# ============================================================
# CLASS IMBALANCE — identical to thesis
# ============================================================
NEG_TO_POS_RATIO = 5  # 1:5 (positive:negative) during training
RANDOM_SEED = 42

# ============================================================
# TRAINING — common to all DL models
# ============================================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 8
VAL_SPLIT_RATIO = 0.10  # 10% of training data held out for early stopping
THRESHOLD_RANGE = [round(0.05 + 0.05 * i, 3) for i in range(19)]  # 0.05..0.95

# ============================================================
# MODEL-SPECIFIC HYPERPARAMETERS
# ============================================================
EEGNET_CONFIG = {
    "F1": 8,            # Number of temporal filters
    "D": 2,             # Depth multiplier (spatial filters per temporal filter)
    "F2": 16,           # Number of pointwise filters
    "kernel_length": 64,  # Half the sampling rate (256/2 = 128 → use 64)
    "dropout": 0.5,
}

SHALLOWCONVNET_CONFIG = {
    "n_filters_time": 40,
    "filter_time_length": 25,
    "n_filters_spat": 40,
    "pool_time_length": 75,
    "pool_time_stride": 15,
    "dropout": 0.5,
}

CNNLSTM_CONFIG = {
    "cnn_filters": 32,
    "cnn_kernel": 7,
    "lstm_hidden": 64,
    "lstm_layers": 2,
    "dropout": 0.4,
}

# ============================================================
# DEVICE — lazy import to allow config to be loaded without torch
# ============================================================
def get_device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_pin_memory():
    return get_device() == "cuda"


# Resolved at runtime when first needed
DEVICE = "cuda"  # default; will be checked properly in training.py
NUM_WORKERS = 0
PIN_MEMORY = True

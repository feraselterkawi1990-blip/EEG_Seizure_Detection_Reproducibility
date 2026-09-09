# Seizure DL v3 — Deep Learning Extension to Master's Thesis

**Author:** Firas Amrajaa Abdulwahid Altarkawi
**Goal:** Strict cross-patient (LOOCV) deep-learning evaluation on CHB-MIT,
following the same protocol as the Master's thesis classical-ML pipeline.

This codebase trains and evaluates three EEG-specific deep architectures
under the **identical 21-patient LOOCV protocol** used for RF/SVM/XGBoost
in the thesis. The output is a directly-comparable extension that
demonstrates whether deep learning rescues cross-patient generalisation.

## Architectures Compared

| Model | Reference | Parameters | Why |
|-------|-----------|------------|-----|
| **EEGNet** | Lawhern et al. 2018 | ~3K | Compact, EEG-specific, the de-facto baseline |
| **ShallowConvNet** | Schirrmeister et al. 2017 | ~40K | EEG-specific spectral filtering |
| **CNN-LSTM** | This work | ~150K | Spatial CNN + temporal LSTM |

## Hardware Requirements

- **GPU:** NVIDIA RTX 4070 Laptop (8 GB VRAM) — confirmed working
- **CUDA:** 12.x
- **RAM:** 16 GB minimum (32 GB recommended)
- **Storage:** 50 GB free for raw segment cache

## Quickstart

```powershell
# 1. Setup (one-time, ~10 minutes)
cd C:\Users\ASUS\Desktop\seizure_dl_v3
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# 2. Verify GPU
python scripts/verify_gpu.py

# 3. Prepare raw segments (one-time, ~30 minutes)
python scripts/prepare_raw_segments.py

# 4. Smoke test on one patient (~15 min)
python scripts/run_smoke_test.py

# 5. Full LOOCV (12-15 hours)
python scripts/run_dl_loocv.py --model eegnet
python scripts/run_dl_loocv.py --model shallowconvnet
python scripts/run_dl_loocv.py --model cnnlstm

# 6. Analyse results
python scripts/analyze_dl_results.py
```

## Protocol — Identical to Thesis

- 21 patients (chb01-chb20, chb22) — chb21/23/24 excluded (same as thesis)
- 4-second windows, 50% overlap
- Standardisation: per-patient z-score during training, applied to test
- Class imbalance: 1:5 ratio (same as thesis)
- Threshold tuning: Youden-J on validation split (same as thesis)
- Reporting: Accuracy, Sensitivity, Specificity, AUC, MCC, F1
- Statistical testing: Wilcoxon Holm-corrected pairwise (same as thesis)

## Output Structure

```
results/
├── per_fold_eegnet.csv
├── per_fold_shallowconvnet.csv
├── per_fold_cnnlstm.csv
├── summary.csv
├── statistical_comparison.csv
└── figures/
    ├── dl_vs_classical_boxplots.png
    ├── per_patient_dl_vs_rf.png
    └── universal_collapse_dl_check.png
```

## Reference to Thesis

Classical-ML baselines (for direct comparison):

| Model | Mean AUC | 95% CI |
|-------|----------|--------|
| RF    | 0.575    | [0.509, 0.631] |
| SVM   | 0.540    | [0.474, 0.605] |
| XGB   | 0.503    | [0.440, 0.560] |

**Universal-collapse subset:** chb03, chb06, chb07, chb11, chb13, chb16
**Calibration-collapse subset:** chb04, chb05, chb22

## Ethics & Reproducibility

- Random seed fixed: 42
- All hyperparameters in `src/config.py`
- All training logs saved to `results/training_logs/`
- Models saved per fold for inspection

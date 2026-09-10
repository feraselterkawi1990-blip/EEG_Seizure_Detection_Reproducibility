# EXP-A — 4-Second Windows + StandardScaler

## Overview

EXP-A is a controlled experiment in the EEG seizure-detection reproducibility pipeline.

Its purpose is to isolate the effect of **window length** while keeping the classical Part I preprocessing and normalization strategy unchanged.

The experiment uses:

* CHB-MIT Scalp EEG
* 21-patient leave-one-patient-out cross-validation (LOOCV)
* 4-second EEG windows
* 50% window overlap
* 23 native EEG channels
* 0.5–30 Hz band-pass filtering
* ICA-based artifact cleaning
* 207 raw features (9 features × 23 channels)
* variance-based feature filtering
* StandardScaler fitted on the training data only
* RF, SVM, and XGB classifiers
* Youden's J threshold selection using the validation subset

The experiment is designed as the **4-second + StandardScaler** cell of the thesis 2×2 factorial design.

---

## Scientific Question

The main question is:

> Does increasing the EEG window length from 1 second to 4 seconds improve seizure-detection performance when the original Part I feature extraction and StandardScaler pipeline are preserved?

This allows the effect of window length to be separated from the effect of per-segment normalization.

The complete factorial design is:

| Condition   |    Window | Normalization       | RF Mean AUC |
| ----------- | --------: | ------------------- | ----------: |
| Part I      |     1 sec | StandardScaler      |      0.5745 |
| Stage 3-iso |     1 sec | Per-segment z-score |      0.5891 |
| **EXP-A**   | **4 sec** | **StandardScaler**  |  **0.5821** |
| Stage 3     |     4 sec | Per-segment z-score |      0.8660 |

All values above are based on 21 LOOCV folds.

---

## Experimental Design

EXP-A changes the **window duration** relative to Part I while retaining the Part-I-style classical processing pipeline.

### Processing pipeline

```text
CHB-MIT EDF recordings
        │
        ▼
Channel standardization
        │
        ▼
0.5–30 Hz band-pass filter
        │
        ▼
ICA artifact cleaning
        │
        ▼
4-second segmentation
50% overlap
        │
        ▼
Seizure labels based on
window-center time
        │
        ▼
9 features × 23 channels
= 207 raw features
        │
        ▼
VarianceThreshold
        │
        ▼
StandardScaler
fit on training data only
        │
        ├──► Random Forest
        ├──► SVM
        └──► XGBoost
```

### What EXP-A does NOT use

EXP-A does **not** apply per-segment z-score normalization.

The raw 4-second segments are passed to feature extraction, and StandardScaler is subsequently fitted using the training data only.

This is intentional: the experiment is designed to test the effect of changing the window length without simultaneously changing the normalization strategy.

---

## Dataset

The experiment uses the following 21-patient cohort:

```text
chb01
chb02
chb03
chb04
chb05
chb06
chb07
chb08
chb09
chb10
chb11
chb12
chb13
chb14
chb15
chb16
chb17
chb18
chb19
chb20
chb22
```

Patient data are not included in this repository.

The CHB-MIT dataset should be obtained independently from the official dataset source.

After downloading the dataset, update `config.py`:

```python
CHB_MIT_PATH = Path(r"C:\path\to\chbmit")
```

The repository contains the code required to construct the EXP-A cache from the downloaded EDF recordings.

---

## Repository Structure

```text
EXP-A/
│
├── README.md
├── config.py
├── requirements.txt
│
├── scripts/
│   └── run_loocv_4sec.py
│
├── src/
│   ├── __init__.py
│   ├── cache.py
│   ├── cache_4sec.py
│   ├── data_loader.py
│   ├── features.py
│   ├── loocv.py
│   ├── loocv_4sec.py
│   ├── metrics.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── run_loocv_4sec.py
│   └── visualize.py
│
└── results/
    ├── cache_4sec/
    ├── tables/
    ├── figures/
    └── logs/
```

Generated caches and experimental outputs are not required to be committed to the repository.

---

## Requirements

Recommended environment:

* Python 3.10+
* NumPy
* SciPy
* pandas
* scikit-learn
* MNE
* XGBoost
* Matplotlib

Install the required packages with:

```powershell
pip install -r requirements.txt
```

A virtual environment is recommended.

---

## Configuration

The main configuration file is:

```text
config.py
```

Important settings include:

```python
SAMPLING_RATE = 256

LOWCUT = 0.5
HIGHCUT = 30.0

OVERLAP_RATIO = 0.5

N_CHANNELS = 23
N_FEATURES_PER_CHANNEL = 9
N_TOTAL_FEATURES = 207

TRAIN_BALANCE_STRATEGY = "class_weight"
TEST_BALANCE = True

RANDOM_SEED = 42
```

### Important: 4-second windows

Although `config.py` retains:

```python
SEGMENT_DURATION_SEC = 1.0
```

this value should **not** be changed for EXP-A.

The EXP-A cache builder explicitly overrides the segmentation duration:

```python
WINDOW_DURATION_SEC = 4.0
```

Therefore `cache_4sec.py` is the source of truth for the EXP-A window length.

At 256 Hz:

```text
4 seconds × 256 Hz = 1024 samples/window
```

---

## Running EXP-A

From the `EXP-A` directory:

### 1. Smoke test

Run only two patients:

```powershell
python scripts/run_loocv_4sec.py --dry-run
```

This is useful for verifying:

* dataset paths
* EDF loading
* channel standardization
* preprocessing
* ICA
* 4-second segmentation
* feature extraction
* cache generation
* model training

### 2. Build the complete 4-second cache

```powershell
python scripts/run_loocv_4sec.py --cache-only
```

This builds:

```text
results/cache_4sec/
```

with one compressed cache file per patient:

```text
chb01.npz
chb02.npz
...
chb22.npz
```

### 3. Run the complete experiment

The default command runs:

```text
RF + SVM + XGB
```

using all 21 patients:

```powershell
python scripts/run_loocv_4sec.py
```

If the cache has already been created:

```powershell
python scripts/run_loocv_4sec.py --skip-cache
```

### 4. Rebuild the cache

To force regeneration:

```powershell
python scripts/run_loocv_4sec.py --force
```

### 5. Run selected models

For example:

```powershell
python scripts/run_loocv_4sec.py --models RF XGB
```

This skips SVM and is useful for faster exploratory runs.

The complete reported EXP-A experiment, however, includes **RF + SVM + XGB**.

---

## LOOCV Procedure

For each of the 21 patients:

1. One patient is held out as the test patient.
2. The remaining 20 patients form the training pool.
3. Training data are pooled across patients.
4. The training pool is optionally subsampled according to the configured class-balancing strategy.
5. A validation subset is selected from the training pool.
6. Missing/infinite feature values are replaced with zero.
7. Variance filtering is fitted using training data only.
8. StandardScaler is fitted using training data only.
9. The selected classifier is trained.
10. The validation probabilities are used to select a threshold using Youden's J statistic.
11. The held-out patient's test probabilities are evaluated using that threshold.
12. Accuracy, sensitivity, specificity, AUC, MCC, and related statistics are recorded.

This procedure is repeated for all 21 held-out patients.

---

## Results

The final EXP-A per-fold results are stored in:

```text
results/tables/loocv_per_fold_4sec_stdscaler.csv
```

### Aggregate AUC

| Model |   Mean AUC |     SD | Minimum | Maximum | Folds |
| ----- | ---------: | -----: | ------: | ------: | ----: |
| RF    | **0.5821** | 0.1446 |  0.2771 |  0.8166 |    21 |
| SVM   | **0.5481** | 0.1467 |  0.2616 |  0.8633 |    21 |
| XGB   | **0.5194** | 0.1623 |  0.0814 |  0.8509 |    21 |

The RF result is the principal result for comparison with the thesis factorial design.

---

## 2×2 Factorial Comparison

The four conditions are:

| Cell        | Window | Normalization       | RF Mean AUC |
| ----------- | -----: | ------------------- | ----------: |
| Part I      |  1 sec | StandardScaler      |  **0.5745** |
| Stage 3-iso |  1 sec | Per-segment z-score |  **0.5891** |
| EXP-A       |  4 sec | StandardScaler      |  **0.5821** |
| Stage 3     |  4 sec | Per-segment z-score |  **0.8660** |

The comparison shows that increasing the window length from 1 second to 4 seconds under StandardScaler produces only a modest change in RF mean AUC:

```text
0.5745 → 0.5821
```

The much larger change occurs when per-segment normalization is used in the 4-second condition:

```text
0.5821 → 0.8660
```

This is the central reason EXP-A is included: it provides the controlled **4-second + StandardScaler** condition needed to interpret the Stage 3 result.

---

## Output Files

After a complete run, the main outputs are:

```text
results/
│
├── cache_4sec/
│   ├── chb01.npz
│   ├── chb02.npz
│   └── ...
│
├── tables/
│   └── loocv_per_fold_4sec_stdscaler.csv
│
└── logs/
    └── loocv_4sec.log
```

The cache files can be regenerated from the original CHB-MIT EDF data and are therefore treated as generated artifacts rather than source code.

---

## Reproducibility Notes

### Random seed

The pipeline uses:

```python
RANDOM_SEED = 42
```

The LOOCV fold uses:

```python
RANDOM_SEED + fold_idx
```

to make fold-specific sampling deterministic.

### Test balancing

The current EXP-A configuration uses:

```python
TEST_BALANCE = True
```

Therefore the held-out test set is balanced by randomly sampling equal numbers of ictal and non-ictal segments when both classes are available.

### Feature count

The initial feature representation contains:

```text
23 channels × 9 features = 207 features
```

A variance filter is fitted on the training data and reduces the feature dimensionality before StandardScaler.

In the reported EXP-A run, the resulting feature count was:

```text
92 features
```

across the folds.

---

## Scientific Interpretation

EXP-A is not intended to outperform Stage 3.

Its purpose is methodological.

By keeping the classical Part-I-style feature extraction and StandardScaler treatment while changing the window length from 1 second to 4 seconds, EXP-A provides a controlled comparison against the 4-second per-segment-normalized Stage 3 condition.

The observed RF mean AUCs are:

```text
Part I       0.5745
Stage 3-iso  0.5891
EXP-A        0.5821
Stage 3      0.8660
```

Thus, within this experimental design, the large performance difference associated with Stage 3 cannot be attributed to the 4-second window alone.

---

## Citation

If this repository is used in academic work, please cite the associated thesis:

> Firas Altarkawi, “The Window-Normalization Effect in EEG Seizure Detection: A Methodological Audit,” MSc Neuroscience, Bahçeşehir University.

The CHB-MIT dataset should be cited according to the dataset's official citation requirements.

---

## License

This repository contains research code only.

The CHB-MIT EEG recordings are not redistributed with this repository. Users must obtain the dataset independently and comply with its applicable terms of use.

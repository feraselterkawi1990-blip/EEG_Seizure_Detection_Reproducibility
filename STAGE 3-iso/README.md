# Stage 3-Isolated — Reproducibility Package

## Overview

This directory contains the reproducibility code for the **Stage 3-Isolated** experiment reported in the thesis:

> **The Window-Normalization Effect in EEG Seizure Detection: A Methodological Audit**

The purpose of Stage 3-Isolated is to isolate the effect of **per-segment z-score normalization** while keeping the main classical machine-learning pipeline aligned with the Part I baseline.

The experiment uses the same patient cohort, EEG preprocessing framework, 1-second segmentation, feature representation, variance filtering, class-balancing strategy, and LOOCV evaluation framework as the baseline pipeline. The key methodological change is that **each EEG segment is z-score normalized independently before feature extraction**.

This experiment therefore provides a controlled comparison for assessing whether normalization itself contributes to the performance differences observed in the thesis.

---

## Scientific Question

The experiment addresses the following methodological question:

> **Does per-segment z-score normalization improve cross-patient EEG seizure detection when the rest of the classical pipeline is kept consistent with the Part I baseline?**

The experiment is designed to help distinguish the effect of normalization from other factors such as:

* window length,
* channel representation,
* frequency range,
* feature representation,
* classifier architecture.

Stage 3-Isolated uses **1-second windows and the original 23-channel / 207-feature representation**, while applying per-segment z-score normalization before feature extraction.

---

## Experimental Design

### Pipeline

The Stage 3-Isolated processing sequence is:

```text
CHB-MIT EEG
    ↓
EDF loading and channel standardization
    ↓
0.5–30 Hz band-pass filtering
    ↓
ICA artifact cleaning
    ↓
1-second segmentation
    ↓
50% overlap
    ↓
Per-segment z-score normalization
    ↓
Feature extraction
    ↓
207-feature representation
    ↓
LOOCV
    ↓
VarianceThreshold
    ↓
Classical ML classifier
    ↓
Validation-based Youden J threshold
    ↓
Held-out patient evaluation
```

The important distinction from the standard Part I cache is the position of the normalization step:

```text
Standard pipeline:
raw EEG → preprocessing → segmentation → feature extraction

Stage 3-Isolated:
raw EEG → preprocessing → segmentation → per-segment z-score
         → feature extraction
```

The normalization is therefore performed **before feature extraction** and independently for every EEG segment and channel.

---

## Dataset

The experiment uses the **CHB-MIT Scalp EEG Database**.

The reproducibility repository does **not** contain the EEG dataset or generated cache files. The dataset must be obtained separately from the original public CHB-MIT source.

The configured patient cohort is:

```text
chb01–chb20
chb22
```

for a total of **21 patients**.

`chb21` is excluded because it is a re-recording of `chb01`, which would introduce subject overlap. `chb23` and `chb24` are also outside the cohort used for the thesis experiments.

### Dataset location

The local dataset path is controlled through the `CHBMIT_PATH` environment variable.

If the variable is not defined, the current configuration uses:

```text
C:\Users\ASUS\Desktop\THESIS WORK\chbmit
```

For another computer, set `CHBMIT_PATH` to the local CHB-MIT dataset directory.

---

## Repository Structure

The Stage 3-Isolated directory contains the following main components:

```text
STAGE 3-iso/
│
├── config.py
├── requirements.txt
├── CHANGELOG.md
├── README.md
│
├── src/
│   ├── cache_iso.py
│   ├── loocv_iso_full.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   └── metrics.py
│
├── scripts/
│   ├── run_loocv.py
│   ├── run_loocv_iso_full.py
│   ├── run_distribution_shift.py
│   └── run_statistical_tests.py
│
└── results/
    ├── cache/
    ├── cache_iso/
    ├── figures/
    ├── logs/
    └── tables/
```

The `results/` directories are intended for locally generated artifacts.

Generated cache and result files should **not** be committed to the repository.

---

# Installation

## 1. Python environment

The pipeline requires Python with the scientific Python packages listed in:

```text
requirements.txt
```

Install the dependencies with:

```powershell
pip install -r requirements.txt
```

The pipeline uses packages including:

* NumPy
* pandas
* SciPy
* scikit-learn
* MNE
* XGBoost

The exact versions should follow `requirements.txt`.

---

## 2. Configure the CHB-MIT dataset

Set the `CHBMIT_PATH` environment variable to the location of the downloaded CHB-MIT dataset.

For example, in PowerShell:

```powershell
$env:CHBMIT_PATH = "C:\path\to\chbmit"
```

Verify the path:

```powershell
python -c "import os; print(os.environ.get('CHBMIT_PATH'))"
```

If `CHBMIT_PATH` is not set, `config.py` uses its configured default path.

---

# Configuration

The principal experimental settings are defined in:

```text
config.py
```

Important settings include:

| Parameter                          |        Value |
| ---------------------------------- | -----------: |
| Patients                           |           21 |
| Sampling rate                      |       256 Hz |
| Band-pass                          |    0.5–30 Hz |
| ICA components                     |           20 |
| Segment duration                   |     1 second |
| Overlap                            |          50% |
| Native channels                    |           23 |
| Features before variance filtering |          207 |
| Feature layout                     |       23 × 9 |
| Variance threshold                 |        1e-10 |
| Maximum training negatives         |      100,000 |
| Training strategy                  | class-weight |
| Test balancing                     |      enabled |
| Random seed                        |           42 |
| Random forest estimators           |          200 |
| SVM subsample                      |       40,000 |
| Default models                     | RF, SVM, XGB |

The default model set is:

```python
MODELS_TO_RUN = ["RF", "SVM", "XGB"]
```

The full ISO runner also allows individual models to be selected from the command line.

---

# Building the Stage 3-Isolated Cache

The Stage 3-Isolated cache is separate from the standard Part I cache.

It is stored in:

```text
results/cache_iso/
```

Each patient produces a compressed `.npz` cache containing:

```text
F
y
file_ids
n_seizures
```

The raw EEG segments are not retained in the cache.

### Why a separate cache?

The standard cache and the isolated cache differ in the point at which normalization is applied.

Standard:

```text
segment → feature extraction
```

Stage 3-Isolated:

```text
segment → per-segment z-score → feature extraction
```

Keeping a separate cache prevents the two experimental conditions from being mixed.

---

# Per-Segment Z-Score Normalization

For every EEG segment and channel, normalization is performed independently over the time dimension.

For a segment \(X\):

$$
X_{norm} =
\frac{X-\mu_X}{\sigma_X+\epsilon}
$$

where:

* \(\mu_X\) is the mean of that segment/channel,
* \(\sigma_X\) is its standard deviation,
* \(\epsilon = 10^{-8}\).

The normalized signal is then passed to the same feature-extraction framework.

This is deliberately different from the train-pool `StandardScaler` used in the Part I feature pipeline.

In Stage 3-Isolated, the per-segment normalization is part of **cache generation**.

---

# Reproducing the Cache

The cache can be rebuilt locally from the CHB-MIT dataset.

The relevant implementation is:

```text
src/cache_iso.py
```

The cache-building function is:

```python
build_all_iso_caches(force=False)
```

When rebuilding the cache, the generated files remain under:

```text
results/cache_iso/
```

These files are local derived artifacts and are **not distributed through this repository**.

---

# Running the Full Stage 3-Isolated LOOCV

The specific entry point for the full isolated experiment is:

```text
scripts/run_loocv_iso_full.py
```

Once the required `cache_iso` files have been generated, run:

```powershell
python scripts/run_loocv_iso_full.py
```

The default configuration runs:

```text
RF
SVM
XGB
```

### Run selected models

Random Forest and XGBoost:

```powershell
python scripts/run_loocv_iso_full.py --models RF XGB
```

XGBoost only:

```powershell
python scripts/run_loocv_iso_full.py --models XGB
```

The full runner uses the existing `cache_iso` files and does **not** rebuild them automatically.

---

# LOOCV Procedure

The experiment uses strict **leave-one-patient-out cross-validation**.

For each fold:

1. One patient is held out as the test patient.
2. The remaining 20 patients form the training pool.
3. Training data are loaded from the ISO caches.
4. Negative training segments are capped at 100,000.
5. All available ictal training segments are retained.
6. A validation subset is sampled from the training pool.
7. `VarianceThreshold` is fitted using training data only.
8. The same fitted feature filter is applied to validation and test data.
9. The classifier is trained.
10. Validation probabilities are used to select the decision threshold using Youden's J statistic.
11. The selected threshold is applied to the completely held-out test patient.
12. Performance metrics are recorded for that fold.

This procedure prevents the held-out patient's data from influencing model fitting, feature selection, or threshold selection.

---

# Feature Preprocessing During LOOCV

After loading the ISO features, the LOOCV implementation performs:

```text
NaN/Inf handling
      ↓
VarianceThreshold
      ↓
Classifier
```

Importantly, **no StandardScaler is applied in `loocv_iso_full.py`**.

This is intentional.

The per-segment z-score has already been applied during cache generation. Adding another feature-level StandardScaler at the LOOCV stage would introduce an additional preprocessing operation and would no longer represent the intended isolated comparison.

The variance filter is fitted on the training data only and then applied to validation and test data.

The configured variance threshold is:

```text
1e-10
```

The expected feature representation starts at:

```text
207 features
```

with the variance filter typically retaining approximately:

```text
92 features
```

depending on the actual fold/data.

---

# Class Balancing

The configuration uses:

```python
TRAIN_BALANCE_STRATEGY = "class_weight"
```

and:

```python
TEST_BALANCE = True
```

Training negatives are additionally limited by:

```python
MAX_TRAIN_NEG = 100000
```

For balanced evaluation, the held-out test patient's available classes are balanced before computing the reported classification metrics.

The same random seed framework is used across folds:

```text
RANDOM_SEED = 42
```

with fold-specific random-number-generator offsets.

---

# Threshold Selection

The classification threshold is **not selected using the held-out test patient**.

For each fold, the threshold is selected from validation predictions using Youden's J statistic.

The tested threshold range is:

```text
0.05 to 0.95
```

using 19 evenly spaced candidate thresholds.

If the validation data contain only one class, the implementation falls back to:

```text
0.5
```

The selected threshold is then applied unchanged to the held-out test patient.

---

# Models

The default Stage 3-Isolated configuration supports:

### Random Forest

```text
n_estimators = 200
class_weight = balanced
random_state = 42
n_jobs = -1
```

### SVM

```text
kernel = RBF
C = 1
gamma = scale
probability = True
class_weight = balanced
```

Because SVM training can become computationally expensive, the implementation limits the training sample to the configured maximum SVM subsample size when necessary:

```text
40,000 samples
```

### XGBoost

The configured XGBoost model uses:

```text
n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
tree_method = hist
random_state = 42
```

---

# Output Files

The full ISO LOOCV runner writes the main per-fold results to:

```text
results/tables/loocv_per_fold_iso_full.csv
```

The log is written to:

```text
results/logs/loocv_iso_full.log
```

The per-fold results contain measurements including:

```text
accuracy
sensitivity
specificity
AUC
MCC
fit time
decision threshold
number of retained features
```

The runner also prints model-level mean and standard-deviation summaries to the console.

If the Part I results file is available locally:

```text
results/tables/loocv_per_fold.csv
```

the ISO runner can additionally compare the Stage 3-Isolated results with the Part I results, including the mean AUC difference and the number of improved folds.

---

# Expected Reproducibility Workflow

A complete reproduction on a new machine is:

```text
1. Clone/copy this repository
        ↓
2. Obtain the CHB-MIT dataset separately
        ↓
3. Install requirements.txt
        ↓
4. Configure CHBMIT_PATH
        ↓
5. Build results/cache_iso/
        ↓
6. Run scripts/run_loocv_iso_full.py
        ↓
7. Inspect results/tables/loocv_per_fold_iso_full.csv
        ↓
8. Compare the resulting metrics with the thesis
```

The dataset and generated cache are intentionally not included in the repository.

---

# Reproducibility and Data Policy

This repository contains **code and documentation**, not the CHB-MIT EEG recordings.

The following are generated locally and should remain outside version control:

```text
results/cache/
results/cache_iso/
results/logs/
results/tables/
results/figures/
```

In particular:

```text
results/cache_iso/*.npz
```

must not be uploaded to GitHub.

A researcher wishing to reproduce the experiment should obtain the CHB-MIT dataset independently and rebuild the cache locally.

---

# Important Methodological Distinction

Stage 3-Isolated should not be confused with the original Stage 3 experiment.

| Component     | Part I baseline           | Stage 3-Isolated            | Stage 3                  |
| ------------- | ------------------------- | --------------------------- | ------------------------ |
| Window        | 1 s                       | **1 s**                     | 4 s                      |
| Overlap       | 50%                       | **50%**                     | 50%                      |
| Channels      | 23 native                 | **23 native**               | 18 bipolar               |
| Band-pass     | 0.5–30 Hz                 | **0.5–30 Hz**               | 0.5–50 Hz                |
| Features      | 207                       | **207 → variance-filtered** | 198                      |
| Normalization | train-pool StandardScaler | **per-segment z-score**     | per-segment z-score      |
| Models        | RF/SVM/XGB                | **RF/SVM/XGB**              | EEGNet/RF variants       |
| Evaluation    | patient-level LOOCV       | **patient-level LOOCV**     | patient-level evaluation |

The purpose of Stage 3-Isolated is therefore not to reproduce every detail of the original Stage 3 implementation.

Instead, it provides a **controlled normalization experiment using the classical Part I framework**.

---

# Interpretation

The result of this experiment should be interpreted as evidence about the effect of the **normalization strategy under the 1-second classical pipeline**.

It should not be interpreted as a direct reproduction of the original Stage 3 result, because Stage 3 and Stage 3-Isolated differ in several other methodological components, including window length, channel representation, frequency range, and feature representation.

The experiment is specifically useful for separating:

```text
Window effect
        vs.
Normalization effect
```

within a controlled classical-machine-learning setting.

No target AUC is hard-coded into the reproducibility procedure. The purpose of the repository is to allow the result to be independently reproduced from the stated dataset, configuration, and code.

---

# Files Not Included

The following are intentionally excluded from the public reproducibility package:

* CHB-MIT EEG recordings
* Generated `.npz` cache files
* Local machine-specific paths
* Generated logs
* Generated result tables
* Generated figures

These artifacts can be recreated locally by following the workflow above.

---

# Reproducibility Checklist

Before reproducing the experiment, verify:

* [ ] Python environment is installed.
* [ ] `requirements.txt` has been installed.
* [ ] CHB-MIT has been obtained separately.
* [ ] `CHBMIT_PATH` points to the dataset.
* [ ] The configured 21-patient cohort is available.
* [ ] `results/cache_iso/` can be generated.
* [ ] The ISO cache is generated using `src/cache_iso.py`.
* [ ] The full LOOCV is run using `scripts/run_loocv_iso_full.py`.
* [ ] `results/tables/loocv_per_fold_iso_full.csv` is generated.
* [ ] Generated cache and result artifacts remain excluded from Git.

---

## Citation

If this code is used or adapted for research, please cite the associated thesis:

> **The Window-Normalization Effect in EEG Seizure Detection: A Methodological Audit**

The CHB-MIT dataset should be cited according to the requirements of its original distribution/source.

---

## License / Research Use

This repository is provided for research reproducibility and academic use. Please consult the licenses and terms of the individual datasets and third-party software packages used by the pipeline.

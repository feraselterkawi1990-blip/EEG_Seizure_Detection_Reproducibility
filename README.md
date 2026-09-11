# EEG Seizure Detection — LOOCV Pipeline (v2)

A reproducible Python pipeline for **Leave-One-Patient-Out Cross-Validation (LOOCV)** on the **CHB-MIT Scalp EEG Database**.

The pipeline implements the classical machine-learning experiments reported in the thesis:

> **The Window-Normalization Effect in EEG Seizure Detection: A Methodological Audit**

The primary Part I experiment evaluates **Random Forest (RF), Support Vector Machine (SVM), and XGBoost (XGB)** under strict patient-level cross-validation.

This repository contains the source code, configuration, and execution instructions required to reproduce the experiment. **CHB-MIT EEG recordings and generated cache files are not included.**

---

## 1. Overview

The main research question addressed by Part I is whether classical machine-learning models can generalize reliably across previously unseen patients when evaluated using strict patient-level cross-validation.

The primary pipeline uses:

* 21-patient LOOCV
* 1-second EEG windows
* 50% overlap
* 256 Hz sampling rate
* 23 canonical bipolar EEG channels
* 207 handcrafted features per segment
* 0.5–30 Hz band-pass filtering
* ICA preprocessing
* Training-only fitting of classical feature transformations
* Validation-based threshold selection using Youden's J statistic
* Random Forest, SVM, and XGBoost classifiers

---

## 2. Dataset

The experiments use the **CHB-MIT Scalp EEG Database** available through PhysioNet.

The final LOOCV cohort contains:

```text
chb01–chb20 + chb22
```

for a total of **21 patients**.

`chb21` is excluded because it is a re-recording of `chb01`.

The repository does **not** contain the CHB-MIT EEG recordings. Users must obtain the dataset separately and provide its local path through `config.py` or the `CHBMIT_PATH` environment variable.

Dataset source:

https://physionet.org/content/chbmit/1.0.0/

---

## 3. System Requirements

Recommended environment:

* Python 3.10+
* 16 GB RAM minimum
* 32 GB RAM recommended
* At least 50 GB of free disk space for dataset processing and generated cache
* GPU is not required for the RF/SVM/XGB experiment

The primary classical Part I experiment runs on CPU.

---

## 4. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows PowerShell

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies include:

* NumPy
* SciPy
* pandas
* scikit-learn
* MNE
* Matplotlib
* XGBoost

TensorFlow is optional and is only required if the LSTM/CNN branches are enabled.

---

## 5. Dataset Configuration

The dataset path can be configured in two ways.

### Option 1 — Edit `config.py`

Set:

```python
CHB_MIT_PATH = Path(r"/your/path/to/chbmit")
```

### Option 2 — Environment variable

#### Windows PowerShell

```powershell
$env:CHBMIT_PATH = "C:\path\to\chbmit"
```

#### Linux/macOS

```bash
export CHBMIT_PATH=/your/path/to/chbmit
```

Using the environment variable is recommended when sharing the repository because machine-specific paths do not need to be committed to Git.

---

# 6. Signal Processing and Feature Extraction

Each EEG recording is processed using the following pipeline:

```text
CHB-MIT EDF
    ↓
EEG channel selection
    ↓
0.5–30 Hz band-pass filtering
    ↓
ICA artifact processing
    ↓
1-second segmentation
    ↓
50% overlap
    ↓
Feature extraction
    ↓
207-dimensional feature vector
```

### Sampling

* Sampling rate: **256 Hz**
* Segment duration: **1 second**
* Samples per segment: **256**
* Overlap: **50%**

### EEG channels

The pipeline uses 23 canonical bipolar EEG channels.

### Features

Nine features are extracted from each channel.

**Time-domain features**

* Mean
* Standard deviation
* Maximum
* Minimum
* Peak-to-peak amplitude

**Frequency-domain features**

* Delta power: 0.5–4 Hz
* Theta power: 4–8 Hz
* Alpha power: 8–13 Hz
* Beta power: 13–30 Hz

Therefore:

```text
23 channels × 9 features = 207 features
```

---

# 7. ICA Preprocessing

ICA is applied during signal preprocessing.

Configuration:

```text
ICA method: FastICA
Number of components: 20
Random state: 97
Kurtosis threshold: 5.0
Frontal asymmetry threshold: 0.6
```

The preprocessing implementation is located in:

```text
src/preprocessing.py
```

---

# 8. Leave-One-Patient-Out Cross-Validation

The primary evaluation protocol is strict **Leave-One-Patient-Out Cross-Validation**.

For each fold:

```text
20 patients → training pool
1 patient    → held-out test patient
```

The process is repeated for all 21 patients:

```text
21 patients = 21 LOOCV folds
```

The held-out patient's data are not used for fitting the model or for fitting the classical feature preprocessing transformations.

### Test-set balancing

With the default configuration:

```python
TEST_BALANCE = True
```

the held-out patient's test segments are balanced **after the patient-level split**.

The balancing retains all available ictal segments and randomly samples an equal number of non-ictal segments.

Therefore, the held-out patient remains completely excluded from model training and threshold tuning.

---

# 9. Training Data Handling

The training pool contains all patients except the held-out patient.

The default configuration uses:

```python
TRAIN_BALANCE_STRATEGY = "class_weight"
```

All available ictal training segments are retained.

Because the pooled CHB-MIT training data contain a very large number of non-ictal segments, non-ictal samples are capped at:

```python
MAX_TRAIN_NEG = 100_000
```

when necessary.

The classical classifiers then use class weighting to address the remaining class imbalance.

For the default RF/SVM/XGB experiment, this produces a substantially reduced training pool while retaining all ictal samples.

---

# 10. Validation and Threshold Selection

After construction of the training pool, a validation subset is drawn from the training data.

Configuration:

```python
VAL_SPLIT_RATIO = 0.10
VAL_SPLIT_FLOOR = 1000
```

The validation subset is derived from the training pool and is not the held-out patient.

The same validation split is used for all enabled classifiers in a fold so that threshold selection is comparable across models.

The classification threshold is selected using **Youden's J statistic**:

```text
J = sensitivity + specificity − 1
```

The search uses 19 evenly spaced thresholds:

```text
0.05, 0.10, 0.15, ..., 0.90, 0.95
```

The selected threshold is then applied to the held-out test patient.

The test patient is therefore not used for threshold optimization.

---

# 11. Classical Feature Preprocessing

Before model fitting, the handcrafted feature matrix is processed as follows:

```text
NaN/Inf handling
    ↓
VarianceThreshold
    ↓
StandardScaler
    ↓
Model fitting
```

Configuration:

```python
VARIANCE_THRESHOLD = 1e-10
```

Both `VarianceThreshold` and `StandardScaler` are **fitted on the training subset only**.

The fitted transformations are then applied unchanged to the validation and held-out test data.

This prevents the held-out test patient from influencing the learned feature transformation parameters.

The number of retained features is also recorded for each fold.

---

# 12. Models

The default Part I configuration is:

```python
MODELS_TO_RUN = ["RF", "SVM", "XGB"]
```

## Random Forest

Configuration:

```text
n_estimators = 200
max_depth = None
min_samples_split = 2
class_weight = balanced
random_state = 42
```

The Random Forest therefore uses **200 trees**.

## Support Vector Machine

Configuration:

```text
kernel = RBF
C = 1.0
gamma = scale
probability = True
class_weight = balanced
random_state = 42
cache_size = 2000 MB
```

For computational efficiency, the SVM training set is limited to a maximum of:

```text
40,000 samples
```

when the training set is larger.

This subsampling is performed after the training/validation split and after the classical feature transformations have been fitted on the training subset.

## XGBoost

Configuration:

```text
n_estimators = 500
max_depth = 6
learning_rate = 0.05
subsample = 0.8
colsample_bytree = 0.8
tree_method = hist
random_state = 42
objective = binary:logistic
eval_metric = auc
```

No early stopping is used for the XGBoost classifier in the primary LOOCV implementation.

---

# 13. Evaluation Metrics

For each held-out patient and model, the pipeline reports:

* Accuracy
* Sensitivity
* Specificity
* ROC AUC
* TP
* TN
* FP
* FN
* Number of test samples
* Wilson confidence interval for accuracy
* Selected validation threshold
* Number of retained features
* Model fitting time

The final summary reports the mean and standard deviation across the LOOCV folds.

---

# 14. Main Entry Point

The main experiment is executed through:

```text
scripts/run_loocv.py
```

The main LOOCV implementation is:

```text
src/loocv.py
```

Run the complete pipeline with:

```bash
python scripts/run_loocv.py
```

---

# 15. Reproducing the Experiment

## Important cache provenance

The cache used for the reported/reproduced Part I experiment was **generated previously using a separate cache-generation codebase** located in:

```text
CLASSICAL BASELINE/
└── PART I/
    ├── cache_generation/
    └── experiment/
```

The `cache_generation` directory contains the code and documentation for the separate cache-generation workflow.

The `experiment` directory contains the LOOCV experiment itself.

Although the experiment repository also contains:

```text
src/cache.py
```

the `experiment/src/cache.py` implementation was **not used to generate the historical cache used for the reported/reproduced Part I results**.

The historical cache was generated previously and then reused because the EEG preprocessing and feature-extraction stage is computationally expensive.

The workflow used for the reported/reproduced experiment was therefore:

```text
CHB-MIT recordings
        ↓
Separate cache-generation workflow
        ↓
Pre-generated patient cache
        ↓
Part I experiment
        ↓
scripts/run_loocv.py
        ↓
21-patient LOOCV
        ↓
RF / SVM / XGB results
```

The cache files themselves are intentionally **not included in this repository**.

For a new reproduction, researchers should follow the documented cache-generation workflow in:

```text
CLASSICAL BASELINE/PART I/cache_generation/
```

and then use the resulting compatible cache with the experiment pipeline.

---

# 16. Using an Existing Cache

If compatible patient cache files have already been generated and placed in the configured cache location, the cache-building stage can be skipped:

```bash
python scripts/run_loocv.py --skip-cache
```

This runs the experimental LOOCV stage without rebuilding the cache.

---

# 17. Smoke Test

A single-patient smoke test can be used to verify that the pipeline and environment are working:

```bash
python scripts/run_loocv.py --smoke --skip-cache
```

This uses `chb01` only and is intended as a functional test rather than a scientific evaluation.

---

# 18. Statistical Analysis

Statistical analysis can be run separately after LOOCV results have been generated:

```bash
python scripts/run_statistical_tests.py
```

The repository includes:

* Wilcoxon signed-rank tests
* Holm multiple-comparison correction
* Bootstrap confidence intervals

---

# 19. Distribution-Shift Analysis

The repository also contains an optional patient-level distribution-shift analysis using:

* Maximum Mean Discrepancy (MMD)
* Wasserstein distance
* Mahalanobis distance
* Spearman correlation with model performance

Run:

```bash
python scripts/run_distribution_shift.py
```

or, if supported by the current runner:

```bash
python scripts/run_loocv.py --shift
```

This analysis is supplementary to the primary LOOCV experiment.

---

# 20. Output Files

The main generated outputs are stored under:

```text
results/
├── tables/
│   ├── loocv_per_fold.csv
│   ├── loocv_summary.csv
│   ├── thesis_headline_table.csv
│   ├── bootstrap_ci_auc.csv
│   ├── bootstrap_ci_sensitivity.csv
│   ├── bootstrap_ci_specificity.csv
│   └── wilcoxon_auc_holm.csv
├── figures/
└── logs/
    └── run_loocv.log
```

Generated cache files may also exist locally under:

```text
results/cache/
```

but they should **not** be committed to GitHub.

---

# 21. Reproduced Part I Results

The reproduced 21-patient LOOCV experiment produced:

| Model |      Accuracy |   Sensitivity |   Specificity |               AUC |
| ----- | ------------: | ------------: | ------------: | ----------------: |
| RF    | 0.525 ± 0.057 | 0.137 ± 0.142 | 0.912 ± 0.120 | **0.575 ± 0.147** |
| SVM   | 0.527 ± 0.069 | 0.259 ± 0.217 | 0.794 ± 0.194 | **0.540 ± 0.156** |
| XGB   | 0.534 ± 0.065 | 0.170 ± 0.139 | 0.899 ± 0.110 | **0.503 ± 0.143** |

Number of LOOCV folds:

```text
21
```

These values are generated from the repository's LOOCV implementation and are saved in:

```text
results/tables/loocv_summary.csv
results/tables/thesis_headline_table.csv
```

---

# 22. Repository Structure

```text
experiment/
├── config.py
├── requirements.txt
├── README.md
├── CHANGELOG.md
│
├── src/
│   ├── __init__.py
│   ├── cache.py
│   ├── data_loader.py
│   ├── distribution_shift.py
│   ├── features.py
│   ├── loocv.py
│   ├── metrics.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── statistical_tests.py
│   └── visualize.py
│
├── scripts/
│   ├── run_loocv.py
│   ├── run_statistical_tests.py
│   └── run_distribution_shift.py
│
└── results/
    ├── figures/
    ├── logs/
    └── tables/
```

---

# 23. Configuration

The main experimental parameters are centralized in:

```text
config.py
```

Important parameters include:

```python
SAMPLING_RATE = 256

LOWCUT = 0.5
HIGHCUT = 30.0

SEGMENT_DURATION_SEC = 1.0
OVERLAP_RATIO = 0.5

ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 97

N_CHANNELS = 23
N_FEATURES_PER_CHANNEL = 9
N_TOTAL_FEATURES = 207

TRAIN_BALANCE_STRATEGY = "class_weight"
TEST_BALANCE = True
MAX_TRAIN_NEG = 100_000

RANDOM_SEED = 42

VAL_SPLIT_RATIO = 0.10
VAL_SPLIT_FLOOR = 1000

THRESHOLD_RANGE = (0.05, 0.95)
THRESHOLD_N_STEPS = 19
```

The configuration file should be treated as the **single source of truth for the experimental parameters**.

---

# 24. Windows / XGBoost Troubleshooting

On some Windows systems, XGBoost may fail to import because the required Microsoft Visual C++ runtime is not installed.

If an error mentions:

```text
vcomp140.dll
```

install the appropriate **Microsoft Visual C++ Redistributable for Visual Studio 2015–2022 (x64)** from Microsoft.

After installation, restart PowerShell if necessary and verify XGBoost:

```bash
python -c "import xgboost; print(xgboost.__version__)"
```

---

# 25. Data and Privacy

This repository intentionally excludes:

* CHB-MIT EDF recordings
* EEG data
* `.npz` cache files
* Large generated intermediate files
* Machine-specific virtual environments

Users must obtain the dataset separately and configure the local dataset path.

---

# 26. Reproducibility Principle

The purpose of this repository is not to distribute the EEG dataset or pre-generated cache files.

Instead, it provides:

```text
Code
+
Configuration
+
Dependencies
+
Execution instructions
+
Evaluation protocol
+
Generated result structure
```

so that another researcher can understand and reproduce the experimental procedure independently using the CHB-MIT dataset.

The repository separates the **cache-generation workflow** from the **experimental LOOCV workflow** so that the provenance of the data-processing stage and the reported experiment remains explicit.

---

# 27. Citation

If this code contributes to your research, please cite the associated thesis:

> Firas Altarkawi, *The Window-Normalization Effect in EEG Seizure Detection: A Methodological Audit*, MSc Neuroscience, Bahçeşehir University.

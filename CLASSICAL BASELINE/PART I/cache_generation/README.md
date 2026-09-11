# CHB-MIT Cache Generation Pipeline

This directory contains the preprocessing and feature-extraction pipeline used for the classical EEG seizure-detection workflow.

The main purpose of this component is to transform the publicly available CHB-MIT Scalp EEG recordings into per-patient cached files that can be used by the downstream experiment pipeline.

> **Important:** This directory documents the cache-generation workflow. It should not be interpreted as a standalone reproduction of the final thesis results. The downstream LOOCV experiment is documented separately under `PART I/experiment/`.

---

## 1. Dataset

The pipeline uses the **CHB-MIT Scalp EEG Database**.

The dataset is publicly available from PhysioNet and is **not included in this repository**.

The repository therefore provides the code and instructions required to regenerate the intermediate cache from the original EDF recordings.

### Cohort

The intended cohort contains 21 patients:

```text
chb01 – chb20
chb22
```

`chb21` is excluded because it is a re-recording of `chb01`.

The cohort definition is controlled by:

```python
PATIENTS = [f"chb{i:02d}" for i in range(1, 23) if i != 21]
```

---

## 2. Pipeline Overview

The cache-generation workflow can be summarized as:

```text
CHB-MIT EDF recordings
        │
        ▼
MNE EDF loading
        │
        ▼
EEG channel selection
        │
        ▼
Resampling to 256 Hz
        │
        ▼
Canonical channel selection/order
        │
        ▼
0.5–30 Hz band-pass filtering
        │
        ▼
ICA artifact detection and cleaning
        │
        ▼
1-second segmentation
50% overlap
        │
        ▼
Seizure labeling
        │
        ├──────────────► X_raw
        │
        ▼
Feature extraction
        │
        ▼
207-dimensional feature vector
        │
        ▼
F
        │
        ▼
Per-patient .npz cache
```

Each patient is processed independently and stored as a compressed NumPy archive.

---

## 3. Software Requirements

The project provides the following requirements in `requirements.txt`:

```text
numpy>=1.24
scipy>=1.11
pandas>=2.0
scikit-learn>=1.3
mne>=1.6
matplotlib>=3.7
tensorflow>=2.15
h5py>=3.10
```

Python 3.10 or newer is recommended.

The cache-generation preprocessing itself does not require a GPU. TensorFlow remains in the project requirements because it is used elsewhere in the broader project.

A minimum of approximately 16 GB of RAM is recommended for comfortable processing, although actual memory requirements depend on the recording and processing configuration.

---

## 4. Installation

Open PowerShell and move into this directory:

```powershell
cd "C:\path\to\EEG_Seizure_Detection_Reproducibility\CLASSICAL BASELINE\PART I\cache_generation"
```

Create and activate a Python virtual environment if desired:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
pip install -r requirements.txt
```

---

## 5. Dataset Location

The location of the CHB-MIT dataset is controlled through the `CHBMIT_PATH` environment variable.

For example:

```powershell
$env:CHBMIT_PATH = "C:\path\to\chbmit"
```

The directory should contain the CHB-MIT patient folders and summary files, for example:

```text
chbmit/
├── chb01/
├── chb02/
├── chb03/
├── ...
├── chb20/
└── chb22/
```

The dataset itself is not distributed with this repository.

If `CHBMIT_PATH` is not set, `config.py` uses its local default path. This default should be changed for a different machine.

---

## 6. Configuration

The main configuration file is:

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
ICA_METHOD = "fastica"

N_CHANNELS = 23
N_FEATURES_PER_CHANNEL = 9
N_TOTAL_FEATURES = 207

RANDOM_SEED = 42
```

The configuration also defines the four frequency bands:

```python
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}
```

---

## 7. EDF File Discovery and Seizure Annotations

Patient recordings are discovered from the CHB-MIT patient directories.

The pipeline also reads the corresponding patient summary file:

```text
chbXX-summary.txt
```

The summary information is used to identify seizure start and end times for recordings containing annotated seizures.

The data loader parses these annotations and associates seizure intervals with the corresponding EDF recording.

---

## 8. EEG Loading

EDF files are loaded using MNE:

```python
mne.io.read_raw_edf(..., preload=True)
```

The loader keeps EEG channels and excludes non-EEG channels such as:

* MEG
* stimulus channels
* ECG
* EOG
* marked bad channels

Recordings are resampled to the configured sampling rate when necessary.

The target sampling rate is:

```text
256 Hz
```

---

## 9. Canonical EEG Channels

The preprocessing pipeline uses a canonical set of 23 bipolar EEG channels:

```text
FP1-F7
F7-T7
T7-P7
P7-O1
FP1-F3
F3-C3
C3-P3
P3-O1
FP2-F4
F4-C4
C4-P4
P4-O2
FP2-F8
F8-T8
T8-P8-0
P8-O2
FZ-CZ
CZ-PZ
P7-T7
T7-FT9
FT9-FT10
FT10-T8
T8-P8-1
```

The available canonical channels are selected and reordered into a consistent representation.

The implementation does not zero-pad missing canonical channels.

Therefore, recordings that do not satisfy the required channel configuration may be skipped during processing.

---

## 10. Band-Pass Filtering

The EEG recordings are filtered using a zero-phase FIR band-pass filter.

The configured frequency range is:

```text
0.5–30 Hz
```

This corresponds to:

```python
LOWCUT = 0.5
HIGHCUT = 30.0
```

---

## 11. ICA Artifact Cleaning

Independent Component Analysis (ICA) is used to identify and remove artifact-related components.

The configured ICA parameters are:

```python
ICA_N_COMPONENTS = 20
ICA_RANDOM_STATE = 97
ICA_METHOD = "fastica"
```

The implementation detects candidate artifact components using criteria including:

* component kurtosis
* frontal-channel loading/asymmetry

Components exceeding the configured artifact criteria are excluded before reconstruction of the cleaned EEG signal.

The relevant thresholds are defined in `config.py`.

---

## 12. EEG Segmentation

After preprocessing, each recording is divided into fixed-length EEG segments.

The configuration is:

```text
Segment duration: 1 second
Sampling rate:   256 Hz
Samples/window:  256
Overlap:         50%
```

The resulting segment representation is conceptually:

```text
(number of windows, number of channels, 256 samples)
```

When the canonical 23-channel representation is available, this corresponds to:

```text
(number of windows, 23, 256)
```

---

## 13. Seizure Labeling

Each segment is assigned a binary label according to its temporal relationship to the annotated seizure intervals.

The implementation uses the center time of each segment for the label decision.

The resulting label array contains:

```text
0 = non-seizure
1 = seizure
```

The labeling is performed independently for each recording before the patient-level cache is assembled.

---

## 14. Feature Extraction

Feature extraction is implemented in:

```text
src/features.py
```

For each EEG segment, nine features are extracted per channel.

### Time-domain features

Five time-domain statistics are calculated:

1. Mean
2. Standard deviation
3. Maximum
4. Minimum
5. Peak-to-peak amplitude

### Spectral features

Four spectral features are calculated:

1. Delta band power: 0.5–4 Hz
2. Theta band power: 4–8 Hz
3. Alpha band power: 8–13 Hz
4. Beta band power: 13–30 Hz

Spectral power is estimated using Welch's method.

Therefore:

```text
9 features/channel × 23 channels = 207 features
```

The resulting classifier feature matrix is therefore conceptually:

```text
(number of windows, 207)
```

The feature vector is ordered channel by channel:

```text
channel 0:
    mean, std, max, min, p2p, delta, theta, alpha, beta

channel 1:
    mean, std, max, min, p2p, delta, theta, alpha, beta

...

channel 22:
    mean, std, max, min, p2p, delta, theta, alpha, beta
```

### Additional analysis features

`features.py` also provides:

* Power Spectral Density (PSD)
* Inter-channel coherence

These functions are provided for visualization and exploratory analysis and are **not part of the 207-dimensional classifier feature matrix**.

---

## 15. Per-Patient Cache

The cache-generation code is implemented in:

```text
src/cache.py
```

Each patient is stored as a compressed NumPy archive:

```text
<patient_id>.npz
```

For example:

```text
chb01.npz
```

The cache contains the following arrays:

```text
X_raw
F
y
file_ids
n_seizures
```

### `X_raw`

The preprocessed EEG segments.

Conceptually:

```text
(number of windows, number of channels, 256)
```

### `F`

The extracted feature matrix.

For the intended 23-channel representation:

```text
(number of windows, 207)
```

### `y`

Binary seizure labels:

```text
0 = non-seizure
1 = seizure
```

### `file_ids`

Identifiers linking cached windows back to their source EDF recording.

### `n_seizures`

The number of annotated seizures associated with the patient data processed into the cache.

---

## 16. Building a Cache for One Patient

Run commands from the `cache_generation` directory.

To build a cache for one patient:

```powershell
python -c "from src.cache import build_patient_cache; build_patient_cache('chb01')"
```

To force regeneration:

```powershell
python -c "from src.cache import build_patient_cache; build_patient_cache('chb01', force=True)"
```

The generated cache is written to the directory configured by:

```python
CACHE_PATH
```

---

## 17. Building the Full Cohort Cache

To build caches for the configured cohort:

```powershell
python -c "from src.cache import build_all_caches; build_all_caches()"
```

To force regeneration of existing caches:

```powershell
python -c "from src.cache import build_all_caches; build_all_caches(force=True)"
```

Cache generation is the most computationally expensive part of this preprocessing workflow because it requires loading and preprocessing the original EDF recordings.

Existing cache files can be reused by the downstream experiment without rebuilding them.

---

## 18. Optional Pipeline Runner

The directory also contains:

```text
scripts/run_loocv.py
```

This script is a broader end-to-end pipeline runner rather than a cache-generation-only script.

Its stages are:

```text
Stage 1 — Build per-patient caches
Stage 2 — Run LOOCV
Stage 3 — Generate figures and summary tables
```

Examples:

```powershell
python scripts/run_loocv.py
```

Skip cache generation and use existing caches:

```powershell
python scripts/run_loocv.py --skip-cache
```

Force cache rebuilding:

```powershell
python scripts/run_loocv.py --force-cache
```

Run a cache-only smoke test for the first configured patient:

```powershell
python scripts/run_loocv.py --smoke
```

For reproducibility work focused specifically on rebuilding the intermediate cache, the direct functions in `src/cache.py` are clearer because they avoid automatically proceeding to LOOCV and result generation.

---

## 19. Output and Logging

The configured output directories include:

```text
results/
├── cache/
├── figures/
├── logs/
└── tables/
```

The exact location of the generated patient caches is controlled by:

```python
CACHE_PATH
```

The main pipeline runner writes its log to:

```text
results/logs/run_loocv.log
```

Processing messages and errors from individual EDF files are logged during execution.

---

## 20. Relationship to the Downstream Experiment

The cache-generation pipeline is the preprocessing stage.

The downstream experiment is located at:

```text
CLASSICAL BASELINE/PART I/experiment/
```

Conceptually:

```text
CHB-MIT EDF data
       │
       ▼
cache_generation
       │
       ▼
per-patient .npz caches
       │
       ▼
experiment
       │
       ▼
LOOCV
       │
       ▼
metrics / statistical tests / figures / tables
```

The two directories should therefore be treated as separate components:

* `cache_generation/` — preprocessing, segmentation, labeling, and feature-cache construction
* `experiment/` — downstream machine-learning evaluation and analysis

---

## 21. Historical Cache Provenance

An important reproducibility distinction must be made between:

1. the cache-generation code currently provided in this repository, and
2. the exact historical implementation/version that originally produced the cache used for the reported thesis experiment.

The current `cache_generation` directory documents the preprocessing and cache-generation workflow and provides a reproducible implementation for rebuilding caches from the public CHB-MIT EDF data.

However, the exact historical cache-generation version used to produce the original experimental cache may differ from the current implementation.

Therefore, regeneration of a cache using the current code should not automatically be described as bit-for-bit reproduction of the historical cache.

The downstream experiment documentation records the provenance distinction and should be consulted when reproducing the reported thesis results.

---

## 22. Reproducibility Workflow

A clean reproduction workflow is:

### Step 1 — Obtain CHB-MIT

Download the CHB-MIT Scalp EEG Database from its official public source.

The dataset is not included in this repository.

### Step 2 — Configure the dataset path

Set:

```powershell
$env:CHBMIT_PATH = "C:\path\to\chbmit"
```

or update the local default in `config.py`.

### Step 3 — Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Build one patient cache

Start with:

```powershell
python -c "from src.cache import build_patient_cache; build_patient_cache('chb01')"
```

### Step 5 — Inspect the generated cache

Confirm that the patient cache contains:

```text
X_raw
F
y
file_ids
n_seizures
```

and that the feature matrix has the expected 207 columns when the full 23-channel representation is available.

### Step 6 — Build the remaining cohort

```powershell
python -c "from src.cache import build_all_caches; build_all_caches()"
```

### Step 7 — Run the downstream experiment

Follow the instructions in:

```text
CLASSICAL BASELINE/PART I/experiment/README.md
```

---

## 23. Data and Repository Policy

The repository does not distribute the CHB-MIT EDF recordings or generated cache files.

The root `.gitignore` excludes:

```text
*.edf
*.EDF
*.npz
```

This prevents the raw dataset and large generated cache files from being committed accidentally.

Users wishing to reproduce the pipeline should obtain the public dataset independently and generate the caches locally.

---

## 24. Important Configuration Note

The cache-generation `config.py` uses environment-variable support for the CHB-MIT dataset path:

```python
CHB_MIT_PATH = Path(os.environ.get(
    "CHBMIT_PATH",
    r"..."
))
```

The `os` import is required for this configuration mechanism.

This configuration change only allows the dataset path to be supplied through the environment; it does not change the numerical preprocessing parameters or the feature definitions.

---

## 25. Main Source Files

The main preprocessing components are:

```text
src/
├── cache.py
├── data_loader.py
├── features.py
├── preprocessing.py
└── __init__.py
```

### `data_loader.py`

Responsible for:

* discovering patient recordings
* parsing patient summary files
* loading EDF recordings
* selecting EEG channels
* resampling recordings
* standardizing the canonical channel order

### `preprocessing.py`

Responsible for:

* band-pass filtering
* ICA fitting and artifact detection
* ICA cleaning
* EEG segmentation
* seizure labeling

### `features.py`

Responsible for:

* time-domain feature extraction
* Welch spectral band-power extraction
* construction of the 207-dimensional feature matrix
* PSD calculation
* coherence analysis

### `cache.py`

Responsible for:

* processing individual patients
* assembling patient-level arrays
* saving compressed `.npz` cache files
* loading previously generated caches
* processing the configured cohort

---

## 26. Directory Structure

The relevant Part I structure is:

```text
CLASSICAL BASELINE/
└── PART I/
    ├── cache_generation/
    │   ├── README.md
    │   ├── config.py
    │   ├── requirements.txt
    │   ├── scripts/
    │   │   └── run_loocv.py
    │   ├── src/
    │   │   ├── cache.py
    │   │   ├── data_loader.py
    │   │   ├── features.py
    │   │   ├── preprocessing.py
    │   │   └── __init__.py
    │   └── results/
    │
    └── experiment/
        ├── README.md
        ├── CHANGELOG.md
        ├── config.py
        ├── requirements.txt
        ├── scripts/
        ├── src/
        └── results/
```

The `cache_generation` directory may contain additional modules supporting the broader historical pipeline. The files listed above represent the main components relevant to understanding and regenerating the preprocessing cache.

---

## 27. Summary

This pipeline converts the publicly available CHB-MIT EEG recordings into patient-level cached data for downstream seizure-detection experiments.

The main preprocessing configuration is:

```text
Cohort:             21 patients
Sampling rate:      256 Hz
Band-pass:          0.5–30 Hz
ICA:                FastICA, up to 20 components
Window duration:    1 second
Overlap:            50%
Channels:           23 canonical bipolar channels
Features/channel:   9
Total features:     207
```

The generated cache contains:

```text
X_raw
F
y
file_ids
n_seizures
```

The cache-generation pipeline is intentionally separated from the downstream LOOCV experiment so that preprocessing provenance and experimental evaluation can be documented independently.

For the exact procedure used to reproduce the thesis-level experimental results, consult:

```text
CLASSICAL BASELINE/PART I/experiment/README.md
```

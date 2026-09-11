# CHANGELOG — Pipeline v2 (May 2026)

This document explains every change between **v1** (the version Firas
uploaded with `loocv.py` at 403 lines, RF + SVM + LSTM + CNN) and **v2**
(this release). It maps directly to the 24-point review I produced
earlier.

---

## Summary of changes

| File | Status | Lines (v1 → v2) |
|---|---|---|
| `config.py` | **modified** | 150 → 215 |
| `src/loocv.py` | **modified** | 403 → 430 |
| `src/models.py` | **modified** | 217 → 235 |
| `src/visualize.py` | **modified** | 127 → 215 |
| `src/statistical_tests.py` | **NEW** | — → 230 |
| `src/distribution_shift.py` | **NEW** | — → 240 |
| `src/cache.py` | unchanged | 147 |
| `src/data_loader.py` | unchanged | 131 |
| `src/features.py` | unchanged | 146 |
| `src/preprocessing.py` | unchanged | 187 |
| `src/metrics.py` | unchanged | 76 |
| `src/__init__.py` | unchanged | 1 |
| `scripts/run_loocv.py` | **modified** | 99 → 115 |
| `scripts/run_statistical_tests.py` | **NEW** | — → 45 |
| `scripts/run_distribution_shift.py` | **NEW** | — → 35 |

---

## Issues addressed (referencing the 24-point review)

### 🔴 Critical fixes

- **#1 Threshold tuning unified across all models** (`loocv.py`)
  - In v1, RF and SVM had Youden's J threshold tuning; LSTM and CNN
    used a hard-coded threshold of 0.5. This made the comparison
    structurally unfair to the deep models.
  - In v2, the new `tune_threshold()` function is called for every
    classifier (RF, SVM, XGB, LSTM, CNN) using the same validation
    split.

- **#2 Unified validation split** (`loocv.py`)
  - In v1, the deep-learning blocks drew an independent 15% validation
    split from `len(y_train)` after the MAX_TRAIN_NEG cap, which
    overlapped with the RF/SVM training set.
  - In v2, a single `train_mask` and `val_idx` are computed once and
    used by all five classifiers, ensuring the same train/val partition
    everywhere.

- **#3 Patient documentation corrected** (`config.py`)
  - The comment "chb21 was re-recorded as chb24" was historically wrong
    (chb21 is a re-recording of chb01, not chb24). The README claim
    "evaluates on 22 patients" contradicted the code's 21. Both fixed.
  - The behaviour (excluding chb21, chb23, chb24) is unchanged — only
    the documentation is corrected. The thesis text §3.1.2 should now
    be updated to match.

- **#4 All numerical parameters moved to config** (`config.py`,
  `loocv.py`)
  - Previously `VARIANCE_THRESHOLD = 1e-10`, `MAX_TRAIN_NEG = 100_000`,
    `max_svm_train = 40000`, the threshold range `[0.05, 0.95, 19]`,
    and `n_val = max(0.10*N, 1000)` were all hard-coded inside
    `loocv.py`. Thesis §3.6.3 explicitly claims `config.py` is the
    "single source of truth" — this restores that promise.
  - New constants: `VARIANCE_THRESHOLD`, `MAX_TRAIN_NEG`,
    `SVM_TRAIN_SUBSAMPLE`, `VAL_SPLIT_RATIO`, `VAL_SPLIT_FLOOR`,
    `THRESHOLD_RANGE`, `THRESHOLD_N_STEPS`.

- **#5 Dead ICA parameter removed** (`config.py`)
  - `ICA_VARIANCE_THRESHOLD = 0.95` was never read by `preprocessing.py`,
    which actually uses kurtosis + frontal asymmetry. Removed.
  - Added `ICA_FRONTAL_ASYMMETRY_THRESHOLD = 0.6` and
    `ICA_FRONTAL_PREFIXES = ("FP1", "FP2")` to match what the code
    actually does.

### 🟡 Methodological additions

- **#20 XGBoost added to main experiments** (`models.py`, `loocv.py`,
  `config.py`)
  - New `build_xgb()` factory in `models.py`.
  - New `XGB` block in `loocv.evaluate_fold` mirroring the RF block.
  - `XGB_PARAMS` in `config.py` — `tree_method="hist"` for CPU efficiency.
  - Default `MODELS_TO_RUN = ["RF", "SVM", "XGB"]`.

- **#21 Wilcoxon signed-rank tests** (`statistical_tests.py`)
  - Pairwise Wilcoxon across all model pairs on the per-fold AUC.
  - Holm-Bonferroni multiple-testing correction.
  - Effect size: matched-pairs rank biserial correlation (r_rb).
  - Replaces the "paired t-test" mentioned in §5.1 of the thesis,
    which is inappropriate for the heavy-tailed AUC distribution
    induced by catastrophic-collapse folds.

- **#22 Bootstrap 95% CIs on aggregate metrics**
  (`statistical_tests.py`)
  - `bootstrap_ci_per_model()` produces CI bands for the headline
    Table 4.1.

- **#23 Mechanistic distribution-shift analysis**
  (`distribution_shift.py`)
  - Three metrics computed per held-out patient:
    Maximum Mean Discrepancy (RBF kernel, unbiased estimator),
    average per-feature Wasserstein-1 distance,
    Mahalanobis distance to training centroid (regularised pinv).
  - Spearman correlation with per-fold AUC reported for each model.
  - This converts §5.2.3 of the thesis from a qualitative hypothesis
    ("seizure morphology is qualitatively different") into a
    quantitative claim ("Mahalanobis distance correlates rho=-0.X
    with per-fold AUC, p<0.0X").

### 🟢 Cosmetic / portability

- **#10 SVM cache_size raised to 2 GB** (`config.py`)
  - Default sklearn `cache_size=200 MB` is too small for 40k samples.
    Raising to 2 GB gives ~2x speedup on the same hardware.

- **#13 Path now overridable via env variable** (`config.py`)
  - `os.environ.get("CHBMIT_PATH", default)` allows reproducible runs
    on any machine without editing the file.

- **NEW figures in `visualize.py`**:
  - `plot_catastrophic_collapse()` — sensitivity vs AUC scatter,
    collapse zone shaded.
  - `plot_distribution_shift()` — joint scatter of shift metrics vs AUC.

---

## What is NOT addressed in v2 (deferred to thesis text revision)

These are not code changes — they are corrections to the **thesis
manuscript itself** that you (Firas) must make in Word. The code is
already consistent; the manuscript is not.

- Thesis Abstract, §4.2.2, and §6.1 list **three different sets** of
  "catastrophic-collapse patients". Pick one criterion (recommended:
  sensitivity < 0.05 OR AUC < 0.5) and apply it consistently to
  Table 4.2 to derive the canonical list, then update all three
  locations.
- §3.1.2 says chb21 was retained and chb24 excluded — opposite of
  what the code does. Fix to: "chb21, chb23, and chb24 were excluded".
- §5.1 says "paired t-test" — change to "Wilcoxon signed-rank with
  Holm-Bonferroni correction" once you run `run_statistical_tests.py`.
- §3.2.3 lists "Fp1, Fp2, F3, F4, F7, F8" as frontal channels — but
  the code only uses FP1/FP2 prefixes. Either reword the thesis to
  "FP1- and FP2-prefixed bipolar derivations" or extend
  `ICA_FRONTAL_PREFIXES` in `config.py`.

---

## How to run v2

### Quick smoke test (1 patient, ~10 min)
```bash
python scripts/run_loocv.py --smoke
```

### Full run (RF + SVM + XGB)
```bash
python scripts/run_loocv.py
```

### With distribution-shift analysis (adds ~30 min)
```bash
python scripts/run_loocv.py --shift
```

### Re-run statistical tests on existing results
```bash
python scripts/run_statistical_tests.py
```

### Add deep models (CPU 8-15 hours)
Edit `config.py`:
```python
MODELS_TO_RUN = ["RF", "SVM", "XGB", "CNN", "LSTM"]
```
Then re-run `python scripts/run_loocv.py --skip-cache`.

---

## Backwards compatibility

- The cache format (`.npz`) is **unchanged**. v1 caches work directly
  with v2 — no rebuild required.
- Per-fold CSV columns are a **superset** of v1: `n_features_kept` is
  newly added, all previous columns retained.
- `MODELS_TO_RUN` default changed from `["RF", "SVM"]` to
  `["RF", "SVM", "XGB"]`. Set explicitly if you want only RF+SVM.

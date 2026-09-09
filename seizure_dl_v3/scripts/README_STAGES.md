# Stages 1.5 / 2 / 3 — Execution Guide

## Overview

Three follow-up experiments to Stage 1 (which is already complete with
`per_fold_eegnet.csv`).

| Stage | Question | Time | Output |
|-------|----------|------|--------|
| **1.5** | Is gamma/EMG band the source of high AUC? | ~70 min | `per_fold_eegnet_25hz.csv` |
| **2** | Is per-segment amplitude information the source? | ~50 min | `per_fold_eegnet_perrec.csv` |
| **3** | Is the gain over thesis RF due to architecture or normalisation? | ~30 min | `per_fold_classical_perseg.csv` |

**Total: ~2.5 hours.**

## Prerequisites

- Stage 1 already complete (Stage 1's cache stays at `cache_raw/`)
- Same `.venv` activated
- ~10 GB free disk for Stage 1.5 cache

## Step-by-step

### Stage 1.5 — Bandpass 0.5-25 Hz (removes EMG/gamma)

```powershell
# Re-prepare cache with stricter bandpass (~25 min)
python scripts\prepare_raw_segments_25hz.py

# Run LOOCV on the new cache (~45 min)
python scripts\run_dl_loocv_25hz.py
```

**Expected output:** `results\per_fold_eegnet_25hz.csv`

**Interpretation:**
- AUC drops to ~0.6 → EMG/gamma was the source
- AUC stays ~0.83 → DL learns sub-25 Hz features

### Stage 2 — Per-recording (global) z-score

```powershell
# Reuses the original 0.5-50 Hz cache from Stage 1
python scripts\run_dl_loocv_perrec.py
```

⏳ ~50 min

**Expected output:** `results\per_fold_eegnet_perrec.csv`

**Interpretation:**
- AUC drops substantially → amplitude bias was a major source
- AUC stays high → DL learns morphology, not amplitude

### Stage 3 — Classical RF + per-segment z-score

```powershell
# Reuses 0.5-50 Hz cache; extracts band powers + Hjorth + line-length features
python scripts\run_classical_perseg.py
```

⏳ ~30 min (CPU-bound, no GPU needed)

**Expected output:** `results\per_fold_classical_perseg.csv`

**Interpretation:**
- AUC ≈ 0.575 (matches thesis) → architecture is the difference
- AUC rises substantially (e.g. > 0.7) → normalisation was the difference

## Setup

Copy the four scripts into `scripts/` directory:

```powershell
copy "$env:USERPROFILE\Downloads\prepare_raw_segments_25hz.py" "scripts\" -Force
copy "$env:USERPROFILE\Downloads\run_dl_loocv_25hz.py"          "scripts\" -Force
copy "$env:USERPROFILE\Downloads\run_dl_loocv_perrec.py"        "scripts\" -Force
copy "$env:USERPROFILE\Downloads\run_classical_perseg.py"       "scripts\" -Force
```

## Send results back when done

After each stage, send the CSV file content for analysis.

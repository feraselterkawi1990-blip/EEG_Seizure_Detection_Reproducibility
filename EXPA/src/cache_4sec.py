"""
cache_4sec.py
=============
Build cache (cache_4sec/) with 4-second windows on Part I configuration.

For EXP A: completes the 2x2 factorial design by testing
"Part I + 4-second windows + StandardScaler".

CONFIGURATION (single change from cache.py):
  - SEGMENT_DURATION_SEC = 4.0 (overridden internally)
  - All other Part I settings kept:
    - 23 native channels
    - 0.5-30 Hz bandpass
    - 50% overlap
    - ICA cleaning
    - 9 features × 23 channels = 207 raw features
  - NO per-segment z-score (raw signal feeds feature extraction)

Output: cache_4sec/<patient>.npz

This cache is used by loocv_4sec.py with StandardScaler (Part I logic),
NOT per-segment z-score.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config
from . import data_loader, preprocessing, features as feat

logger = logging.getLogger(__name__)


# Separate cache directory
CACHE_4SEC_PATH = config.OUTPUT_PATH / "cache_4sec"
CACHE_4SEC_PATH.mkdir(parents=True, exist_ok=True)

# 4-second window configuration (overrides config defaults internally)
WINDOW_DURATION_SEC = 4.0
SAMPLES_PER_SEGMENT_4SEC = int(config.SAMPLING_RATE * WINDOW_DURATION_SEC)  # 1024


def cache_path_for_4sec(patient_id: str) -> Path:
    return CACHE_4SEC_PATH / f"{patient_id}.npz"


def is_cached_4sec(patient_id: str) -> bool:
    return cache_path_for_4sec(patient_id).exists()


def segment_with_labels_4sec(raw, seizure_intervals: List[Tuple[float, float]]):
    """
    Same as preprocessing.segment_with_labels but with 4-second windows
    and 50% overlap.

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_channels, 1024)
    y : np.ndarray, shape (n_windows,)
    """
    data = raw.get_data()
    sfreq = raw.info["sfreq"]
    n_samples_per_segment = int(WINDOW_DURATION_SEC * sfreq)
    hop = int(n_samples_per_segment * (1 - config.OVERLAP_RATIO))

    if hop < 1:
        hop = 1

    n_total = data.shape[1]
    starts = np.arange(0, n_total - n_samples_per_segment + 1, hop)

    X_list = []
    y_list = []
    for s in starts:
        seg = data[:, s : s + n_samples_per_segment]
        center_time = (s + n_samples_per_segment / 2) / sfreq
        is_ictal = any(start <= center_time <= end
                       for start, end in seizure_intervals)
        X_list.append(seg)
        y_list.append(1 if is_ictal else 0)

    if not X_list:
        return np.zeros((0, data.shape[0], n_samples_per_segment)), np.array([], dtype=np.int8)

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int8)
    return X, y


def preprocess_recording_4sec(raw, seizure_intervals):
    """
    Full per-recording pipeline at 4-second windows:
      bandpass -> ICA cleaning -> 4-sec segmentation -> labels
    
    Mirrors preprocessing.preprocess_recording but uses 4-sec segmentation.
    """
    raw = preprocessing.bandpass_filter(raw)
    raw, ica_info = preprocessing.apply_ica_cleaning(raw)
    X, y = segment_with_labels_4sec(raw, seizure_intervals)
    return X, y, ica_info


def build_patient_cache_4sec(patient_id: str, force: bool = False) -> dict:
    """
    Build 4-second window cache for one patient.

    KEY DIFFERENCE from cache_iso.py:
      - 4-sec windows (vs 1-sec)
      - NO per-segment z-score (raw signal goes to feature extraction)
    """
    cache_file = cache_path_for_4sec(patient_id)
    if cache_file.exists() and not force:
        logger.info(f"[{patient_id}] using existing 4-sec cache")
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}

    logger.info(f"[{patient_id}] building 4-sec cache (StandardScaler-compatible)...")

    summary = data_loader.get_patient_summary(patient_id)
    edf_files = data_loader.list_patient_files(patient_id)

    F_segments = []
    y_segments = []
    file_ids = []
    n_seizures_total = 0

    for file_idx, edf_path in enumerate(edf_files):
        seizure_intervals = summary.get(edf_path.name, [])
        n_seizures_total += len(seizure_intervals)

        try:
            raw = data_loader.load_edf(edf_path)
            raw = data_loader.standardize_channels(raw)
        except Exception as e:
            logger.warning(f"  skipping {edf_path.name}: {e}")
            continue

        if len(raw.ch_names) < config.N_CHANNELS:
            logger.warning(f"  {edf_path.name}: only {len(raw.ch_names)} channels, skipping")
            continue

        try:
            X, y, ica_info = preprocess_recording_4sec(raw, seizure_intervals)
        except Exception as e:
            logger.warning(f"  preprocess failed on {edf_path.name}: {e}")
            continue

        if X.shape[0] == 0:
            logger.warning(f"  {edf_path.name}: no segments produced, skipping")
            continue

        # Truncate to canonical channel count
        if X.shape[1] > config.N_CHANNELS:
            X = X[:, : config.N_CHANNELS, :]

        # NO per-segment z-score — raw signal goes directly to feature extraction
        # (this is the EXP A "Part I-like" treatment)

        # Extract features (same 9 features × 23 channels = 207)
        F = feat.extract_features_batch(X, sfreq=raw.info["sfreq"])

        F_segments.append(F.astype(np.float32))
        y_segments.append(y)
        file_ids.append(np.full(len(y), file_idx, dtype=np.int32))

        logger.info(
            f"  {edf_path.name}: {len(y)} segs, {y.sum()} ictal, "
            f"ICA rejected {ica_info['n_rejected']}/{ica_info['n_components']}"
        )

    if not F_segments:
        raise RuntimeError(f"No usable recordings for {patient_id}")

    F_all = np.concatenate(F_segments, axis=0)
    y_all = np.concatenate(y_segments, axis=0)
    file_ids_all = np.concatenate(file_ids, axis=0)

    np.savez_compressed(
        cache_file,
        F=F_all,
        y=y_all,
        file_ids=file_ids_all,
        n_seizures=n_seizures_total,
    )

    logger.info(
        f"[{patient_id}] 4-sec cached: {len(y_all)} segments, "
        f"{int(y_all.sum())} ictal ({y_all.mean()*100:.2f}%)"
    )

    return {
        "F": F_all,
        "y": y_all,
        "file_ids": file_ids_all,
        "n_seizures": n_seizures_total,
    }


def load_patient_cache_4sec(patient_id: str) -> dict:
    """Load 4-sec cached data for one patient."""
    cache_file = cache_path_for_4sec(patient_id)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No 4-sec cache for {patient_id}. Run build_patient_cache_4sec first."
        )
    with np.load(cache_file) as data:
        return {k: data[k] for k in data.files}


def build_all_4sec_caches(force: bool = False) -> None:
    """Build 4-sec caches for all patients."""
    for pid in config.PATIENTS:
        try:
            build_patient_cache_4sec(pid, force=force)
        except Exception as e:
            logger.error(f"Failed to 4-sec-cache {pid}: {e}")

"""
cache_iso.py
============
Build a parallel cache (cache_iso/) where features are extracted from
per-segment z-scored signals, isolating the normalisation effect for
the Stage 3-isolated experiment.

KEY DIFFERENCE from src/cache.py:
  Standard cache flow:
    raw → bandpass → ICA → segment → extract_features → save F

  ISO cache flow:
    raw → bandpass → ICA → segment → [PER-SEGMENT Z-SCORE] → extract_features → save F
                                       ^^^^^^^^^^^^^^^^^^^^
                                       THE ONLY CHANGE

Per-segment z-score: each (n_channels, n_samples) segment is normalised
independently to zero mean / unit variance per channel.

Everything else is bit-identical to cache.build_patient_cache:
  - Same EDF loading (data_loader)
  - Same channel standardisation (23 native bipolar)
  - Same bandpass (0.5-30 Hz)
  - Same ICA cleaning (kurtosis + frontal asymmetry)
  - Same 1-sec segmentation, 50% overlap
  - Same 9 features × 23 channels = 207 layout
  - Same .npz output format

Output location: config.OUTPUT_PATH / "cache_iso" (separate from Part I cache)
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

import config
from . import data_loader, preprocessing, features as feat

logger = logging.getLogger(__name__)


# Separate cache directory — does NOT overwrite Part I cache
CACHE_ISO_PATH = config.OUTPUT_PATH / "cache_iso"
CACHE_ISO_PATH.mkdir(parents=True, exist_ok=True)


def cache_path_for_iso(patient_id: str) -> Path:
    return CACHE_ISO_PATH / f"{patient_id}.npz"


def is_cached_iso(patient_id: str) -> bool:
    return cache_path_for_iso(patient_id).exists()


def per_segment_zscore(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply per-segment, per-channel z-score normalisation.

    Each segment is normalised independently:
        For each channel c in segment s:
            X[s, c, :] = (X[s, c, :] - mean) / (std + eps)

    Parameters
    ----------
    X : np.ndarray, shape (n_segments, n_channels, n_samples)
        Raw EEG segments after bandpass + ICA cleaning.
    eps : float
        Small constant to prevent division by zero on flat-line segments.

    Returns
    -------
    X_normalised : np.ndarray, same shape as X, dtype float32

    Notes
    -----
    - Computed along axis=2 (time samples), so mean/std are per channel
      per segment (NOT pooled across segments).
    - This matches the per-segment z-score described in the thesis §8.5.4
      and Lawhern et al. 2018 (the EEGNet original).
    """
    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True) + eps
    X_norm = (X - mean) / std
    return X_norm.astype(np.float32)


def build_patient_cache_iso(patient_id: str, force: bool = False) -> dict:
    """
    Run the full ISO preprocessing + feature extraction pipeline for one patient.

    Differs from cache.build_patient_cache ONLY in:
      - Inserts per_segment_zscore(X) after preprocessing.preprocess_recording()
      - Saves to CACHE_ISO_PATH instead of config.CACHE_PATH

    Returns a dict with the cached arrays (also written to disk).
    """
    cache_file = cache_path_for_iso(patient_id)
    if cache_file.exists() and not force:
        logger.info(f"[{patient_id}] using existing ISO cache")
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}

    logger.info(f"[{patient_id}] building ISO cache (per-segment z-score)...")

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
            X, y, ica_info = preprocessing.preprocess_recording(
                raw, seizure_intervals
            )
        except Exception as e:
            logger.warning(f"  preprocess failed on {edf_path.name}: {e}")
            continue

        # Truncate to canonical channel count BEFORE z-scoring
        if X.shape[1] > config.N_CHANNELS:
            X = X[:, : config.N_CHANNELS, :]

        # ============================================================
        # THE KEY MODIFICATION: per-segment z-score on raw EEG
        # ============================================================
        X_normalised = per_segment_zscore(X)

        # Now extract features from the NORMALISED signal
        F = feat.extract_features_batch(X_normalised, sfreq=raw.info["sfreq"])

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

    # Save (no X_raw — RF doesn't need it, saves ~10 GB per patient)
    np.savez_compressed(
        cache_file,
        F=F_all,
        y=y_all,
        file_ids=file_ids_all,
        n_seizures=n_seizures_total,
    )

    logger.info(
        f"[{patient_id}] ISO cached: {len(y_all)} segments, "
        f"{int(y_all.sum())} ictal ({y_all.mean()*100:.2f}%)"
    )

    return {
        "F": F_all,
        "y": y_all,
        "file_ids": file_ids_all,
        "n_seizures": n_seizures_total,
    }


def load_patient_cache_iso(patient_id: str) -> dict:
    """Load ISO-cached data for one patient."""
    cache_file = cache_path_for_iso(patient_id)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No ISO cache for {patient_id}. Run build_patient_cache_iso first."
        )
    with np.load(cache_file) as data:
        return {k: data[k] for k in data.files}


def build_all_iso_caches(force: bool = False) -> None:
    """Build ISO caches for all patients listed in config."""
    for pid in config.PATIENTS:
        try:
            build_patient_cache_iso(pid, force=force)
        except Exception as e:
            logger.error(f"Failed to ISO-cache {pid}: {e}")

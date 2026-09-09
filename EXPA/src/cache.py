"""
cache.py
========
Cache preprocessed features per patient to disk so the expensive
preprocessing + ICA + feature extraction step runs only once.

Each patient gets one .npz file with:
    - F          : (n_windows, 207)  feature matrix
    - X_raw      : (n_windows, 23, 256)  raw segments (for LSTM/CNN)
    - y          : (n_windows,)  labels
    - file_ids   : (n_windows,)  index of source EDF file
    - n_seizures : int  total seizure count
"""

import logging
from pathlib import Path
from typing import Tuple

import numpy as np

import config
from . import data_loader, preprocessing, features as feat

logger = logging.getLogger(__name__)


def cache_path_for(patient_id: str) -> Path:
    return config.CACHE_PATH / f"{patient_id}.npz"


def is_cached(patient_id: str) -> bool:
    return cache_path_for(patient_id).exists()


def build_patient_cache(patient_id: str, force: bool = False) -> dict:
    """
    Run the full preprocessing + feature extraction pipeline for one patient,
    pooling segments across all their EDF recordings.

    Returns a dict with the cached arrays (also written to disk).
    """
    cache_file = cache_path_for(patient_id)
    if cache_file.exists() and not force:
        logger.info(f"[{patient_id}] using existing cache")
        with np.load(cache_file) as data:
            return {k: data[k] for k in data.files}

    logger.info(f"[{patient_id}] building cache...")

    summary = data_loader.get_patient_summary(patient_id)
    edf_files = data_loader.list_patient_files(patient_id)

    X_segments = []
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

        # Extract features for the classical models
        F = feat.extract_features_batch(X, sfreq=raw.info["sfreq"])

        # Truncate/pad to canonical channel count for consistency
        if X.shape[1] > config.N_CHANNELS:
            X = X[:, : config.N_CHANNELS, :]

        X_segments.append(X.astype(np.float32))
        F_segments.append(F.astype(np.float32))
        y_segments.append(y)
        file_ids.append(np.full(len(y), file_idx, dtype=np.int32))

        logger.info(
            f"  {edf_path.name}: {len(y)} segs, {y.sum()} ictal, "
            f"ICA rejected {ica_info['n_rejected']}/{ica_info['n_components']}"
        )

    if not X_segments:
        raise RuntimeError(f"No usable recordings for {patient_id}")

    X_all = np.concatenate(X_segments, axis=0)
    F_all = np.concatenate(F_segments, axis=0)
    y_all = np.concatenate(y_segments, axis=0)
    file_ids_all = np.concatenate(file_ids, axis=0)

    np.savez_compressed(
        cache_file,
        X_raw=X_all,
        F=F_all,
        y=y_all,
        file_ids=file_ids_all,
        n_seizures=n_seizures_total,
    )

    logger.info(
        f"[{patient_id}] cached: {len(y_all)} segments, "
        f"{int(y_all.sum())} ictal ({y_all.mean()*100:.2f}%)"
    )

    return {
        "X_raw": X_all,
        "F": F_all,
        "y": y_all,
        "file_ids": file_ids_all,
        "n_seizures": n_seizures_total,
    }


def load_patient_cache(patient_id: str) -> dict:
    """Load cached data for one patient."""
    cache_file = cache_path_for(patient_id)
    if not cache_file.exists():
        raise FileNotFoundError(
            f"No cache for {patient_id}. Run build_patient_cache first."
        )
    with np.load(cache_file) as data:
        return {k: data[k] for k in data.files}


def build_all_caches(force: bool = False) -> None:
    """Build caches for all patients listed in config."""
    for pid in config.PATIENTS:
        try:
            build_patient_cache(pid, force=force)
        except Exception as e:
            logger.error(f"Failed to cache {pid}: {e}")

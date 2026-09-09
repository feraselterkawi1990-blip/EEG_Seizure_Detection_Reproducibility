"""
preprocessing.py
================
EEG preprocessing pipeline:
1. Bandpass filter (0.5-30 Hz, zero-phase FIR)
2. ICA decomposition + automatic artifact component rejection
   (kurtosis-based + frontal-asymmetry-based)
3. Sliding window segmentation (1s, 50% overlap)

This module FIXES the ICA limitation in the original thesis: components
identified as artifacts are actually rejected before reconstruction.
"""

import logging
from typing import Tuple, List

import mne
import numpy as np
from scipy import stats

import config

mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)


def bandpass_filter(raw: mne.io.Raw) -> mne.io.Raw:
    """Apply zero-phase FIR bandpass filter (0.5-30 Hz)."""
    raw.filter(
        l_freq=config.LOWCUT,
        h_freq=config.HIGHCUT,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="ERROR",
    )
    return raw


def fit_ica(raw: mne.io.Raw, n_components: int = None) -> mne.preprocessing.ICA:
    """Fit ICA on filtered EEG data."""
    n_components = n_components or config.ICA_N_COMPONENTS
    n_components = min(n_components, len(raw.ch_names))

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=config.ICA_RANDOM_STATE,
        method=config.ICA_METHOD,
        max_iter="auto",
    )
    ica.fit(raw, verbose="ERROR")
    return ica


def find_artifact_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.Raw,
) -> List[int]:
    """
    Identify artifact components using two criteria:

    1. KURTOSIS: Components with very high kurtosis often correspond to
       transient artifacts (eye blinks, jerks, electrode pops).
       Threshold: kurtosis > config.ICA_KURTOSIS_THRESHOLD

    2. FRONTAL ASYMMETRY: Components whose topographic weights are
       dominated by frontal channels (FP1-*, FP2-*) are likely ocular.
       Heuristic: |weight(frontal)| / |weight(all)| > 0.6

    Returns the list of component indices to exclude.
    """
    # Get the unmixing matrix to inspect component time courses
    sources = ica.get_sources(raw).get_data()  # shape: (n_components, n_samples)

    bad_components = set()

    # --- Criterion 1: Kurtosis ---
    for idx, comp in enumerate(sources):
        k = stats.kurtosis(comp, fisher=True, bias=False)
        if abs(k) > config.ICA_KURTOSIS_THRESHOLD:
            bad_components.add(idx)
            logger.debug(f"  IC{idx}: rejected (kurtosis={k:.2f})")

    # --- Criterion 2: Frontal-dominant components (likely ocular) ---
    frontal_keywords = ("FP1", "FP2")
    frontal_idx = [
        i for i, name in enumerate(raw.ch_names)
        if any(kw in name for kw in frontal_keywords)
    ]
    if frontal_idx:
        # ICA mixing matrix: shape (n_channels, n_components)
        mixing = ica.mixing_matrix_
        for comp_idx in range(mixing.shape[1]):
            weights = np.abs(mixing[:, comp_idx])
            if weights.sum() == 0:
                continue
            frontal_weight = weights[frontal_idx].sum()
            ratio = frontal_weight / weights.sum()
            if ratio > 0.6:
                bad_components.add(comp_idx)
                logger.debug(f"  IC{comp_idx}: rejected (frontal ratio={ratio:.2f})")

    return sorted(bad_components)


def apply_ica_cleaning(raw: mne.io.Raw) -> Tuple[mne.io.Raw, dict]:
    """
    Fit ICA, identify artifact components, exclude them, and reconstruct.

    Returns
    -------
    cleaned_raw : mne.io.Raw
        EEG signal with artifact components projected out
    info : dict
        Diagnostic info: number of components, number rejected
    """
    ica = fit_ica(raw)
    bad = find_artifact_components(ica, raw)
    ica.exclude = bad

    # Apply: reconstructs signal without the excluded components
    cleaned = raw.copy()
    ica.apply(cleaned, verbose="ERROR")

    info = {
        "n_components": ica.n_components_,
        "n_rejected": len(bad),
        "rejected_indices": bad,
    }
    return cleaned, info


def segment_with_labels(
    raw: mne.io.Raw,
    seizure_intervals: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Segment a continuous recording into fixed-length windows with seizure labels.

    A window is labeled `1` (seizure) if its center time falls within any
    seizure interval; otherwise `0` (non-seizure).

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_channels, n_samples)
    y : np.ndarray, shape (n_windows,)
    """
    data = raw.get_data()  # (n_channels, n_samples)
    sfreq = raw.info["sfreq"]
    n_samples_per_segment = int(config.SEGMENT_DURATION_SEC * sfreq)
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

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.int8)
    return X, y


def preprocess_recording(
    raw: mne.io.Raw,
    seizure_intervals: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Full per-recording pipeline:
      bandpass -> ICA cleaning -> segmentation -> labels

    Returns segments X (n_windows, n_channels, n_samples), labels y, diagnostics.
    """
    raw = bandpass_filter(raw)
    raw, ica_info = apply_ica_cleaning(raw)
    X, y = segment_with_labels(raw, seizure_intervals)
    return X, y, ica_info

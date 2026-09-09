"""
features.py
===========
Feature extraction for EEG segments.

Per channel (9 features):
- 5 time-domain: mean, std, max, min, peak-to-peak
- 4 spectral:   delta, theta, alpha, beta band power

23 channels × 9 features = 207 features per epoch (matches thesis).

Also provides Inter-Channel Coherence (ICC) and Power Spectral Density (PSD)
for visualization and exploratory analysis (not used in the classifier matrix).
"""

import logging
from typing import Tuple

import numpy as np
from scipy import signal

import config

logger = logging.getLogger(__name__)


def time_domain_features(segment: np.ndarray) -> np.ndarray:
    """
    Compute 5 time-domain stats per channel.

    Parameters
    ----------
    segment : np.ndarray, shape (n_channels, n_samples)

    Returns
    -------
    feats : np.ndarray, shape (n_channels, 5)
    """
    means = np.mean(segment, axis=1)
    stds = np.std(segment, axis=1)
    maxs = np.max(segment, axis=1)
    mins = np.min(segment, axis=1)
    p2p = maxs - mins
    return np.stack([means, stds, maxs, mins, p2p], axis=1)


def band_power(segment: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute spectral power in delta/theta/alpha/beta bands per channel
    using Welch's method.

    Returns
    -------
    powers : np.ndarray, shape (n_channels, 4)
    """
    nperseg = min(segment.shape[1], int(sfreq))
    freqs, psd = signal.welch(segment, fs=sfreq, nperseg=nperseg, axis=1)
    powers = np.zeros((segment.shape[0], len(config.BANDS)))
    # NumPy 2.x renamed trapz -> trapezoid; pick whichever is available
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    for i, (band_name, (lo, hi)) in enumerate(config.BANDS.items()):
        mask = (freqs >= lo) & (freqs < hi)
        powers[:, i] = integrate(psd[:, mask], freqs[mask], axis=1)
    return powers


def extract_features_one(segment: np.ndarray, sfreq: float = None) -> np.ndarray:
    """
    Extract the 207-dim feature vector for a single segment.

    Returns
    -------
    feature_vec : np.ndarray, shape (n_channels * 9,)
        Layout: [ch0_t1..ch0_t5, ch0_b1..ch0_b4, ch1_t1..ch1_t5, ...]
    """
    sfreq = sfreq or config.SAMPLING_RATE
    td = time_domain_features(segment)             # (n_ch, 5)
    bp = band_power(segment, sfreq)                # (n_ch, 4)
    feats = np.concatenate([td, bp], axis=1)       # (n_ch, 9)
    return feats.flatten()


def extract_features_batch(X: np.ndarray, sfreq: float = None) -> np.ndarray:
    """
    Extract features for a batch of segments.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_channels, n_samples)

    Returns
    -------
    F : np.ndarray, shape (n_windows, n_channels * 9)
    """
    sfreq = sfreq or config.SAMPLING_RATE
    n_windows = X.shape[0]
    F = np.zeros((n_windows, X.shape[1] * 9), dtype=np.float32)
    for i in range(n_windows):
        F[i] = extract_features_one(X[i], sfreq)
    return F


def feature_names(channel_names: list) -> list:
    """Generate human-readable names for each of the 207 features."""
    stat_labels = ["mean", "std", "max", "min", "p2p"]
    band_labels = list(config.BANDS.keys())
    names = []
    for ch in channel_names:
        for s in stat_labels:
            names.append(f"{ch}_{s}")
        for b in band_labels:
            names.append(f"{ch}_{b}")
    return names


# ---------------------------------------------------------------------
# Visualization-only features (not used in the classifier matrix)
# ---------------------------------------------------------------------
def compute_psd(segment: np.ndarray, sfreq: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD for visualization. Returns (freqs, psd) where psd shape is (n_ch, n_freqs)."""
    sfreq = sfreq or config.SAMPLING_RATE
    nperseg = min(segment.shape[1], int(sfreq))
    freqs, psd = signal.welch(segment, fs=sfreq, nperseg=nperseg, axis=1)
    return freqs, psd


def compute_coherence_matrix(segment: np.ndarray, sfreq: float = None) -> np.ndarray:
    """
    Mean coherence across frequency for every channel pair.

    Returns
    -------
    C : np.ndarray, shape (n_channels, n_channels)
    """
    sfreq = sfreq or config.SAMPLING_RATE
    n_ch = segment.shape[0]
    C = np.eye(n_ch)
    nperseg = min(segment.shape[1], int(sfreq))
    for i in range(n_ch):
        for j in range(i + 1, n_ch):
            f, Cxy = signal.coherence(segment[i], segment[j],
                                       fs=sfreq, nperseg=nperseg)
            mean_c = np.mean(Cxy)
            C[i, j] = mean_c
            C[j, i] = mean_c
    return C

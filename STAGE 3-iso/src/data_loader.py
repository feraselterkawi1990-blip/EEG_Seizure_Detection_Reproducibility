"""
data_loader.py
==============
Load CHB-MIT EDF files and parse seizure annotations from
chbXX-summary.txt files.

The summary files have format:
    File Name: chb01_03.edf
    File Start Time: 13:43:04
    File End Time: 14:43:04
    Number of Seizures in File: 1
    Seizure Start Time: 2996 seconds
    Seizure End Time: 3036 seconds
"""

import re
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import mne
import numpy as np

import config

mne.set_log_level("WARNING")
logger = logging.getLogger(__name__)


def parse_summary_file(summary_path: Path) -> Dict[str, List[Tuple[float, float]]]:
    """
    Parse a chbXX-summary.txt file.

    Returns
    -------
    dict mapping edf filename -> list of (seizure_start_sec, seizure_end_sec)
    Files with zero seizures map to an empty list.
    """
    text = summary_path.read_text(errors="ignore")
    blocks = re.split(r"\n(?=File Name:)", text)
    seizures = {}

    for block in blocks:
        m_name = re.search(r"File Name:\s*(\S+\.edf)", block)
        if not m_name:
            continue
        fname = m_name.group(1)

        # Number of seizures
        m_n = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        n = int(m_n.group(1)) if m_n else 0

        if n == 0:
            seizures[fname] = []
            continue

        # Capture all "Seizure ... Start Time" / "End Time" pairs
        starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", block)
        ends = re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", block)

        if len(starts) != len(ends):
            logger.warning(f"Mismatched seizure annotations in {fname}")
            continue

        seizures[fname] = [(float(s), float(e)) for s, e in zip(starts, ends)]

    return seizures


def list_patient_files(patient_id: str) -> List[Path]:
    """List all .edf files for a given patient, sorted."""
    patient_dir = config.CHB_MIT_PATH / patient_id
    if not patient_dir.exists():
        raise FileNotFoundError(f"Patient directory not found: {patient_dir}")
    return sorted(patient_dir.glob("*.edf"))


def get_patient_summary(patient_id: str) -> Dict[str, List[Tuple[float, float]]]:
    """Load seizure annotations for a patient."""
    summary_path = config.CHB_MIT_PATH / patient_id / f"{patient_id}-summary.txt"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    return parse_summary_file(summary_path)


def load_edf(edf_path: Path) -> mne.io.Raw:
    """
    Load a single EDF file with MNE.

    Picks only EEG channels and drops dummy / status channels (e.g. 'STI 014').
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    # Keep only EEG channels (drop ECG, status, etc. if present)
    eeg_picks = mne.pick_types(raw.info, eeg=True, meg=False, stim=False,
                                ecg=False, eog=False, exclude="bads")
    if len(eeg_picks) > 0:
        raw.pick(eeg_picks)
    # Resample to canonical rate (256 Hz). Some CHB-MIT recordings are at
    # 1024 Hz; without this step, segment counts and cache sizes blow up by 4x.
    if abs(raw.info["sfreq"] - config.SAMPLING_RATE) > 0.5:
        raw.resample(config.SAMPLING_RATE, npad="auto", verbose="ERROR")
    return raw


def get_canonical_channels() -> List[str]:
    """
    Return the canonical list of 23 bipolar channels used in CHB-MIT.
    Some files have additional channels which we drop to maintain consistency.
    """
    return [
        "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
        "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
        "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
        "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
        "FZ-CZ", "CZ-PZ",
        "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1",
    ]


def standardize_channels(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Standardize channel set across recordings:
    - Pick only canonical channels that exist
    - Pad missing channels with zeros (rare; logged as warning)
    """
    canonical = get_canonical_channels()
    existing = [ch for ch in canonical if ch in raw.ch_names]
    missing = [ch for ch in canonical if ch not in raw.ch_names]

    if missing:
        logger.debug(f"Missing channels in {raw.filenames[0]}: {missing}")

    # Pick existing canonical channels in the canonical order
    raw.reorder_channels(existing)
    return raw

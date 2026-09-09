"""
Prepare raw EEG segments from CHB-MIT EDFs (DISK-AWARE VERSION).

Difference from previous version:
- Subsamples baseline windows during preparation to avoid 100+ GB cache
- Keeps at most BASELINE_PER_POS_RATIO × n_pos baseline windows per patient
- Default ratio = 20 (gives plenty of headroom for 1:5 training subsample)
- Uses fixed seed for reproducibility

Output: cache_raw/<patient>_raw.npz with X_seizure and X_baseline arrays.
Expected total cache size: ~5-8 GB (down from ~110 GB).
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch, resample_poly

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import (
    DATA_ROOT, RAW_CACHE_DIR, PATIENTS, COMMON_CHANNELS,
    TARGET_FS, WINDOW_SAMPLES, STRIDE_SAMPLES,
    BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQ, RANDOM_SEED
)

mne.set_log_level("ERROR")
warnings.filterwarnings("ignore")

# Baseline subsampling: keep this many neg windows per pos window
# 20× provides headroom for 1:5 training subsample + 4× validation/diagnostics
BASELINE_PER_POS_RATIO = 20

# Floor: keep at least this many baseline windows even if pos count is tiny
# Useful for diagnostic patients with very few seizures
BASELINE_MIN = 2000

# Cap: do not store more than this many baseline windows per patient
# Prevents extremely long recordings from dominating disk
BASELINE_MAX = 30000


def parse_summary_file(summary_path):
    seizures = {}
    current_file = None
    seiz_starts, seiz_ends = [], []
    with open(summary_path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                if current_file is not None:
                    seizures[current_file] = list(zip(seiz_starts, seiz_ends))
                current_file = line.split(":", 1)[1].strip()
                seiz_starts, seiz_ends = [], []
            elif "Seizure" in line and "Start Time" in line:
                t = int(line.split(":")[-1].strip().split()[0])
                seiz_starts.append(t)
            elif "Seizure" in line and "End Time" in line:
                t = int(line.split(":")[-1].strip().split()[0])
                seiz_ends.append(t)
        if current_file is not None:
            seizures[current_file] = list(zip(seiz_starts, seiz_ends))
    return seizures


def design_filters(fs):
    nyq = fs / 2
    b_bp, a_bp = butter(4, [BANDPASS_LOW / nyq, BANDPASS_HIGH / nyq], btype="band")
    b_notch, a_notch = iirnotch(NOTCH_FREQ, 30, fs)
    return (b_bp, a_bp), (b_notch, a_notch)


def preprocess_signal(data, fs):
    (b_bp, a_bp), (b_notch, a_notch) = design_filters(fs)
    data = filtfilt(b_bp, a_bp, data, axis=-1)
    data = filtfilt(b_notch, a_notch, data, axis=-1)
    if int(fs) != TARGET_FS:
        from math import gcd
        g = gcd(int(fs), TARGET_FS)
        up = TARGET_FS // g
        down = int(fs) // g
        data = resample_poly(data, up, down, axis=-1)
    return data.astype(np.float32)


def is_dummy_channel(name):
    s = name.strip()
    if s in ("", ".", "-", "--"):
        return True
    if s.startswith("-") and (len(s) == 1 or not s[1].isalpha()):
        return True
    if s.upper().startswith(("ECG", "VNS")):
        return True
    return False


def normalise_name(name):
    s = name.strip().upper()
    if len(s) >= 3 and s[-2] == "-" and s[-1].isdigit():
        base = s[:-2]
        if "-" in base:
            return base
    return s


def read_edf_robust(edf_path):
    try:
        raw_test = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception:
        return None, False

    exclude_list = [ch for ch in raw_test.ch_names if is_dummy_channel(ch)]
    del raw_test

    try:
        raw = mne.io.read_raw_edf(
            str(edf_path), preload=True, verbose=False, exclude=exclude_list
        )
        return raw, True
    except Exception:
        try:
            raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
            return raw, True
        except Exception:
            return None, False


def select_channels(raw):
    norm_to_first_idx = {}
    for i, orig in enumerate(raw.ch_names):
        norm = normalise_name(orig)
        if norm not in norm_to_first_idx:
            norm_to_first_idx[norm] = i

    selected_indices = []
    missing = []
    for target in COMMON_CHANNELS:
        norm_target = normalise_name(target)
        if norm_target in norm_to_first_idx:
            selected_indices.append(norm_to_first_idx[norm_target])
        else:
            missing.append(target)

    if missing:
        return None, False, missing

    all_data = raw.get_data()
    return all_data[selected_indices, :], True, []


def slide_windows(data, label_per_sample):
    n_ch, n_t = data.shape
    if n_t < WINDOW_SAMPLES:
        return np.zeros((0, n_ch, WINDOW_SAMPLES), dtype=np.float32), \
               np.zeros(0, dtype=np.int8)
    starts = list(range(0, n_t - WINDOW_SAMPLES + 1, STRIDE_SAMPLES))
    X = np.zeros((len(starts), n_ch, WINDOW_SAMPLES), dtype=np.float32)
    y = np.zeros(len(starts), dtype=np.int8)
    for i, s in enumerate(starts):
        X[i] = data[:, s:s + WINDOW_SAMPLES]
        y[i] = int(label_per_sample[s:s + WINDOW_SAMPLES].any())
    return X, y


def process_patient(patient, rng):
    pdir = DATA_ROOT / patient
    if not pdir.exists():
        print(f"  [SKIP] {patient}: directory not found")
        return None

    summary_path = pdir / f"{patient}-summary.txt"
    if not summary_path.exists():
        print(f"  [SKIP] {patient}: no summary file")
        return None

    seizures = parse_summary_file(summary_path)

    X_pos_list, X_neg_list = [], []
    edf_files = sorted(pdir.glob("*.edf"))
    n_seiz_total = sum(len(v) for v in seizures.values())
    print(f"  {patient}: {len(edf_files)} EDF files, {n_seiz_total} seizures")

    n_files_used, n_files_skipped = 0, 0
    skip_reasons = {}

    for edf in edf_files:
        seiz_intervals = seizures.get(edf.name, [])

        raw, ok = read_edf_robust(edf)
        if not ok:
            n_files_skipped += 1
            skip_reasons["read failed"] = skip_reasons.get("read failed", 0) + 1
            continue

        data, ok, missing = select_channels(raw)
        if not ok:
            n_files_skipped += 1
            key = f"missing {len(missing)}/{len(COMMON_CHANNELS)} channels"
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
            del raw
            continue

        fs = raw.info["sfreq"]
        del raw

        try:
            data = preprocess_signal(data, fs)
        except Exception as e:
            n_files_skipped += 1
            key = f"preprocess: {str(e)[:30]}"
            skip_reasons[key] = skip_reasons.get(key, 0) + 1
            continue

        n_samples_resampled = data.shape[-1]
        label_per_sample = np.zeros(n_samples_resampled, dtype=np.int8)
        for (start_s, end_s) in seiz_intervals:
            i0 = int(start_s * TARGET_FS)
            i1 = min(int(end_s * TARGET_FS), n_samples_resampled)
            label_per_sample[i0:i1] = 1

        X, y = slide_windows(data, label_per_sample)
        if len(X) == 0:
            continue

        X_pos_list.append(X[y == 1])
        X_neg_list.append(X[y == 0])
        n_files_used += 1

    print(f"    Files used: {n_files_used}/{len(edf_files)}")
    if n_files_skipped > 0:
        for reason, count in skip_reasons.items():
            print(f"      [{count}×] {reason}")

    if not X_pos_list and not X_neg_list:
        print(f"  [SKIP] {patient}: no usable data")
        return None

    X_pos = np.concatenate(X_pos_list, axis=0) if X_pos_list else np.zeros(
        (0, len(COMMON_CHANNELS), WINDOW_SAMPLES), dtype=np.float32
    )
    X_neg_full = np.concatenate(X_neg_list, axis=0) if X_neg_list else np.zeros(
        (0, len(COMMON_CHANNELS), WINDOW_SAMPLES), dtype=np.float32
    )

    if n_seiz_total > 0 and len(X_pos) == 0:
        print(f"  [SKIP] {patient}: {n_seiz_total} seizures in summary "
              f"but 0 extracted")
        return None

    # ============================================================
    # KEY FIX: subsample baseline to control disk usage
    # ============================================================
    n_pos = len(X_pos)
    target_neg = max(BASELINE_MIN, n_pos * BASELINE_PER_POS_RATIO)
    target_neg = min(target_neg, BASELINE_MAX)
    target_neg = min(target_neg, len(X_neg_full))

    if target_neg < len(X_neg_full):
        idx = rng.choice(len(X_neg_full), size=target_neg, replace=False)
        idx.sort()  # keep temporal order for reproducibility
        X_neg = X_neg_full[idx]
        print(f"    Subsampled baseline: {len(X_neg_full)} → {len(X_neg)}")
    else:
        X_neg = X_neg_full

    RAW_CACHE_DIR.mkdir(exist_ok=True, parents=True)
    out = RAW_CACHE_DIR / f"{patient}_raw.npz"
    np.savez_compressed(out, X_seizure=X_pos, X_baseline=X_neg)

    size_mb = out.stat().st_size / 1e6
    print(f"  → {out.name}: {len(X_pos)} pos, {len(X_neg)} neg ({size_mb:.0f} MB)")

    return {
        "patient": patient,
        "n_pos": len(X_pos),
        "n_neg": len(X_neg),
        "size_mb": size_mb,
    }


def main():
    print(f"Preparing raw segments → {RAW_CACHE_DIR}")
    print(f"Target Fs={TARGET_FS}, window={WINDOW_SAMPLES} samples "
          f"({WINDOW_SAMPLES/TARGET_FS:.1f}s), stride={STRIDE_SAMPLES} samples")
    print(f"Baseline subsampling: {BASELINE_PER_POS_RATIO}× n_pos "
          f"(min {BASELINE_MIN}, max {BASELINE_MAX})")
    print(f"Channels needed ({len(COMMON_CHANNELS)}): {COMMON_CHANNELS}")
    print()

    if RAW_CACHE_DIR.exists():
        for f in RAW_CACHE_DIR.glob("*.npz"):
            f.unlink()
        print("Cleaned previous cache.\n")

    rng = np.random.RandomState(RANDOM_SEED)

    results = []
    for patient in PATIENTS:
        r = process_patient(patient, rng)
        if r:
            results.append(r)

    print("\n" + "=" * 75)
    print(f"{'Patient':<10} {'n_pos':>8} {'n_neg':>10} {'pos%':>7} {'size_MB':>10}")
    print("-" * 75)
    total_pos, total_neg, total_mb = 0, 0, 0.0
    for r in results:
        pct = 100 * r["n_pos"] / max(r["n_pos"] + r["n_neg"], 1)
        print(f"{r['patient']:<10} {r['n_pos']:>8d} {r['n_neg']:>10d} "
              f"{pct:>6.2f}% {r['size_mb']:>9.0f}")
        total_pos += r["n_pos"]
        total_neg += r["n_neg"]
        total_mb += r["size_mb"]
    print("-" * 75)
    print(f"{'TOTAL':<10} {total_pos:>8d} {total_neg:>10d} "
          f"{100*total_pos/max(total_pos+total_neg,1):>6.2f}% "
          f"{total_mb:>9.0f}")
    print(f"\nProcessed: {len(results)}/{len(PATIENTS)} patients")
    print(f"Total cache size: {total_mb/1000:.1f} GB")


if __name__ == "__main__":
    main()

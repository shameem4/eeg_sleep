"""Data pipeline: EDF/SET -> preprocessed HDF5 epochs, single-channel, device-agnostic.

Core preprocessing (resamples to 128Hz, bandpass 0.3-45Hz, z-normalizes)
and HDF5 caching. Dataset-specific readers are in readers.py and readers_extra.py.

Usage:
    python data_pipeline.py --dataset <key> --acq <type>
    python data_pipeline.py --sanitize              # clean all caches
    python data_pipeline.py --audit                 # audit without modifying
"""
import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import h5py
import mne
import numpy as np
from numpy.typing import NDArray

from config import (
    CACHE_DIR, DATASETS, EPOCH_SAMPLES,
    HIGHPASS_HZ, LOWPASS_HZ, NUM_STAGES, STAGE_NAMES, TARGET_SFREQ,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

mne.set_log_level("ERROR")


# ── Core preprocessing ────────────────────────────────────────────────

def preprocess_raw(raw: mne.io.BaseRaw, channel: str) -> NDArray[np.float32]:
    """Pick single channel, filter, resample, z-normalize. Returns 1D array."""
    raw_ch = raw.copy().pick([channel])
    # Zero-fill NaNs before filtering (scattered NaNs poison the entire signal)
    nan_mask = np.isnan(raw_ch.get_data()[0])
    if nan_mask.any():
        raw_ch.apply_function(lambda x: np.where(np.isnan(x), 0.0, x),
                              picks="all", verbose=False)
    # Filter first at native sample rate (better filter characteristics), then downsample
    raw_ch.filter(l_freq=HIGHPASS_HZ, h_freq=LOWPASS_HZ, verbose=False)
    if raw_ch.info["sfreq"] != TARGET_SFREQ:
        raw_ch.resample(TARGET_SFREQ)
    data = np.asarray(raw_ch.get_data(units="uV")[0], dtype=np.float32)
    mu, sigma = data.mean(), data.std()
    if sigma > 1e-8:
        data = (data - mu) / sigma
    return data


def extract_epochs(data: NDArray[np.float32],
                   stages: NDArray[np.int8],
                   onsets_samples: NDArray[np.int64],
                   ) -> tuple[NDArray[np.float32], NDArray[np.int8]]:
    """Extract fixed-length epochs aligned to stage annotations.

    Returns:
        epochs: (N, EPOCH_SAMPLES) float32
        labels: (N,) int8 with values 0-4
    """
    valid = ((stages >= 0) & (stages < NUM_STAGES)
             & (onsets_samples + EPOCH_SAMPLES <= len(data)))
    if not valid.any():
        return np.empty((0, EPOCH_SAMPLES), np.float32), np.empty(0, np.int8)
    onsets = onsets_samples[valid]
    labels = stages[valid]
    # Vectorized extraction via broadcasting: (N,1) + (EPOCH_SAMPLES,) -> (N, EPOCH_SAMPLES)
    idx = onsets[:, None] + np.arange(EPOCH_SAMPLES)
    return data[idx], labels


# ── Epoch quality validation ─────────────────────────────────────────

# Thresholds for epoch rejection (applied to z-normalized signal)
_FLAT_STD_THRESH = 1e-6      # std below this = dead channel / flat line
_EXTREME_AMP_THRESH = 20.0   # |sample| above this = severe artifact (20 sigma)
_EXTREME_FRAC_THRESH = 0.05  # reject if >5% of samples exceed amplitude threshold


def validate_epochs(epochs: NDArray[np.float32],
                    labels: NDArray[np.int8],
                    ) -> tuple[NDArray[np.float32], NDArray[np.int8], dict]:
    """Drop bad epochs: flat-line, NaN/Inf, extreme artifacts.

    Returns:
        epochs: cleaned (M, EPOCH_SAMPLES)
        labels: cleaned (M,)
        stats: dict with rejection counts per reason
    """
    n = len(epochs)
    if n == 0:
        return epochs, labels, {"total": 0, "flat": 0, "nan_inf": 0, "artifact": 0}

    keep = np.ones(n, dtype=bool)

    # NaN / Inf
    bad_nan = np.isnan(epochs).any(axis=1) | np.isinf(epochs).any(axis=1)
    keep &= ~bad_nan

    # Flat-line (std < threshold)
    stds = epochs.std(axis=1)
    bad_flat = stds < _FLAT_STD_THRESH
    keep &= ~bad_flat

    # Extreme artifacts: reject if >5% of samples exceed threshold
    extreme_samples = np.abs(epochs) > _EXTREME_AMP_THRESH
    bad_artifact = extreme_samples.mean(axis=1) > _EXTREME_FRAC_THRESH
    keep &= ~bad_artifact

    stats = {
        "total": n,
        "flat": int(bad_flat.sum()),
        "nan_inf": int(bad_nan.sum()),
        "artifact": int(bad_artifact.sum()),
        "kept": int(keep.sum()),
    }
    return epochs[keep], labels[keep], stats


# ── HDF5 caching ──────────────────────────────────────────────────────

def _get_reader_and_items(dataset_key: str, ds_path: Path, acq: str,
                          max_subjects: Optional[int] = None):
    """Get reader function and recording list for any dataset."""
    from readers import (
        read_boas_subject, read_eesm17_subject, read_ceegrid_subject,
        read_sleep_edf_recording, read_cap_recording, read_hmc_recording,
        read_eesm23_recording, read_dreem_recording, read_dreamt_recording,
        iter_sleep_edf, iter_cap, iter_hmc,
        iter_eesm23, iter_dreem, iter_dreamt,
    )
    from readers_extra import (
        read_ysyw_recording, read_ds006695_subject, read_svuh_recording,
        read_dreams_recording, read_eesm19_recording, read_dcsm_recording,
        iter_ysyw, iter_svuh, iter_dreams, iter_eesm19, iter_dcsm,
    )

    # BIDS datasets: sub-* directories
    bids_config = {
        "ds005555": read_boas_subject,
        "ds004348": read_eesm17_subject,
        "ds005207": read_ceegrid_subject,
        "ds006695": read_ds006695_subject,
    }
    if dataset_key in bids_config:
        reader_fn = bids_config[dataset_key]
        sub_dirs = sorted(ds_path.glob("sub-*"))
        if max_subjects:
            sub_dirs = sub_dirs[:max_subjects]
        items = [(d.name, d, acq) for d in sub_dirs]
        def bids_reader(item, fn=reader_fn):
            return fn(item[1], acq=item[2])
        return bids_reader, items

    # File-based datasets
    file_config = {
        "sleep_edf": (read_sleep_edf_recording, iter_sleep_edf),
        "cap_slpdb": (read_cap_recording, iter_cap),
        "hmc": (read_hmc_recording, iter_hmc),
        "ds005178": (read_eesm23_recording, iter_eesm23),
        "dreem_dod_h": (read_dreem_recording, iter_dreem),
        "dreem_dod_o": (read_dreem_recording, iter_dreem),
        "dreamt": (read_dreamt_recording, iter_dreamt),
        "ysyw": (read_ysyw_recording, iter_ysyw),
        "svuh": (read_svuh_recording, iter_svuh),
        "dreams": (read_dreams_recording, iter_dreams),
        "ds005185": (read_eesm19_recording, iter_eesm19),
        "dcsm": (read_dcsm_recording, iter_dcsm),
    }
    if dataset_key not in file_config:
        raise NotImplementedError(f"No reader for {dataset_key}")

    reader_fn, iter_fn = file_config[dataset_key]
    items = iter_fn(ds_path)
    if max_subjects:
        items = items[:max_subjects]
    def file_reader(item, fn=reader_fn):
        return fn(item[1], item[2])
    return file_reader, items


def cache_dataset(dataset_key: str, acq: str = "headband",
                  max_subjects: Optional[int] = None) -> Path:
    """Process an entire dataset into a single HDF5 cache file.

    HDF5 structure:
        /{subject_id}/epochs  -> (N_epochs, 3840) float32
        /{subject_id}/labels  -> (N_epochs,) int8
        /{subject_id}/attrs: channel, device, dataset
    """
    ds_info = DATASETS[dataset_key]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    h5_path = CACHE_DIR / f"{dataset_key}_{acq}.h5"
    tmp_path = h5_path.with_suffix(".h5.tmp")

    if h5_path.exists():
        log.info("Cache exists: %s", h5_path)
        return h5_path

    reader, items = _get_reader_and_items(
        dataset_key, ds_info["path"], acq, max_subjects)
    log.info("Processing %s (%d recordings) -> %s", dataset_key, len(items), h5_path)

    total_epochs = 0
    failed = 0
    with h5py.File(tmp_path, "w") as hf:
        for i, item in enumerate(items):
            name = item[0]
            try:
                epochs, labels, ch_name = reader(item)
                if len(epochs) == 0:
                    log.warning("No valid epochs for %s", name)
                    failed += 1
                    continue

                # Epoch quality validation
                epochs, labels, qstats = validate_epochs(epochs, labels)
                dropped = qstats["total"] - qstats["kept"]
                if dropped > 0:
                    log.info("  %s: dropped %d/%d epochs (flat=%d, nan_inf=%d, artifact=%d)",
                             name, dropped, qstats["total"],
                             qstats["flat"], qstats["nan_inf"], qstats["artifact"])
                if len(epochs) == 0:
                    log.warning("No valid epochs after QC for %s", name)
                    failed += 1
                    continue

                grp = hf.create_group(name)
                grp.create_dataset("epochs", data=epochs, compression="gzip",
                                   compression_opts=4,
                                   chunks=(min(32, len(epochs)), EPOCH_SAMPLES))
                grp.create_dataset("labels", data=labels, compression="gzip")
                grp.attrs["channel"] = ch_name
                grp.attrs["device"] = ds_info["device"]
                grp.attrs["dataset"] = dataset_key
                total_epochs += len(epochs)

                if (i + 1) % 10 == 0:
                    log.info("  [%d/%d] %s: %d epochs (total: %d)",
                             i + 1, len(items), name, len(epochs), total_epochs)

            except Exception as e:
                log.error("Failed %s: %s", name, e)
                failed += 1

    log.info("Done: %d total epochs, %d failed", total_epochs, failed)
    tmp_path.rename(h5_path)

    # Print stage distribution
    if total_epochs > 0:
        with h5py.File(h5_path, "r") as hf:
            label_arrays = [np.asarray(hf[s]["labels"]) for s in hf.keys()]
            all_labels = np.concatenate(label_arrays)
            for stage_idx, sname in enumerate(STAGE_NAMES):
                count = (all_labels == stage_idx).sum()
                pct = 100 * count / len(all_labels)
                log.info("  %s: %6d (%.1f%%)", sname, count, pct)

    return h5_path


# ── Cache sanitization ────────────────────────────────────────────────

def audit_cache(h5_path: Path) -> dict:
    """Audit an HDF5 cache file for quality issues without modifying it."""
    stats = {"file": h5_path.stem, "subjects": 0, "total_epochs": 0,
             "kept_epochs": 0,
             "flat": 0, "nan_inf": 0, "artifact": 0, "bad_subjects": []}
    with h5py.File(h5_path, "r") as hf:
        for subj in sorted(hf.keys()):
            epochs = np.array(hf[subj]["epochs"])
            labels = np.array(hf[subj]["labels"])
            _, _, qstats = validate_epochs(epochs, labels)
            stats["subjects"] += 1
            stats["total_epochs"] += qstats["total"]
            stats["kept_epochs"] += qstats["kept"]
            stats["flat"] += qstats["flat"]
            stats["nan_inf"] += qstats["nan_inf"]
            stats["artifact"] += qstats["artifact"]
            if qstats["total"] > 0 and qstats["kept"] == 0:
                stats["bad_subjects"].append(subj)
    return stats


def sanitize_cache(h5_path: Path, dry_run: bool = False) -> dict:
    """Remove bad epochs from an existing HDF5 cache file.

    Rewrites the file (via temp) dropping flat-line, NaN/Inf, and artifact epochs.
    Returns audit stats.
    """
    stats = audit_cache(h5_path)
    # Use exact dropped count (total - kept) to avoid double-counting epochs that
    # satisfy multiple rejection criteria simultaneously.
    total_dropped = stats["total_epochs"] - stats["kept_epochs"]
    if total_dropped == 0:
        log.info("  %s: clean (%d epochs, %d subjects)",
                 h5_path.stem, stats["total_epochs"], stats["subjects"])
        return stats

    if dry_run:
        log.info("  %s: would drop %d epochs (flat=%d, nan_inf=%d, artifact=%d), "
                 "%d bad subjects",
                 h5_path.stem, total_dropped,
                 stats["flat"], stats["nan_inf"], stats["artifact"],
                 len(stats["bad_subjects"]))
        return stats

    # Rewrite file without bad epochs
    tmp_path = h5_path.with_suffix(".h5.sanitized")
    kept_epochs = 0
    dropped_subjects = 0
    with h5py.File(h5_path, "r") as src, h5py.File(tmp_path, "w") as dst:
        for subj in sorted(src.keys()):
            epochs = np.array(src[subj]["epochs"])
            labels = np.array(src[subj]["labels"])
            epochs, labels, _ = validate_epochs(epochs, labels)
            if len(epochs) == 0:
                dropped_subjects += 1
                continue
            grp = dst.create_group(subj)
            grp.create_dataset("epochs", data=epochs, compression="gzip",
                               compression_opts=4,
                               chunks=(min(32, len(epochs)), EPOCH_SAMPLES))
            grp.create_dataset("labels", data=labels, compression="gzip")
            # Copy attributes
            for k, v in src[subj].attrs.items():
                grp.attrs[k] = v
            kept_epochs += len(epochs)

    # Atomic replace
    backup = h5_path.with_suffix(".h5.bak")
    shutil.move(str(h5_path), str(backup))
    shutil.move(str(tmp_path), str(h5_path))
    backup.unlink()

    log.info("  %s: dropped %d epochs, %d subjects removed, %d epochs kept",
             h5_path.stem, total_dropped, dropped_subjects, kept_epochs)
    return stats


def sanitize_all_caches(dry_run: bool = False) -> None:
    """Audit and sanitize all HDF5 cache files."""
    h5_files = sorted(CACHE_DIR.glob("*.h5"))
    if not h5_files:
        log.warning("No HDF5 files in %s", CACHE_DIR)
        return

    print(f"\n{'File':<40} {'Subj':>5} {'Epochs':>8} {'Flat':>6} {'NaN':>5} "
          f"{'Artif':>6} {'BadSub':>6} {'Action'}")
    print("-" * 95)

    total_dropped = 0
    for h5_path in h5_files:
        stats = sanitize_cache(h5_path, dry_run=dry_run)
        dropped = stats["total_epochs"] - stats["kept_epochs"]
        total_dropped += dropped
        action = "CLEAN" if dropped == 0 else ("WOULD FIX" if dry_run else "FIXED")
        print(f"{stats['file']:<40} {stats['subjects']:>5} {stats['total_epochs']:>8} "
              f"{stats['flat']:>6} {stats['nan_inf']:>5} {stats['artifact']:>6} "
              f"{len(stats['bad_subjects']):>6} {action}")

    print(f"\nTotal dropped: {total_dropped} epochs")
    if dry_run and total_dropped > 0:
        print("Run with --sanitize (without --audit) to apply fixes.")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess EEG datasets to HDF5")
    parser.add_argument("--dataset", default="ds005555", choices=list(DATASETS.keys()))
    parser.add_argument("--acq", default="headband",
                        choices=["headband", "psg", "ear", "ceegrid", "scalp",
                                 "in_ear", "around_ear", "forehead"],
                        help="Acquisition type")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects (for testing)")
    parser.add_argument("--sanitize", action="store_true",
                        help="Sanitize all cached HDF5 files (drop bad epochs)")
    parser.add_argument("--audit", action="store_true",
                        help="Audit all cached HDF5 files (dry run, no changes)")
    args = parser.parse_args()

    if args.audit or args.sanitize:
        sanitize_all_caches(dry_run=args.audit)
        return

    h5_path = cache_dataset(args.dataset, acq=args.acq, max_subjects=args.max_subjects)
    print(f"\nCached to: {h5_path}")


if __name__ == "__main__":
    main()

"""Additional dataset readers: YSYW, ds006695, SVUH, DREAMS, ds005185, DCSM.

Same contract as readers.py: each reader returns (epochs, labels, channel_name).
"""
from pathlib import Path
from typing import Optional

import h5py
import mne
import numpy as np
import pandas as pd
import scipy.io as sio
from numpy.typing import NDArray

from config import EPOCH_SAMPLES, EPOCH_SEC, TARGET_SFREQ
from data_pipeline import preprocess_raw, extract_epochs
from readers import EESM_STAGE_TO_AASM

# ── Stage encoding maps ──────────────────────────────────────────────

# YSYW: one-hot keys in HDF5
YSYW_STAGE_KEYS = ["wake", "nonrem1", "nonrem2", "nonrem3", "rem"]
# -> AASM: W=0, N1=1, N2=2, N3=3, REM=4

# ds006695: VisualHypnogram codes
DS006695_STAGE_MAP = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}  # 1=W,2=REM,3=N1,4=N2,5=N3

# SVUH: R&K codes (0=W,1=S1,2=S2,3=S3,4=S4,5=REM,8=movement)
SVUH_STAGE_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 4}  # merge S3+S4->N3

# DREAMS AASM: 5=W,4=REM,3=N1,2=N2,1=N3 (5-sec epochs)
DREAMS_AASM_MAP = {5: 0, 4: 4, 3: 1, 2: 2, 1: 3}


# ── You Snooze You Win (PhysioNet 2018 Challenge) ────────────────────

def iter_ysyw(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all training subjects with signal + arousal annotations."""
    train_dir = ds_path / "training"
    if not train_dir.exists():
        return []
    pairs = []
    for sub_dir in sorted(train_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        mat_files = list(sub_dir.glob("*.mat"))
        arousal_files = [f for f in mat_files if f.name.endswith("-arousal.mat")]
        signal_files = [f for f in mat_files if not f.name.endswith("-arousal.mat")]
        if signal_files and arousal_files:
            pairs.append((sub_dir.name, signal_files[0], arousal_files[0]))
    return pairs


def _parse_ysyw_header(hea_path: Path) -> list[str]:
    """Parse WFDB .hea file to get channel names."""
    lines = hea_path.read_text().strip().splitlines()
    channels = []
    for line in lines[1:]:  # skip first line (record info)
        parts = line.split()
        if len(parts) >= 9:
            channels.append(parts[8])  # channel name is last field
    return channels


def read_ysyw_recording(signal_path: Path, arousal_path: Path,
                        channel: Optional[str] = None,
                        ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one YSYW training recording."""
    # Parse header to get channel names and index
    hea_path = signal_path.with_suffix(".hea")
    ch_names = _parse_ysyw_header(hea_path)

    if channel is None:
        for ch in ["C4-M1", "C3-M2", "F4-M1", "F3-M2", "O2-M1"]:
            if ch in ch_names:
                channel = ch
                break
        if channel is None:
            channel = ch_names[0]

    ch_idx = ch_names.index(channel)

    # Load signal (int16, shape: 13 x N_samples)
    mat = sio.loadmat(str(signal_path))
    signal = mat["val"][ch_idx].astype(np.float64)

    # Create MNE Raw and preprocess
    info = mne.create_info([channel], sfreq=200.0, ch_types=["eeg"])
    raw = mne.io.RawArray(signal.reshape(1, -1) * 1e-6, info, verbose=False)  # uV -> V
    data = preprocess_raw(raw, channel)

    # Load sleep stages from HDF5 arousal file
    with h5py.File(str(arousal_path), "r") as f:
        sleep_grp = f["data"]["sleep_stages"]
        n_samples_annot = sleep_grp["wake"].shape[1]
        # Build per-sample stage array from one-hot
        stage_per_sample = np.full(n_samples_annot, -1, dtype=np.int8)
        for aasm_idx, key in enumerate(YSYW_STAGE_KEYS):
            mask = np.asarray(sleep_grp[key][0], dtype=bool)
            stage_per_sample[mask] = aasm_idx

    # Majority vote per 30s epoch at original 200 Hz
    orig_epoch_samples = 200 * EPOCH_SEC  # 6000
    n_epochs = min(len(data) // EPOCH_SAMPLES,
                   len(stage_per_sample) // orig_epoch_samples)
    stage_chunks = stage_per_sample[:n_epochs * orig_epoch_samples].reshape(n_epochs, -1)

    # Per-epoch majority: count votes for each stage
    stages = np.full(n_epochs, -1, dtype=np.int8)
    for code in range(5):
        counts = np.sum(stage_chunks == code, axis=1)
        stages[counts > orig_epoch_samples // 2] = code
    # Handle ties: use first valid majority
    ambiguous = stages == -1
    if ambiguous.any():
        for i in np.where(ambiguous)[0]:
            vals, cnts = np.unique(stage_chunks[i][stage_chunks[i] >= 0],
                                   return_counts=True)
            if len(vals) > 0:
                stages[i] = vals[cnts.argmax()]

    onsets_samples = np.arange(n_epochs, dtype=np.int64) * EPOCH_SAMPLES
    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


# ── ds006695 (Forehead EEG Patch) ────────────────────────────────────

def read_ds006695_subject(sub_dir: Path, acq: str = "forehead",
                          channel: Optional[str] = None,
                          ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one ds006695 forehead patch subject."""
    sub_id = sub_dir.name
    eeg_dir = sub_dir / "eeg"
    set_path = eeg_dir / f"{sub_id}_task-sleep_eeg.set"
    if not set_path.exists():
        raise FileNotFoundError(f"Set file not found: {set_path}")

    # Load hypnogram from .set metadata
    d = sio.loadmat(str(set_path), squeeze_me=True)
    hypnogram = np.asarray(d["VisualHypnogram"], dtype=int)

    # Read raw EEG
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)
    if channel is None:
        for ch in ["FP1-AFz", "FP2-AFz", "FF"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            channel = raw.ch_names[0]

    data = preprocess_raw(raw, channel)

    # Map hypnogram codes to AASM
    stages = np.array([DS006695_STAGE_MAP.get(int(s), -1) for s in hypnogram],
                      dtype=np.int8)
    onsets_samples = np.arange(len(stages), dtype=np.int64) * EPOCH_SAMPLES
    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


# ── SVUH / UCD Sleep Apnea Database ──────────────────────────────────

def iter_svuh(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all .rec + stage.txt pairs in SVUH."""
    pairs = []
    for rec_file in sorted(ds_path.glob("*.rec")):
        sub_id = rec_file.stem  # e.g. "ucddb002"
        stage_file = ds_path / f"{sub_id}_stage.txt"
        if stage_file.exists():
            pairs.append((sub_id, rec_file, stage_file))
    return pairs


def read_svuh_recording(rec_path: Path, stage_path: Path,
                        channel: Optional[str] = None,
                        ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one SVUH recording (.rec EDF + stage.txt)."""
    # pyedflib reads .rec files that mne rejects
    import pyedflib

    f = pyedflib.EdfReader(str(rec_path))
    try:
        ch_names = f.getSignalLabels()
        sfreqs = [f.getSampleFrequency(i) for i in range(f.signals_in_file)]

        # Pick EEG channel (C3A2 or C4A1, both at 128 Hz)
        if channel is None:
            for ch in ["C3A2", "C4A1"]:
                if ch in ch_names:
                    channel = ch
                    break
            if channel is None:
                channel = ch_names[0]

        ch_idx = ch_names.index(channel)
        signal = f.readSignal(ch_idx).astype(np.float64)
        sfreq = sfreqs[ch_idx]
    finally:
        f.close()

    # Create MNE Raw and preprocess
    info = mne.create_info([channel], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(signal.reshape(1, -1) * 1e-6, info, verbose=False)  # uV -> V
    data = preprocess_raw(raw, channel)

    # Parse stage annotations (one integer per line, per 30s epoch)
    stage_lines = stage_path.read_text().strip().splitlines()
    stages = np.array([SVUH_STAGE_MAP.get(int(s.strip()), -1)
                       for s in stage_lines], dtype=np.int8)

    onsets_samples = np.arange(len(stages), dtype=np.int64) * EPOCH_SAMPLES
    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


# ── DREAMS (Subjects + Patients) ─────────────────────────────────────

def iter_dreams(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all whole-night EDF + AASM hypnogram pairs in DREAMS."""
    ext_dir = ds_path / "extracted"
    if not ext_dir.exists():
        ext_dir = ds_path
    pairs = []
    # Subjects: subject1.edf + HypnogramAASM_subject1.txt
    for edf in sorted(ext_dir.glob("subject*.edf")):
        num = edf.stem.replace("subject", "")
        hyp = ext_dir / f"HypnogramAASM_subject{num}.txt"
        if hyp.exists():
            pairs.append((f"subj{num}", edf, hyp))
    # Patients: patient1.edf + HypnogramAASM_patient1.txt
    for edf in sorted(ext_dir.glob("patient*.edf")):
        num = edf.stem.replace("patient", "")
        hyp = ext_dir / f"HypnogramAASM_patient{num}.txt"
        if hyp.exists():
            pairs.append((f"pat{num}", edf, hyp))
    return pairs


def read_dreams_recording(edf_path: Path, hyp_path: Path,
                          channel: Optional[str] = None,
                          ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one DREAMS recording (EDF + AASM hypnogram at 5-sec resolution)."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    if channel is None:
        for ch in ["CZ-A1", "CZ2-A1", "C3-A1", "FP1-A2", "O1-A2"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            eeg_chs = [c for c in raw.ch_names if any(
                e in c.upper() for e in ("CZ", "C3", "C4", "FP", "F3", "F4", "O1", "O2"))]
            channel = eeg_chs[0] if eeg_chs else raw.ch_names[0]

    data = preprocess_raw(raw, channel)

    # Parse AASM hypnogram: first line is header "[HypnogramAASM]", then one int per 5-sec epoch
    lines = hyp_path.read_text().strip().splitlines()
    five_sec_stages = []
    for line in lines:
        line = line.strip()
        if line.startswith("[") or not line:
            continue
        five_sec_stages.append(int(line))
    five_sec_stages = np.array(five_sec_stages, dtype=int)

    # Aggregate 6 x 5-sec epochs -> 1 x 30-sec epoch via majority vote
    n_30s = len(five_sec_stages) // 6
    chunks = five_sec_stages[:n_30s * 6].reshape(n_30s, 6)
    stages = np.full(n_30s, -1, dtype=np.int8)
    for code_5s, aasm in DREAMS_AASM_MAP.items():
        counts = np.sum(chunks == code_5s, axis=1)
        stages[counts > 3] = aasm  # majority = >3 of 6
    # Handle ties
    ambiguous = stages == -1
    if ambiguous.any():
        for i in np.where(ambiguous)[0]:
            vals, cnts = np.unique(chunks[i], return_counts=True)
            mapped = DREAMS_AASM_MAP.get(int(vals[cnts.argmax()]), -1)
            stages[i] = mapped

    onsets_samples = np.arange(n_30s, dtype=np.int64) * EPOCH_SAMPLES
    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


# ── ds005185 (EESM19, in-ear EEG, multi-session) ─────────────────────

# Same stage encoding as EESM17: 1=W, 2=REM, 3=N1, 4=N2, 5=N3


def iter_eesm19(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all (subject, session) pairs with scoring annotations."""
    pairs = []
    for sub in sorted(ds_path.glob("sub-*")):
        for ses in sorted(sub.glob("ses-*")):
            eeg_dir = ses / "eeg"
            if not eeg_dir.exists():
                continue
            scoring = list(eeg_dir.glob("*scoring1_events.tsv"))
            set_files = list(eeg_dir.glob("*task-sleep*acq-PSG*eeg.set"))
            if scoring and set_files:
                rec_id = f"{sub.name}_{ses.name}"
                pairs.append((rec_id, set_files[0], scoring[0]))
    return pairs


def read_eesm19_recording(set_path: Path, scoring_path: Path,
                          channel: Optional[str] = None,
                          ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one EESM19 PSG session (contains ear + scalp channels)."""
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    if channel is None:
        for ch in ["ELA", "ERA", "ELB", "ERB", "C4", "C3"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            channel = raw.ch_names[0]

    data = preprocess_raw(raw, channel)

    events_df = pd.read_csv(scoring_path, sep="\t")
    onsets_samples = (np.asarray(events_df["onset"]) * TARGET_SFREQ).astype(np.int64)
    stages = np.array([EESM_STAGE_TO_AASM.get(int(s), -1)
                       for s in events_df["Scoring1"].values], dtype=np.int8)

    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


# ── DCSM (Danish Center for Sleep Medicine) ──────────────────────────

DCSM_STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}


def iter_dcsm(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all DCSM recordings with EDF + hypnogram."""
    dcsm_dir = ds_path / "data" / "sleep" / "DCSM"
    if not dcsm_dir.exists():
        dcsm_dir = ds_path
    pairs = []
    for sub_dir in sorted(dcsm_dir.iterdir()):
        if not sub_dir.is_dir():
            continue
        edf = sub_dir / "psg.edf"
        hyp = sub_dir / "hypnogram.ids"
        if edf.exists() and hyp.exists():
            pairs.append((sub_dir.name, edf, hyp))
    return pairs


def read_dcsm_recording(edf_path: Path, hyp_path: Path,
                        channel: Optional[str] = None,
                        ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one DCSM recording (EDF + run-length hypnogram in seconds)."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    if channel is None:
        for ch in ["C4-M1", "C3-M2", "F4-M1", "F3-M2", "O2-M1"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            eeg_chs = [c for c in raw.ch_names if any(
                e in c.upper() for e in ("C3", "C4", "F3", "F4", "O1", "O2"))]
            channel = eeg_chs[0] if eeg_chs else raw.ch_names[0]

    data = preprocess_raw(raw, channel)

    # Parse run-length hypnogram: onset_sec, duration_sec, stage
    lines = hyp_path.read_text().strip().splitlines()
    onsets_list: list[int] = []
    stages_list: list[int] = []
    for line in lines:
        parts = line.split(",")
        onset_sec = int(parts[0])
        dur_sec = int(parts[1])
        stage_str = parts[2].strip()
        aasm = DCSM_STAGE_MAP.get(stage_str, -1)
        # Expand run-length into 30s epochs
        for t in range(0, dur_sec, EPOCH_SEC):
            onsets_list.append(int((onset_sec + t) * TARGET_SFREQ))
            stages_list.append(aasm)

    onsets = np.array(onsets_list, dtype=np.int64)
    stages = np.array(stages_list, dtype=np.int8)
    epochs, labels = extract_epochs(data, stages, onsets)
    return epochs, labels, channel

"""Dataset-specific readers for sleep EEG recordings.

Each reader takes file path(s) and returns (epochs, labels, channel_name).
All readers share preprocess_raw() and extract_epochs() from data_pipeline.

Supported datasets:
    BIDS: ds005555 (BOAS), ds004348 (EESM17), ds005207 (cEEGrid), ds005178 (EESM23)
    External: Sleep-EDF, CAP, HMC, Dreem DOD-H/DOD-O, DREAMT
"""
from pathlib import Path
from typing import Optional

import mne
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from config import DATASETS, EPOCH_SAMPLES, EPOCH_SEC, TARGET_SFREQ
from data_pipeline import preprocess_raw, extract_epochs

_SECONDS_PER_DAY = 86400

# ── Stage encoding maps ──────────────────────────────────────────────

# EESM17 / cEEGrid: 1=Wake, 2=REM, 3=N1, 4=N2, 5=N3, 6+=artifact
EESM_STAGE_TO_AASM = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}

# EESM23 (ds005178): text labels in scoring_events.tsv
EESM23_STAGE_MAP = {"Wake": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4}

# DREAMT 2.0: W=Wake, N1, N2, N3, R=REM, P=Preparation (pre-PSG, discard)
DREAMT_STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}

# Sleep-EDF hypnogram annotations
SLEEP_EDF_STAGE_MAP = {
    "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2,
    "Sleep stage 3": 3, "Sleep stage 4": 3, "Sleep stage R": 4,
}

# CAP: R&K text annotations from .txt files
CAP_TXT_STAGE_MAP = {
    "SLEEP-S0": 0, "SLEEP-S1": 1, "SLEEP-S2": 2,
    "SLEEP-S3": 3, "SLEEP-S4": 3, "SLEEP-REM": 4,
}

# HMC: AASM text labels from scoring CSV
HMC_ANNOT_MAP = {
    "Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2,
    "Sleep stage N3": 3, "Sleep stage R": 4,
}


# ── BIDS readers (ds005555, ds004348, ds005207) ─────────────────────

def read_boas_subject(sub_dir: Path, acq: str = "headband",
                      channel: Optional[str] = None,
                      ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one BOAS (ds005555) subject."""
    sub_id = sub_dir.name
    eeg_dir = sub_dir / "eeg"

    edf_path = eeg_dir / f"{sub_id}_task-Sleep_acq-{acq}_eeg.edf"
    events_path = eeg_dir / f"{sub_id}_task-Sleep_acq-{acq}_events.tsv"
    channels_path = eeg_dir / f"{sub_id}_task-Sleep_acq-{acq}_channels.tsv"

    if not edf_path.exists():
        raise FileNotFoundError(f"EDF not found: {edf_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"Events not found: {events_path}")

    ch_df = pd.read_csv(channels_path, sep="\t")
    eeg_channels = ch_df[ch_df["type"] == "EEG"]["name"].tolist()

    if channel is None:
        defaults = {"headband": ["HB_1", "HB_2"], "psg": ["PSG_C4", "PSG_C3", "PSG_F4"]}
        for ch in defaults.get(acq, []):
            if ch in eeg_channels:
                channel = ch
                break
        if channel is None and eeg_channels:
            channel = eeg_channels[0]
        if channel is None:
            raise ValueError(f"No EEG channels found in {channels_path}")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    orig_sfreq = raw.info["sfreq"]
    data = preprocess_raw(raw, channel)

    events_df = pd.read_csv(events_path, sep="\t")
    label_col = "stage_hum" if "stage_hum" in events_df.columns else "stage_ai"

    scale = TARGET_SFREQ / orig_sfreq
    onsets = (events_df["begsample"].values * scale).astype(np.int64)
    stages = events_df[label_col].values.astype(np.int8)

    epochs, labels = extract_epochs(data, stages, onsets)
    return epochs, labels, channel


def read_eesm17_subject(sub_dir: Path, acq: str = "ear",
                        channel: Optional[str] = None,
                        ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one EESM17 (ds004348) in-ear EEG subject."""
    sub_id = sub_dir.name
    ses_dir = sub_dir / "ses-001" / "eeg"
    if not ses_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {ses_dir}")

    set_files = list(ses_dir.glob(f"{sub_id}_ses-001_task-sleep*.set"))
    if not set_files:
        raise FileNotFoundError(f"No sleep .set file in {ses_dir}")
    events_files = list(ses_dir.glob("*task-sleep*scoring*events.tsv"))
    if not events_files:
        raise FileNotFoundError(f"No scoring events.tsv in {ses_dir}")

    ch_files = list(ses_dir.glob("*task-sleep_channels.tsv"))
    eeg_channels = (pd.read_csv(ch_files[0], sep="\t")
                    .query("type == 'EEG'")["name"].tolist() if ch_files else [])

    if channel is None:
        for ch in ["ELA", "ERA", "ELE", "ERE", "ELI", "ERI"]:
            if ch in eeg_channels:
                channel = ch
                break
        if channel is None:
            channel = eeg_channels[0] if eeg_channels else None
        if channel is None:
            raise ValueError(f"No EEG channels found for {sub_id}")

    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
    data = preprocess_raw(raw, channel)

    events_df = pd.read_csv(events_files[0], sep="\t")
    onsets_samples = (np.asarray(events_df["onset"]) * TARGET_SFREQ).astype(np.int64)
    stages = np.array([EESM_STAGE_TO_AASM.get(int(s), -1)
                       for s in events_df["staging"].values], dtype=np.int8)

    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


def read_ceegrid_subject(sub_dir: Path, acq: str = "ceegrid",
                         channel: Optional[str] = None,
                         ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one Surrey cEEGrid (ds005207) around-ear subject."""
    sub_id = sub_dir.name
    ses_dir = sub_dir / "ses-001" / "eeg"
    if not ses_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {ses_dir}")

    acq_tag = "cEEGrid" if acq == "ceegrid" else "PSG"
    set_files = list(ses_dir.glob(f"*acq-{acq_tag}_eeg.set"))
    if not set_files:
        raise FileNotFoundError(f"No {acq_tag} .set file in {ses_dir}")

    scoring_tag = f"{acq_tag}Scoring"
    events_files = list(ses_dir.glob(f"*acq-{scoring_tag}_events.tsv"))
    if not events_files:
        raise FileNotFoundError(f"No {scoring_tag} events.tsv in {ses_dir}")

    ch_files = list(ses_dir.glob(f"*acq-{acq_tag}_channels.tsv"))
    eeg_channels = (pd.read_csv(ch_files[0], sep="\t")
                    .query("type == 'EEG'")["name"].tolist() if ch_files else [])

    if channel is None:
        for ch in ["L4", "R4", "L5", "R5", "L3", "R6", "L6"]:
            if ch in eeg_channels:
                channel = ch
                break
        if channel is None:
            channel = eeg_channels[0] if eeg_channels else None
        if channel is None:
            raise ValueError(f"No EEG channels found for {sub_id}")

    raw = mne.io.read_raw_eeglab(str(set_files[0]), preload=True, verbose=False)
    data = preprocess_raw(raw, channel)

    events_df = pd.read_csv(events_files[0], sep="\t")
    onsets_samples = (np.asarray(events_df["onset"]) * TARGET_SFREQ).astype(np.int64)
    stages = np.array([EESM_STAGE_TO_AASM.get(int(s), -1)
                       for s in events_df["staging"].values], dtype=np.int8)

    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


def iter_eesm23(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all scored sessions in EESM23 (ds005178). Returns (name, set_path, scoring_path)."""
    pairs = []
    for sub_dir in sorted(ds_path.glob("sub-*")):
        for scoring_path in sorted(sub_dir.rglob("*acq-scoring_events.tsv")):
            ses_dir = scoring_path.parent
            set_files = list(ses_dir.glob("*acq-earEEG_eeg.set"))
            if set_files:
                name = f"{sub_dir.name}_{scoring_path.parent.parent.name}"
                pairs.append((name, set_files[0], scoring_path))
    return pairs


def read_eesm23_recording(set_path: Path, scoring_path: Path,
                          channel: Optional[str] = None,
                          ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one EESM23 ear-EEG session."""
    raw = mne.io.read_raw_eeglab(str(set_path), preload=True, verbose=False)

    if channel is None:
        for ch in ["LB", "RB", "LT", "RT"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            channel = raw.ch_names[0]

    assert channel is not None
    data = preprocess_raw(raw, channel)

    events_df = pd.read_csv(scoring_path, sep="\t")
    onsets_samples = (events_df["onset"].values * TARGET_SFREQ).astype(np.int64)
    stages = np.array([EESM23_STAGE_MAP.get(str(s), -1)
                       for s in events_df["scoring"].values], dtype=np.int8)

    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


def iter_dreem(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all H5 recordings in a Dreem DOD directory."""
    # Dreem has dodh/ or dodo/ subdirectory
    h5_dir = ds_path
    for sub in ["dodh", "dodo"]:
        candidate = ds_path / sub
        if candidate.exists():
            h5_dir = candidate
            break
    pairs = []
    for h5_file in sorted(h5_dir.glob("*.h5")):
        pairs.append((h5_file.stem[:8], h5_file, h5_file))  # dummy second path
    return pairs


def read_dreem_recording(h5_path: Path, _unused: Path,
                         channel: Optional[str] = None,
                         ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one Dreem DOD-H or DOD-O recording from H5."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        # Read hypnogram
        hyp = np.asarray(f["hypnogram"][:], dtype=np.int64)

        # Pick EEG channel - prefer frontal for headband similarity
        eeg_channels = list(f["signals/eeg"].keys())
        if channel is None:
            for ch in ["F3_F4", "FP1_F3", "FP2_F4", "C3_M2", "C4_M1"]:
                if ch in eeg_channels:
                    channel = ch
                    break
            if channel is None:
                channel = eeg_channels[0]

        assert channel is not None
        signal = np.asarray(f[f"signals/eeg/{channel}"][:], dtype=np.float64)

    # Create MNE Raw object from the signal array
    info = mne.create_info([channel], sfreq=250.0, ch_types=["eeg"])
    raw = mne.io.RawArray(signal.reshape(1, -1) * 1e-6, info, verbose=False)  # uV -> V
    data = preprocess_raw(raw, channel)

    # Sequential epochs from hypnogram
    onsets_samples = np.arange(len(hyp), dtype=np.int64) * EPOCH_SAMPLES
    # Dreem hypnogram is already AASM (0=W,1=N1,2=N2,3=N3,4=REM)
    stages = np.where((hyp >= 0) & (hyp <= 4), hyp, -1).astype(np.int8)

    epochs, labels = extract_epochs(data, stages, onsets_samples)
    return epochs, labels, channel


def iter_dreamt(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all subject CSVs in DREAMT 2.0."""
    data_dir = ds_path / "data_100Hz"
    if not data_dir.exists():
        data_dir = ds_path
    pairs = []
    for csv_file in sorted(data_dir.glob("S*_PSG_df_updated.csv")):
        name = csv_file.stem.split("_")[0]  # e.g. "S002"
        pairs.append((name, csv_file, csv_file))  # dummy second path
    return pairs


def read_dreamt_recording(csv_path: Path, _unused: Path,
                          channel: Optional[str] = None,
                          ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one DREAMT 2.0 subject from CSV (100Hz, per-sample labels)."""
    # Read only the EEG channel + stage column to save memory
    if channel is None:
        channel = "C4-M1"

    df = pd.read_csv(csv_path, usecols=[channel, "Sleep_Stage"])
    signal = df[channel].values.astype(np.float64)
    stage_per_sample = df["Sleep_Stage"].to_numpy()

    # Create MNE Raw and preprocess
    info = mne.create_info([channel], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(signal.reshape(1, -1), info, verbose=False)  # already in volts
    data = preprocess_raw(raw, channel)

    # Extract per-epoch stage: majority vote over each 30s window at original 100Hz
    orig_epoch_samples = 100 * EPOCH_SEC  # 3000 samples at 100Hz
    n_epochs = min(len(data) // EPOCH_SAMPLES, len(stage_per_sample) // orig_epoch_samples)

    # Reshape into (n_epochs, 3000) for vectorized majority vote
    stage_chunks = stage_per_sample[:n_epochs * orig_epoch_samples].reshape(n_epochs, -1)
    # Map string labels to AASM integers via majority vote
    stage_map_arr = np.full(n_epochs, -1, dtype=np.int8)
    for label, code in DREAMT_STAGE_MAP.items():
        # For each epoch, count how many samples match this label
        match_counts = np.sum(stage_chunks == label, axis=1)
        # Where this label is the majority (more than half), assign it
        stage_map_arr[match_counts > orig_epoch_samples // 2] = code
    # Handle remaining (no clear majority) with per-epoch mode
    ambiguous = stage_map_arr == -1
    if ambiguous.any():
        for i in np.where(ambiguous)[0]:
            vals, counts = np.unique(stage_chunks[i], return_counts=True)
            stage_map_arr[i] = DREAMT_STAGE_MAP.get(str(vals[counts.argmax()]), -1)

    onsets_samples = np.arange(n_epochs, dtype=np.int64) * EPOCH_SAMPLES
    epochs, labels = extract_epochs(data, stage_map_arr, onsets_samples)
    return epochs, labels, channel


# ── External dataset iterators + readers ─────────────────────────────

def iter_sleep_edf(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all PSG+Hypnogram pairs in Sleep-EDF."""
    data_dir = ds_path / "physionet-sleep-data"
    if not data_dir.exists():
        data_dir = ds_path
    psg_files = sorted(data_dir.glob("*PSG.edf"))
    pairs = []
    for psg in psg_files:
        base6 = psg.name[:6]
        hyps = list(data_dir.glob(f"{base6}*Hypnogram.edf"))
        if hyps:
            pairs.append((base6, psg, hyps[0]))
    return pairs


def read_sleep_edf_recording(psg_path: Path, hyp_path: Path,
                             channel: str = "EEG Fpz-Cz",
                             ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one Sleep-EDF PSG+Hypnogram pair."""
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    data = preprocess_raw(raw, channel)

    annots = mne.read_annotations(str(hyp_path))
    onsets_samples, stages = [], []
    for onset, duration, desc in zip(annots.onset, annots.duration, annots.description):
        stage = SLEEP_EDF_STAGE_MAP.get(desc, -1)
        for j in range(int(duration) // EPOCH_SEC):
            onsets_samples.append(int((onset + j * EPOCH_SEC) * TARGET_SFREQ))
            stages.append(stage)

    epochs, labels = extract_epochs(
        data, np.array(stages, np.int8), np.array(onsets_samples, np.int64))
    return epochs, labels, channel


def iter_cap(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all EDF+.txt pairs in CAP database."""
    pairs = []
    for edf in sorted(ds_path.glob("*.edf")):
        txt_file = edf.with_suffix(".txt")
        if txt_file.exists():
            pairs.append((edf.stem, edf, txt_file))
    return pairs


def _parse_cap_clock(time_str: str) -> int:
    """Parse CAP clock time 'hh.mm.ss' or 'hh:mm:ss' to seconds since midnight."""
    sep = ":" if ":" in time_str else "."
    parts = time_str.split(sep)
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def read_cap_recording(edf_path: Path, txt_path: Path,
                       channel: Optional[str] = None,
                       ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one CAP Sleep Database recording using .txt annotation file."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    if channel is None:
        for ch in ["C4-A1", "C3-A2", "F4-A1", "F3-A2", "O2-A1", "O1-A2"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            eeg_chs = [c for c in raw.ch_names if "-" in c and
                       c.split("-")[0] in ("C3", "C4", "F3", "F4", "O1", "O2")]
            channel = eeg_chs[0] if eeg_chs else raw.ch_names[0]

    assert channel is not None
    data = preprocess_raw(raw, channel)
    edf_duration_sec = len(data) / TARGET_SFREQ

    # EDF recording start time (seconds since midnight)
    meas_date = raw.info["meas_date"]
    edf_start_sec = meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second

    # Parse .txt annotations
    lines = txt_path.read_text(encoding="latin-1").strip().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith("Sleep Stage\t"):
            header_idx = i
            break
    if header_idx is None:
        return np.empty((0, EPOCH_SAMPLES), np.float32), np.empty(0, np.int8), channel

    # Detect column layout: 6-col (with Position) vs 5-col (without)
    header_cols = lines[header_idx].split("\t")
    has_position = any("osition" in c for c in header_cols)
    time_col = 2 if has_position else 1
    event_col = 3 if has_position else 2

    onsets_samples, stages = [], []
    for line in lines[header_idx + 1:]:
        parts = line.split("\t")
        if len(parts) <= event_col:
            continue
        event = parts[event_col]
        if event not in CAP_TXT_STAGE_MAP:
            continue
        # Clock time -> relative offset from EDF start
        clock_sec = _parse_cap_clock(parts[time_col])
        offset_sec = clock_sec - edf_start_sec
        if offset_sec < 0:
            offset_sec += _SECONDS_PER_DAY  # midnight rollover
        if offset_sec + EPOCH_SEC > edf_duration_sec:
            break  # past end of EDF recording
        onsets_samples.append(int(offset_sec * TARGET_SFREQ))
        stages.append(CAP_TXT_STAGE_MAP[event])

    epochs, labels = extract_epochs(
        data, np.array(stages, np.int8), np.array(onsets_samples, np.int64))
    return epochs, labels, channel


def iter_hmc(ds_path: Path) -> list[tuple[str, Path, Path]]:
    """Find all EDF+scoring pairs in HMC database."""
    rec_dir = ds_path / "recordings"
    if not rec_dir.exists():
        rec_dir = ds_path
    pairs = []
    for edf in sorted(rec_dir.glob("SN*.edf")):
        if "_sleepscoring" in edf.name:
            continue
        txt = edf.with_name(edf.stem + "_sleepscoring.txt")
        if txt.exists():
            pairs.append((edf.stem, edf, txt))
    return pairs


def read_hmc_recording(edf_path: Path, txt_path: Path,
                       channel: Optional[str] = None,
                       ) -> tuple[NDArray[np.float32], NDArray[np.int8], str]:
    """Read one HMC recording."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    if channel is None:
        for ch in ["C4-M1", "F4-M1", "O2-M1", "C3-M2"]:
            if ch in raw.ch_names:
                channel = ch
                break
        if channel is None:
            channel = raw.ch_names[0]

    assert channel is not None
    data = preprocess_raw(raw, channel)

    # HMC format: CSV with header, columns: Date, Time, Recording onset, Duration, Annotation, ...
    lines = txt_path.read_text().strip().splitlines()
    onsets_samples, stages = [], []
    for line in lines[1:]:  # skip header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        annot = parts[4]
        if annot not in HMC_ANNOT_MAP:
            continue
        onset_sec = float(parts[2])
        onsets_samples.append(int(onset_sec * TARGET_SFREQ))
        stages.append(HMC_ANNOT_MAP[annot])

    epochs, labels = extract_epochs(
        data, np.array(stages, np.int8), np.array(onsets_samples, np.int64))
    return epochs, labels, channel

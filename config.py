"""Project configuration — paths, constants, dataset registry."""
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DATA_ROOT = Path("D:/sleep_datasets")
CACHE_DIR = DATA_ROOT / "_cache"  # preprocessed HDF5 files
ENCODER_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "encoder"
MODEL_CKPT_DIR = PROJECT_ROOT / "checkpoints" / "sleep_model"

# ── EEG Constants ──────────────────────────────────────────────────────
TARGET_SFREQ = 128  # Hz — sufficient for sleep (max relevant ~45Hz)
EPOCH_SEC = 30      # standard sleep staging epoch
EPOCH_SAMPLES = TARGET_SFREQ * EPOCH_SEC  # 3840
LOWPASS_HZ = 45.0   # nothing above this matters for sleep
HIGHPASS_HZ = 0.3   # remove DC drift

# ── Sleep Stage Labels (AASM) ─────────────────────────────────────────
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
NUM_STAGES = 5

# ── Dataset Registry ───────────────────────────────────────────────────
# Each entry: directory name → metadata for the data pipeline
DATASETS = {
    "ds005555": {
        "path": DATA_ROOT / "ds005555-download",
        "name": "BOAS (Bitbrain)",
        "format": "bids_edf",
        "sfreq": 256,
        "device": "headband",
        "has_labels": True,
        "priority_channels": ["HB_1", "HB_2"],  # forehead AF7/AF8
        "label_source": "events_tsv",  # stage_hum column
    },
    "ds004348": {
        "path": DATA_ROOT / "ds004348-download",
        "name": "EESM17 (Ear-EEG)",
        "format": "bids_set",
        "sfreq": 200,
        "device": "in_ear",
        "has_labels": True,
        "priority_channels": ["ELA", "ERA"],  # left/right ear
        "label_source": "events_tsv",
    },
    "ds005207": {
        "path": DATA_ROOT / "ds005207-download",
        "name": "Surrey cEEGrid",
        "format": "bids_set",
        "sfreq": 250,
        "device": "around_ear",
        "has_labels": True,
        "priority_channels": ["L4", "R4"],  # cEEGrid near-ear
        "label_source": "sleep_profile_txt",
    },
    "ds005178": {
        "path": DATA_ROOT / "ds005178-download",
        "name": "EESM23 (Ear-EEG)",
        "format": "bids_set",
        "sfreq": 250,
        "device": "in_ear",
        "has_labels": True,
        "priority_channels": ["LB", "RB", "LT", "RT"],  # left/right back/top ear
        "label_source": "scoring_events_tsv",
    },
    "sleep_edf": {
        "path": DATA_ROOT / "sleep-edf-expanded",
        "name": "Sleep-EDF Expanded",
        "format": "physionet_edf",
        "sfreq": 100,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["EEG Fpz-Cz", "EEG Pz-Oz"],
        "label_source": "hypnogram_edf",
    },
    "dreem_dod_h": {
        "path": DATA_ROOT / "dreem-dod-h",
        "name": "Dreem DOD-H (Healthy)",
        "format": "dreem_h5",
        "sfreq": 250,
        "device": "headband",
        "has_labels": True,
        "priority_channels": ["F3_F4", "FP1_F3", "FP2_F4"],
        "label_source": "dreem_h5",
    },
    "dreem_dod_o": {
        "path": DATA_ROOT / "dreem-dod-o",
        "name": "Dreem DOD-O (OSA)",
        "format": "dreem_h5",
        "sfreq": 250,
        "device": "headband",
        "has_labels": True,
        "priority_channels": ["F3_F4", "FP1_F3", "FP2_F4"],
        "label_source": "dreem_h5",
    },
    "cap_slpdb": {
        "path": DATA_ROOT / "CAP Sleep Database",
        "name": "CAP Sleep Database",
        "format": "physionet_edf",
        "sfreq": None,  # varies (256/512 Hz)
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["C4-A1", "C3-A2", "F4-A1"],
        "label_source": "txt_file",
    },
    "hmc": {
        "path": DATA_ROOT / "hmc-sleep-staging",
        "name": "HMC Sleep Staging",
        "format": "physionet_edf",
        "sfreq": 256,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["C4-M1", "F4-M1", "O2-M1", "C3-M2"],
        "label_source": "sleepscoring_txt",
    },
    "dreamt": {
        "path": DATA_ROOT / "dreamt_2.0",
        "name": "DREAMT 2.0",
        "format": "dreamt_csv",
        "sfreq": 100,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["C4-M1", "F4-M1", "O2-M1"],
        "label_source": "csv_column",
    },
    "ysyw": {
        "path": DATA_ROOT / "You Snooze You Win",
        "name": "You Snooze You Win (PhysioNet 2018)",
        "format": "ysyw_mat",
        "sfreq": 200,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["C4-M1", "C3-M2", "F4-M1"],
        "label_source": "arousal_mat",
    },
    "ds006695": {
        "path": DATA_ROOT / "ds006695-download",
        "name": "UCSD Forehead Patch",
        "format": "bids_set",
        "sfreq": 500,
        "device": "forehead",
        "has_labels": True,
        "priority_channels": ["FP1-AFz", "FP2-AFz"],
        "label_source": "visual_hypnogram",
    },
    "svuh": {
        "path": DATA_ROOT / "svuh",
        "name": "SVUH/UCD Sleep Apnea",
        "format": "physionet_rec",
        "sfreq": 128,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["C3A2", "C4A1"],
        "label_source": "stage_txt",
    },
    "dreams": {
        "path": DATA_ROOT / "dreams",
        "name": "DREAMS (Subjects+Patients)",
        "format": "dreams_edf",
        "sfreq": 200,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["CZ-A1", "CZ2-A1", "C3-A1"],
        "label_source": "aasm_txt",
    },
    "ds005185": {
        "path": DATA_ROOT / "ds005185-download",
        "name": "EESM19 (Ear-EEG, multi-session)",
        "format": "bids_set",
        "sfreq": 500,
        "device": "in_ear",
        "has_labels": True,
        "priority_channels": ["ELA", "ERA", "ELB", "ERB"],
        "label_source": "scoring1_events_tsv",
    },
    "dcsm": {
        "path": DATA_ROOT / "dcsm_dataset",
        "name": "DCSM (Danish Center for Sleep Medicine)",
        "format": "dcsm_edf",
        "sfreq": 256,
        "device": "scalp",
        "has_labels": True,
        "priority_channels": ["C4-M1", "C3-M2", "F4-M1"],
        "label_source": "hypnogram_ids",
    },
}


# ── Checkpoint utilities ──────────────────────────────────────────────

_LOWER_IS_BETTER = {"val_loss"}


def _extract_metric_value(stem: str, metric: str) -> float:
    """Extract numeric metric value from checkpoint filename stem.

    Handles Lightning's version suffix: ``epoch=3-val_kappa=0.775-v1``.
    """
    split = stem.split(f"{metric}=")
    if len(split) < 2:
        return float("nan")
    val_str = split[1]
    # Lightning appends -vN on name collision; strip it
    parts = val_str.rsplit("-v", 1)
    if len(parts) == 2 and parts[1].isdigit():
        val_str = parts[0]
    return float(val_str)


def find_best_checkpoint(ckpt_dir: Path, metrics: list[str]) -> str:
    """Find best checkpoint by metric value in filename.

    Searches for files matching ``epoch=*-{metric}=*.ckpt`` and returns the
    one with the best metric value (highest for accuracy-like metrics,
    lowest for loss metrics).  *metrics* is tried in order; the first
    metric with matching files wins.

    Raises:
        FileNotFoundError: if no matching checkpoints exist.
    """
    for metric in metrics:
        ckpts = sorted(
            ckpt_dir.glob(f"epoch=*-{metric}=*.ckpt"),
            key=lambda p, m=metric: _extract_metric_value(p.stem, m),
            reverse=(metric not in _LOWER_IS_BETTER),
        )
        if ckpts:
            return str(ckpts[0])
    raise FileNotFoundError(f"No checkpoints matching {metrics} in {ckpt_dir}")

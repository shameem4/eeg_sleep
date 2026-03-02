"""Run inference on any BIDS EEG dataset. Produces predictions, metrics, and plots.

Reads EDF/SET files from a BIDS-formatted directory, preprocesses, runs the
trained sleep stage model, and optionally compares with ground truth labels
if events.tsv files are present.

Usage:
    # Predict on a BIDS dataset (no labels)
    python inference.py /path/to/bids_dataset

    # With specific channel and checkpoint
    python inference.py /path/to/bids_dataset --channel C4-M1 --checkpoint path/to/ckpt

    # With custom output directory
    python inference.py /path/to/bids_dataset --output-dir results/my_dataset
"""
import argparse
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import (
    classification_report, cohen_kappa_score, confusion_matrix, f1_score,
)

from config import (
    EPOCH_SAMPLES, EPOCH_SEC, MODEL_CKPT_DIR, NUM_STAGES, STAGE_NAMES,
    TARGET_SFREQ, find_best_checkpoint,
)
from data_pipeline import preprocess_raw, validate_epochs

matplotlib.use("Agg")
mne.set_log_level("ERROR")

# -- Style -------------------------------------------------------------------

STAGE_COLORS = {
    "W": "#e74c3c", "N1": "#f39c12", "N2": "#3498db",
    "N3": "#2c3e50", "REM": "#27ae60",
}
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f9fa",
    "axes.grid": True, "grid.alpha": 0.3, "font.family": "sans-serif",
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "bold",
    "figure.dpi": 150,
})

# -- BIDS stage maps ---------------------------------------------------------

# Common BIDS/AASM stage label mappings (covers most public datasets)
STAGE_MAPS = {
    # AASM text (e.g., HMC, PhysioNet)
    "Sleep stage W": 0, "Sleep stage N1": 1, "Sleep stage N2": 2,
    "Sleep stage N3": 3, "Sleep stage R": 4,
    # R&K / Sleep-EDF
    "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    # Short text labels
    "W": 0, "N1": 1, "N2": 2, "N3": 3, "REM": 4, "R": 4,
    "Wake": 0, "NREM1": 1, "NREM2": 2, "NREM3": 3,
    # AASM numeric (standard: 0=W, 1=N1, 2=N2, 3=N3, 4=REM)
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 4,
}

# EESM numeric encoding (used by EESM17, EESM19, cEEGrid): 1=W, 2=REM, 3=N1, 4=N2, 5=N3
EESM_STAGE_MAP = {1: 0, 2: 4, 3: 1, 4: 2, 5: 3}
# Columns that use EESM encoding
EESM_COLUMNS = {"scoring1", "scoring2", "staging"}


# -- BIDS reading ------------------------------------------------------------

def find_eeg_files(bids_dir: Path) -> list[dict]:
    """Find all EEG recording files in a BIDS directory.

    Searches for EDF and SET files under sub-*/[ses-*/]eeg/.
    Returns list of dicts with keys: sub_id, eeg_path, events_path (or None),
    channels_path (or None).
    """
    recordings = []
    for eeg_file in sorted(bids_dir.rglob("*.edf")) + sorted(bids_dir.rglob("*.set")):
        # Must be under an eeg/ directory
        if eeg_file.parent.name != "eeg":
            continue
        # Skip non-sleep files
        stem_lower = eeg_file.stem.lower()
        if "hypnogram" in stem_lower or "annotation" in stem_lower:
            continue
        # Skip non-sleep tasks (e.g., ASSR auditory stimulation)
        if "_task-" in stem_lower and "sleep" not in stem_lower:
            continue

        eeg_dir = eeg_file.parent
        stem_prefix = eeg_file.stem.rsplit("_eeg", 1)[0] if "_eeg" in eeg_file.stem else eeg_file.stem

        # Find matching events.tsv (try exact prefix first, then scoring variants)
        events_candidates = sorted(eeg_dir.glob(f"{stem_prefix}*events.tsv"))
        if not events_candidates:
            # Try scoring events (EESM19: acq-scoring1_events.tsv alongside acq-PSG_eeg.set)
            task_prefix = stem_prefix.rsplit("_acq-", 1)[0] if "_acq-" in stem_prefix else stem_prefix
            events_candidates = sorted(eeg_dir.glob(f"{task_prefix}*scoring*events.tsv"))
        events_path = events_candidates[0] if events_candidates else None

        # Find matching channels.tsv
        ch_candidates = sorted(eeg_dir.glob(f"{stem_prefix}*channels.tsv"))
        channels_path = ch_candidates[0] if ch_candidates else None

        # Extract subject ID (include session if present)
        parts = eeg_file.parts
        sub_id = next((p for p in parts if p.startswith("sub-")), eeg_file.stem)
        ses_id = next((p for p in parts if p.startswith("ses-")), None)
        if ses_id:
            sub_id = f"{sub_id}_{ses_id}"

        recordings.append({
            "sub_id": sub_id,
            "eeg_path": eeg_file,
            "events_path": events_path,
            "channels_path": channels_path,
        })

    return recordings


def pick_channel(raw: mne.io.BaseRaw, channels_path: Optional[Path],
                 requested: Optional[str]) -> str:
    """Select the best EEG channel from a recording.

    Priority: requested > channels.tsv EEG list > raw channel names.
    """
    available = raw.ch_names

    if requested and requested in available:
        return requested

    # Try channels.tsv for EEG-typed channels
    eeg_channels = []
    if channels_path and channels_path.exists():
        import pandas as pd
        ch_df = pd.read_csv(channels_path, sep="\t")
        if "type" in ch_df.columns and "name" in ch_df.columns:
            eeg_channels = ch_df[ch_df["type"].str.upper() == "EEG"]["name"].tolist()

    # Common priority channels (clinical + consumer)
    priority = [
        "C4-M1", "C3-M2", "C4-A1", "C3-A2", "F4-M1", "F3-M2",
        "EEG Fpz-Cz", "EEG Pz-Oz", "Fpz-Cz", "Pz-Oz",
        "FP1-AFz", "FP2-AFz", "HB_1", "HB_2",
        "ELA", "ERA", "LB", "RB", "L4", "R4",
    ]
    search_pool = eeg_channels if eeg_channels else available
    for ch in priority:
        if ch in search_pool:
            return ch

    # Fall back to first EEG channel
    if eeg_channels:
        return eeg_channels[0]
    # Fall back to first channel in raw
    return available[0]


def parse_stages_from_events(events_path: Path, orig_sfreq: float,
                             ) -> tuple[NDArray[np.int8], NDArray[np.int64]] | None:
    """Parse sleep stage labels from a BIDS events.tsv file.

    Returns (stages, onsets_samples) at TARGET_SFREQ, or None if no stage
    column found.
    """
    import pandas as pd
    df = pd.read_csv(events_path, sep="\t")

    # Identify stage column (datasets use different names)
    stage_col = None
    for candidate in ["stage_hum", "stage_ai", "Scoring1", "Scoring2",
                      "staging", "value", "trial_type", "description",
                      "stage", "sleep_stage"]:
        if candidate in df.columns:
            stage_col = candidate
            break
    if stage_col is None:
        return None

    # Identify onset column
    onset_col = None
    for candidate in ["onset", "begsample"]:
        if candidate in df.columns:
            onset_col = candidate
            break
    if onset_col is None:
        return None

    # Convert onsets to samples at TARGET_SFREQ
    onsets_raw = df[onset_col].values
    stage_values = df[stage_col].values

    # Detect if onsets are in seconds or samples
    # Heuristic: if max onset < recording length in seconds, treat as seconds
    # If "begsample" column, it's in samples at orig_sfreq
    if onset_col == "begsample":
        scale = TARGET_SFREQ / orig_sfreq
        onsets_samples = (onsets_raw * scale).astype(np.int64)
    elif onset_col == "onset":
        # onset column is typically in seconds
        onsets_samples = (onsets_raw * TARGET_SFREQ).astype(np.int64)
    else:
        onsets_samples = (onsets_raw * TARGET_SFREQ).astype(np.int64)

    # Map stage values to AASM 0-4
    use_eesm = stage_col.lower() in EESM_COLUMNS
    stages = np.full(len(stage_values), -1, dtype=np.int8)
    for i, val in enumerate(stage_values):
        key = str(val).strip()
        if use_eesm:
            # EESM numeric encoding: 1=W, 2=REM, 3=N1, 4=N2, 5=N3
            try:
                v = int(float(key))
                stages[i] = EESM_STAGE_MAP.get(v, -1)
            except (ValueError, TypeError):
                pass
        elif key in STAGE_MAPS:
            stages[i] = STAGE_MAPS[key]
        else:
            try:
                v = int(float(key))
                if 0 <= v < NUM_STAGES:
                    stages[i] = v
            except (ValueError, TypeError):
                pass

    return stages, onsets_samples


def read_recording(rec: dict, channel: Optional[str] = None,
                   ) -> tuple[NDArray[np.float32], Optional[NDArray[np.int8]], str]:
    """Read and preprocess a single recording.

    Returns:
        epochs: (N, EPOCH_SAMPLES) float32 preprocessed epochs
        labels: (N,) int8 stage labels, or None if no labels available
        channel: selected channel name
    """
    eeg_path = rec["eeg_path"]
    suffix = eeg_path.suffix.lower()

    if suffix == ".edf":
        raw = mne.io.read_raw_edf(str(eeg_path), preload=True, verbose=False)
    elif suffix == ".set":
        raw = mne.io.read_raw_eeglab(str(eeg_path), preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported format: {suffix}")

    orig_sfreq = raw.info["sfreq"]
    ch = pick_channel(raw, rec["channels_path"], channel)
    data = preprocess_raw(raw, ch)

    # Try to parse labels from events.tsv
    labels = None
    if rec["events_path"] and rec["events_path"].exists():
        result = parse_stages_from_events(rec["events_path"], orig_sfreq)
        if result is not None:
            stages, onsets_samples = result
            from data_pipeline import extract_epochs
            epochs, labels = extract_epochs(data, stages, onsets_samples)
            epochs, labels, _ = validate_epochs(epochs, labels)
            return epochs, labels, ch

    # No labels -- create sequential epochs from entire signal
    n_epochs = len(data) // EPOCH_SAMPLES
    if n_epochs == 0:
        return np.empty((0, EPOCH_SAMPLES), np.float32), None, ch

    epochs = data[:n_epochs * EPOCH_SAMPLES].reshape(n_epochs, EPOCH_SAMPLES)
    # Quality filter (use dummy labels for validate_epochs)
    dummy_labels = np.zeros(n_epochs, dtype=np.int8)
    epochs, dummy_labels, _ = validate_epochs(epochs, dummy_labels)
    return epochs, None, ch


# -- Model loading and prediction -------------------------------------------

def load_model(checkpoint: Optional[str] = None, exp_name: str = "final",
               device: torch.device = torch.device("cpu"),
               ) -> tuple:
    """Load the trained downstream model + CRF.

    Returns (module, device) where module is SleepStageModule in eval mode.
    """
    from train_model import SleepStageModule

    if checkpoint is None:
        ckpt_dir = MODEL_CKPT_DIR / exp_name
        checkpoint = find_best_checkpoint(ckpt_dir, ["val_kappa", "val_f1_macro"])

    print(f"Checkpoint: {checkpoint}")
    module = SleepStageModule.load_from_checkpoint(
        checkpoint, map_location="cpu", class_weights=torch.ones(NUM_STAGES))
    module = module.to(device).eval()
    return module


def predict_subject(epochs: NDArray[np.float32], module, device: torch.device,
                    seq_len: int = 200) -> NDArray[np.int64]:
    """Predict sleep stages for a single subject's epochs.

    Handles variable-length subjects:
    - >= seq_len: non-overlapping windows, CRF decodes each
    - < seq_len: pad with zeros, predict, trim to actual length
    """
    n = len(epochs)
    if n == 0:
        return np.array([], dtype=np.int64)

    all_preds = []
    with torch.no_grad():
        for start in range(0, n, seq_len):
            end = min(start + seq_len, n)
            chunk = epochs[start:end]
            actual_len = len(chunk)

            # Pad if shorter than seq_len
            if actual_len < seq_len:
                pad = np.zeros((seq_len - actual_len, EPOCH_SAMPLES), dtype=np.float32)
                chunk = np.concatenate([chunk, pad], axis=0)

            x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
            logits = module.model(x)  # (1, seq_len, 5)
            if module.crf is not None:
                paths = module.crf.decode(logits)
                preds = paths[0][:actual_len]
            else:
                preds = logits[0, :actual_len].argmax(dim=-1).cpu().tolist()

            all_preds.extend(preds)

    return np.array(all_preds, dtype=np.int64)


# -- Plotting ----------------------------------------------------------------

def plot_hypnogram(preds: NDArray, labels: Optional[NDArray], sub_id: str,
                   save_path: Path) -> None:
    """Plot predicted (and optionally true) hypnogram as step function."""
    n = len(preds)
    time_hours = np.arange(n) * EPOCH_SEC / 3600

    # Stage order for y-axis: W at top, N3 at bottom, REM between W and N1
    stage_order = [0, 4, 1, 2, 3]  # W, REM, N1, N2, N3
    stage_labels = ["W", "REM", "N1", "N2", "N3"]
    # Map stage indices to y positions
    y_map = {stage: pos for pos, stage in enumerate(stage_order)}

    fig, ax = plt.subplots(figsize=(14, 3.5))

    pred_y = np.array([y_map[p] for p in preds])
    if labels is not None:
        true_y = np.array([y_map[l] for l in labels])
        ax.step(time_hours, true_y, where="post", color="#2563eb",
                linewidth=1.5, alpha=0.7, label="Scored")
        ax.step(time_hours, pred_y, where="post", color="#dc2626",
                linewidth=1.2, alpha=0.8, label="Predicted")
        ax.legend(loc="upper right", fontsize=9)
    else:
        # Color each segment by predicted stage
        for i in range(n - 1):
            stage_name = STAGE_NAMES[preds[i]]
            ax.fill_between(time_hours[i:i+2], pred_y[i], pred_y[i],
                            step="post", alpha=0.3,
                            color=STAGE_COLORS[stage_name])
        ax.step(time_hours, pred_y, where="post", color="#2c3e50",
                linewidth=1.2)

    ax.set_yticks(range(len(stage_order)))
    ax.set_yticklabels(stage_labels)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Sleep Stage")
    ax.set_title(f"Hypnogram: {sub_id}")
    ax.set_xlim(0, time_hours[-1] + EPOCH_SEC / 3600)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(cm: NDArray, save_path: Path) -> None:
    """Plot confusion matrix heatmap with counts and percentages."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Normalize by row (true class) for percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sums > 0, cm / row_sums * 100, 0)

    im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="% of true class")

    # Annotate cells with count and percentage
    for i in range(NUM_STAGES):
        for j in range(NUM_STAGES):
            count = cm[i, j]
            pct = cm_pct[i, j]
            text_color = "white" if pct > 50 else "black"
            ax.text(j, i, f"{count}\n({pct:.0f}%)", ha="center", va="center",
                    fontsize=8, color=text_color)

    ax.set_xticks(range(NUM_STAGES))
    ax.set_yticks(range(NUM_STAGES))
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_yticklabels(STAGE_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_stage_distribution(preds: NDArray, labels: Optional[NDArray],
                            save_path: Path) -> None:
    """Bar chart of stage distribution (predicted vs true if available)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(NUM_STAGES)
    width = 0.35

    pred_counts = np.bincount(preds, minlength=NUM_STAGES)
    pred_pct = pred_counts / pred_counts.sum() * 100

    if labels is not None:
        true_counts = np.bincount(labels, minlength=NUM_STAGES)
        true_pct = true_counts / true_counts.sum() * 100
        bars1 = ax.bar(x - width/2, true_pct, width, label="Scored",
                       color=[STAGE_COLORS[s] for s in STAGE_NAMES], alpha=0.7)
        bars2 = ax.bar(x + width/2, pred_pct, width, label="Predicted",
                       color=[STAGE_COLORS[s] for s in STAGE_NAMES], alpha=0.4,
                       edgecolor="black", linewidth=0.8)
        ax.legend()
    else:
        ax.bar(x, pred_pct, width * 2,
               color=[STAGE_COLORS[s] for s in STAGE_NAMES], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_ylabel("% of epochs")
    ax.set_title("Stage Distribution")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_stage_f1(f1_per: NDArray, save_path: Path) -> None:
    """Bar chart of per-stage F1 scores."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(NUM_STAGES)
    colors = [STAGE_COLORS[s] for s in STAGE_NAMES]
    bars = ax.bar(x, f1_per, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)

    for bar, f1 in zip(bars, f1_per):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{f1:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_NAMES)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Stage F1")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


# -- Main --------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sleep stage inference on a BIDS EEG dataset")
    parser.add_argument("dataset_path", type=str,
                        help="Path to BIDS dataset root directory")
    parser.add_argument("--channel", type=str, default=None,
                        help="EEG channel to use (auto-detected if omitted)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to downstream checkpoint")
    parser.add_argument("--exp-name", type=str, default="final",
                        help="Experiment name for checkpoint search")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results (default: dataset_path/results)")
    parser.add_argument("--seq-len", type=int, default=200,
                        help="Sequence length for prediction windows")
    parser.add_argument("--max-subjects", type=int, default=None,
                        help="Limit number of subjects (for testing)")
    args = parser.parse_args()

    bids_dir = Path(args.dataset_path)
    if not bids_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {bids_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else bids_dir / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Find recordings
    recordings = find_eeg_files(bids_dir)
    if not recordings:
        raise FileNotFoundError(f"No EEG files found in {bids_dir}")
    if args.max_subjects:
        recordings = recordings[:args.max_subjects]
    print(f"Found {len(recordings)} recording(s) in {bids_dir.name}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_model(args.checkpoint, args.exp_name, device)

    # Process each recording
    all_preds = []
    labeled_preds = []  # only predictions that have matching labels
    labeled_labels = []
    results = []

    for i, rec in enumerate(recordings):
        sub_id = rec["sub_id"]
        try:
            epochs, labels, ch = read_recording(rec, args.channel)
        except Exception as e:
            print(f"  [{i+1}/{len(recordings)}] {sub_id}: FAILED ({e})")
            continue

        if len(epochs) == 0:
            print(f"  [{i+1}/{len(recordings)}] {sub_id}: no valid epochs")
            continue

        preds = predict_subject(epochs, module, device, args.seq_len)

        # Per-subject metrics
        sub_info = {"sub_id": sub_id, "channel": ch, "n_epochs": len(preds)}
        if labels is not None and len(labels) == len(preds):
            kappa = cohen_kappa_score(labels, preds)
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            sub_info["kappa"] = kappa
            sub_info["f1"] = f1
            labeled_preds.extend(preds.tolist())
            labeled_labels.extend(labels.tolist())
            print(f"  [{i+1}/{len(recordings)}] {sub_id} ({ch}): "
                  f"{len(preds)} epochs, kappa={kappa:.3f}, F1={f1:.3f}")
        else:
            print(f"  [{i+1}/{len(recordings)}] {sub_id} ({ch}): "
                  f"{len(preds)} epochs (no labels)")

        all_preds.extend(preds.tolist())
        results.append(sub_info)

        # Per-subject hypnogram
        plot_hypnogram(preds, labels, sub_id,
                       plots_dir / f"hypnogram_{sub_id}.png")

    if not results:
        print("No recordings processed successfully.")
        return

    all_preds = np.array(all_preds)
    has_labels = len(labeled_labels) > 0
    labeled_preds_arr = np.array(labeled_preds) if has_labels else None
    labeled_labels_arr = np.array(labeled_labels) if has_labels else None

    # -- Aggregate results ---------------------------------------------------

    print(f"\n{'='*70}")
    print(f"INFERENCE RESULTS: {bids_dir.name}")
    print(f"{'='*70}")
    print(f"Recordings: {len(results)}")
    print(f"Total epochs: {len(all_preds)}")
    print(f"Duration: {len(all_preds) * EPOCH_SEC / 3600:.1f} hours")
    if has_labels:
        print(f"Labeled epochs: {len(labeled_labels)}")

    # Stage distribution
    print(f"\nStage distribution (predicted):")
    counts = np.bincount(all_preds, minlength=NUM_STAGES)
    for stage, name in enumerate(STAGE_NAMES):
        pct = counts[stage] / len(all_preds) * 100
        print(f"  {name}: {counts[stage]:>6} ({pct:>5.1f}%)")

    # Metrics (if any labels available)
    if has_labels:
        kappa = cohen_kappa_score(labeled_labels_arr, labeled_preds_arr)
        f1_macro = f1_score(labeled_labels_arr, labeled_preds_arr, average="macro", zero_division=0)
        f1_per = f1_score(labeled_labels_arr, labeled_preds_arr, average=None, zero_division=0)
        acc = (labeled_labels_arr == labeled_preds_arr).mean()
        cm = confusion_matrix(labeled_labels_arr, labeled_preds_arr, labels=list(range(NUM_STAGES)))

        print(f"\nOverall metrics:")
        print(f"  Cohen's kappa: {kappa:.4f}")
        print(f"  Macro F1:      {f1_macro:.4f}")
        print(f"  Accuracy:      {acc:.4f}")
        print(f"\nPer-stage F1:")
        for stage, name in enumerate(STAGE_NAMES):
            print(f"  {name}: {f1_per[stage]:.3f}")

        print(f"\n{classification_report(labeled_labels_arr, labeled_preds_arr, target_names=STAGE_NAMES, zero_division=0)}")

        # Per-subject summary table
        labeled_results = [r for r in results if "kappa" in r]
        if labeled_results:
            print(f"\n{'Subject':<30} {'Channel':<12} {'Epochs':>7} "
                  f"{'Kappa':>7} {'F1':>7}")
            print("-" * 70)
            for r in sorted(labeled_results, key=lambda x: x["kappa"], reverse=True):
                print(f"{r['sub_id']:<30} {r['channel']:<12} {r['n_epochs']:>7} "
                      f"{r['kappa']:>7.3f} {r['f1']:>7.3f}")

        # Plots
        plot_confusion_matrix(cm, plots_dir / "confusion_matrix.png")
        plot_per_stage_f1(f1_per, plots_dir / "per_stage_f1.png")
        plot_stage_distribution(labeled_preds_arr, labeled_labels_arr,
                                plots_dir / "stage_distribution.png")

        # Save metrics to text file
        with open(output_dir / "metrics.txt", "w") as f:
            f.write(f"Dataset: {bids_dir.name}\n")
            f.write(f"Recordings: {len(results)}\n")
            f.write(f"Labeled epochs: {len(labeled_labels)}\n")
            f.write(f"Total epochs: {len(all_preds)}\n")
            f.write(f"Cohen's kappa: {kappa:.4f}\n")
            f.write(f"Macro F1: {f1_macro:.4f}\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Per-stage F1:\n")
            for stage, name in enumerate(STAGE_NAMES):
                f.write(f"  {name}: {f1_per[stage]:.3f}\n")
            f.write(f"\n{classification_report(labeled_labels_arr, labeled_preds_arr, target_names=STAGE_NAMES, zero_division=0)}")
    else:
        plot_stage_distribution(all_preds, None, plots_dir / "stage_distribution.png")

    # Save predictions as NPZ
    save_data = {"predictions": all_preds, "stage_names": STAGE_NAMES}
    if has_labels:
        save_data["labels"] = labeled_labels_arr
        save_data["labeled_predictions"] = labeled_preds_arr
    np.savez(output_dir / "predictions.npz", **save_data)

    print(f"\nResults saved to: {output_dir}")
    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    main()

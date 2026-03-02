"""Per-dataset evaluation breakdown for the downstream sleep model.

Reproduces the same test split as training (seed=42), then evaluates the
best downstream checkpoint on each dataset independently.

Usage:
    python eval_per_dataset.py
    python eval_per_dataset.py --checkpoint path/to/ckpt
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassCohenKappa, MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from config import (
    CACHE_DIR, DATASETS, MODEL_CKPT_DIR, NUM_STAGES, STAGE_NAMES,
    find_best_checkpoint,
)
from dataset import SleepSequenceDataset, _load_subjects, _split_subjects


def _reproduce_test_splits(seed: int = 42, val_frac: float = 0.15,
                           test_frac: float = 0.15,
                           ) -> list[tuple[Path, list[str]]]:
    """Reproduce the exact test split from SleepDataModule (seed-deterministic)."""
    h5_paths = sorted(CACHE_DIR.glob("*.h5"))
    rng = np.random.default_rng(seed)

    ds5555 = [p for p in h5_paths if p.stem.startswith("ds005555_")]
    others = [p for p in h5_paths if not p.stem.startswith("ds005555_")]

    test_plan: list[tuple[Path, list[str]]] = []

    # ds005555 files share subjects -- split jointly first (matches DataModule order)
    if ds5555:
        with h5py.File(ds5555[0], "r") as hf:
            subjects = sorted(hf.keys())
        _, _, test_subs = _split_subjects(subjects, rng, val_frac, test_frac)
        for p in ds5555:
            test_plan.append((p, test_subs))

    for p in others:
        with h5py.File(p, "r") as hf:
            subjects = sorted(hf.keys())
        _, _, test_subs = _split_subjects(subjects, rng, val_frac, test_frac)
        test_plan.append((p, test_subs))

    return test_plan


def _dataset_display_name(h5_stem: str) -> tuple[str, str]:
    """Return (short_name, device_type) for an H5 file stem."""
    for key, meta in DATASETS.items():
        if h5_stem.startswith(key):
            return meta["name"], meta["device"]
    return h5_stem, "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-dataset evaluation")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--exp-name", type=str, default="gru384",
                        help="Experiment name (scopes checkpoint search)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from train_model import SleepStageModule
    dummy_weights = torch.ones(NUM_STAGES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = args.checkpoint or find_best_checkpoint(
        MODEL_CKPT_DIR / args.exp_name, ["val_kappa", "val_f1_macro"])
    print(f"Checkpoint: {ckpt_path}")

    module = SleepStageModule.load_from_checkpoint(
        ckpt_path, map_location="cpu", class_weights=dummy_weights)
    module = module.to(device).eval()

    # Reproduce test splits
    test_plan = _reproduce_test_splits(seed=args.seed)

    results = []
    for h5_path, test_subs in test_plan:
        name, dev = _dataset_display_name(h5_path.stem)

        # Load test data for this dataset only
        plan = [(h5_path, test_subs)]
        te_ep, te_lb, te_ds, te_rng, te_dsn = _load_subjects(plan)
        if len(te_lb) == 0:
            continue

        ds = SleepSequenceDataset(
            te_ep, te_lb, te_rng, seq_len=args.seq_len, stride=args.seq_len)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

        n_epochs = len(te_lb)
        n_subjects = len(te_rng)  # actual subjects loaded (excludes sanitized-out)

        # Metrics
        acc = MulticlassAccuracy(NUM_STAGES).to(device)
        kappa = MulticlassCohenKappa(NUM_STAGES).to(device)
        f1 = MulticlassF1Score(NUM_STAGES, average="macro").to(device)
        f1_per = MulticlassF1Score(NUM_STAGES, average=None).to(device)
        cm = MulticlassConfusionMatrix(NUM_STAGES).to(device)

        with torch.no_grad():
            for batch in loader:
                epochs_b, labels_b, _ = batch
                epochs_b = epochs_b.to(device)
                labels_b = labels_b.to(device)
                logits = module.model(epochs_b)
                if module.crf is not None:
                    paths = module.crf.decode(logits)
                    preds = torch.tensor(paths, device=device).reshape(-1)
                else:
                    preds = logits.argmax(dim=-1).reshape(-1)
                flat_labels = labels_b.reshape(-1)
                acc.update(preds, flat_labels)
                kappa.update(preds, flat_labels)
                f1.update(preds, flat_labels)
                f1_per.update(preds, flat_labels)
                cm.update(preds, flat_labels)

        r = {
            "name": name, "device": dev, "h5": h5_path.stem,
            "subjects": n_subjects, "epochs": n_epochs,
            "acc": acc.compute().item(),
            "kappa": kappa.compute().item(),
            "f1": f1.compute().item(),
            "f1_per": f1_per.compute().cpu().tolist(),
            "cm": cm.compute().cpu().int(),
        }
        results.append(r)
        print(f"  {name:<35} kappa={r['kappa']:.3f}  F1={r['f1']:.3f}")

    # Sort by kappa descending
    results.sort(key=lambda r: r["kappa"], reverse=True)

    # Summary table
    print("\n" + "=" * 100)
    print("PER-DATASET EVALUATION")
    print("=" * 100)
    print(f"{'Dataset':<35} {'Device':<12} {'Subj':>5} {'Epochs':>8} "
          f"{'Kappa':>7} {'F1':>7} {'Acc':>7}  "
          + "".join(f"{s:>6}" for s in STAGE_NAMES))
    print("-" * 100)

    for r in results:
        per_f1 = "".join(f"{v:>6.3f}" for v in r["f1_per"])
        print(f"{r['name']:<35} {r['device']:<12} {r['subjects']:>5} "
              f"{r['epochs']:>8} {r['kappa']:>7.3f} {r['f1']:>7.3f} "
              f"{r['acc']:>7.3f}  {per_f1}")

    # Device-type aggregation
    print("\n" + "=" * 60)
    print("BY DEVICE TYPE")
    print("=" * 60)
    device_groups: dict[str, list[dict]] = {}
    for r in results:
        device_groups.setdefault(r["device"], []).append(r)

    print(f"{'Device':<15} {'Datasets':>3} {'Subj':>5} {'Epochs':>8} "
          f"{'Kappa':>7} {'F1':>7}")
    print("-" * 60)
    for dev, group in sorted(device_groups.items()):
        n_ds = len(group)
        n_sub = sum(r["subjects"] for r in group)
        n_ep = sum(r["epochs"] for r in group)
        avg_k = sum(r["kappa"] * r["epochs"] for r in group) / n_ep
        avg_f1 = sum(r["f1"] * r["epochs"] for r in group) / n_ep
        print(f"{dev:<15} {n_ds:>3} {n_sub:>5} {n_ep:>8} "
              f"{avg_k:>7.3f} {avg_f1:>7.3f}")


if __name__ == "__main__":
    main()

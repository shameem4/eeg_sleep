"""Evaluate encoder embedding quality: stage separation + domain invariance.

Loads best encoder checkpoint, extracts embeddings from test set, computes:
- Silhouette score by stage (want high, max 1.0)
- Silhouette score by dataset (want ~0, i.e. no domain clustering)
- Calinski-Harabasz index by stage (higher = better separation)
- Davies-Bouldin index by stage (lower = better separation)
- k-NN accuracy (non-linear separability of embeddings)

Usage:
    python eval_embeddings.py
    python eval_embeddings.py --checkpoint path/to/ckpt
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import (
    calinski_harabasz_score, cohen_kappa_score, davies_bouldin_score,
    f1_score, pairwise_distances, silhouette_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

from config import ENCODER_CKPT_DIR, MODEL_CKPT_DIR, NUM_STAGES, STAGE_NAMES, find_best_checkpoint
from dataset import SleepDataModule
from model import EpochEncoder


def eval_embeddings(embeddings: np.ndarray, labels: np.ndarray,
                    dataset_ids: np.ndarray, max_points: int = 15000) -> dict:
    """Compute embedding quality metrics."""
    n = len(embeddings)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        embeddings, labels, dataset_ids = embeddings[idx], labels[idx], dataset_ids[idx]

    results = {}

    # Precompute distance matrix once (biggest speedup: 7 silhouette calls reuse it)
    dist = pairwise_distances(embeddings, metric="euclidean")

    # Stage separation metrics
    results["silhouette_stage"] = silhouette_score(dist, labels, metric="precomputed")
    results["calinski_harabasz"] = calinski_harabasz_score(embeddings, labels)
    results["davies_bouldin"] = davies_bouldin_score(embeddings, labels)

    # Domain invariance: silhouette by dataset (want ~0)
    n_unique_ds = len(np.unique(dataset_ids))
    if n_unique_ds > 1:
        results["silhouette_dataset"] = silhouette_score(
            dist, dataset_ids, metric="precomputed")
    else:
        results["silhouette_dataset"] = 0.0

    # Per-stage silhouette (one-vs-rest)
    for s in range(NUM_STAGES):
        binary = (labels == s).astype(int)
        if binary.sum() > 0 and binary.sum() < len(binary):
            results[f"silhouette_{STAGE_NAMES[s]}"] = silhouette_score(
                dist, binary, metric="precomputed")

    # k-NN metrics (5-fold, k=5)
    knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
    preds = cross_val_predict(knn, dist, labels, cv=5)
    results["knn_kappa"] = cohen_kappa_score(labels, preds)
    results["knn_f1_macro"] = f1_score(labels, preds, average="macro")
    results["knn_acc"] = (preds == labels).mean()

    return results


def print_results(results: dict) -> None:
    """Pretty-print embedding quality metrics."""
    print("\n" + "=" * 55)
    print("EMBEDDING QUALITY METRICS")
    print("=" * 55)
    print(f"{'Metric':<30} {'Value':>10}")
    print("-" * 55)
    print(f"{'Silhouette (stage)':<30} {results['silhouette_stage']:>10.3f}   (want high)")
    print(f"{'Silhouette (dataset)':<30} {results['silhouette_dataset']:>10.3f}   (want ~0)")
    print(f"{'Calinski-Harabasz':<30} {results['calinski_harabasz']:>10.1f}   (higher=better)")
    print(f"{'Davies-Bouldin':<30} {results['davies_bouldin']:>10.3f}   (lower=better)")
    print(f"{'k-NN kappa (k=5)':<30} {results['knn_kappa']:>10.3f}   (want high)")
    print(f"{'k-NN F1 macro':<30} {results['knn_f1_macro']:>10.3f}")
    print(f"{'k-NN accuracy':<30} {results['knn_acc']:>10.3f}")
    print("-" * 55)
    print("Per-stage silhouette (one-vs-rest):")
    for s in range(NUM_STAGES):
        key = f"silhouette_{STAGE_NAMES[s]}"
        if key in results:
            print(f"  {STAGE_NAMES[s]:<26} {results[key]:>10.3f}")
    print("=" * 55)


def extract_encoder_embeddings(encoder: torch.nn.Module,
                               dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings using a bare EpochEncoder from any source."""
    encoder.eval()
    device = next(encoder.parameters()).device
    embs, labs, ds_ids = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            epochs, labels, domain_ids = batch
            emb = encoder(epochs.to(device))
            embs.append(emb.cpu().numpy())
            labs.append(labels.numpy())
            ds_ids.append(domain_ids.numpy())
    return np.concatenate(embs), np.concatenate(labs), np.concatenate(ds_ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate encoder embeddings")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint (default: best in checkpoints/encoder/)")
    parser.add_argument("--from-model", action="store_true",
                        help="Load encoder from fine-tuned SleepStageModule checkpoint")
    parser.add_argument("--exp-name", type=str, default="v12",
                        help="Experiment name (scopes checkpoint search)")
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()

    # Find checkpoint
    if args.from_model:
        ckpt_dir = MODEL_CKPT_DIR / args.exp_name
        metrics = ["val_kappa"]
    else:
        ckpt_dir = ENCODER_CKPT_DIR / args.exp_name
        metrics = ["val_loss", "val_knn", "val_kappa"]

    ckpt_path = args.checkpoint or find_best_checkpoint(ckpt_dir, metrics)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Source: {'fine-tuned model' if args.from_model else 'standalone encoder'}")

    # Load data
    dm = SleepDataModule(
        dataset_key="all", multi_dataset=True,
        batch_size=args.batch_size, num_workers=4,
        seed=42, epoch_mode=True,
    )
    dm.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.from_model:
        # Load fine-tuned model, extract encoder
        from train_model import SleepStageModule
        module = SleepStageModule.load_from_checkpoint(
            ckpt_path, class_weights=dm.class_weights)
        encoder = module.model.epoch_encoder.to(device)
        print("Extracting embeddings from fine-tuned encoder...")
        embs, labs, ds_ids = extract_encoder_embeddings(encoder, dm.test_dataloader())
    else:
        # Load encoder weights from checkpoint (works with any checkpoint format)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        encoder_state = {k.replace("encoder.", ""): v
                         for k, v in ckpt["state_dict"].items()
                         if k.startswith("encoder.")}
        encoder = EpochEncoder()
        encoder.load_state_dict(encoder_state)
        encoder.to(device)
        del ckpt
        print("Extracting embeddings from encoder...")
        embs, labs, ds_ids = extract_encoder_embeddings(encoder, dm.test_dataloader())

    print(f"Embeddings: {embs.shape} from {len(np.unique(ds_ids))} datasets")

    # Compute and print metrics
    print("Computing metrics...")
    results = eval_embeddings(embs, labs, ds_ids)
    print_results(results)


if __name__ == "__main__":
    main()

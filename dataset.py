"""PyTorch Dataset and Lightning DataModule for sleep staging.

Reads preprocessed HDF5 files, preloads into RAM for fast training.
Supports:
- Single-dataset or multi-dataset (all cached H5s) training
- Sequence-based loading (N consecutive epochs with temporal context)
- Subject-aware train/val/test splits (no subject leakage)
- Class-weighted sampling for imbalanced stages
"""
import logging
from pathlib import Path
from typing import Optional

import h5py
import lightning as L
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from config import CACHE_DIR, EPOCH_SAMPLES, NUM_STAGES

log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _class_weights(labels: Tensor) -> Tensor:
    """Inverse frequency weights for loss function, normalized to mean=1."""
    counts = torch.bincount(labels, minlength=NUM_STAGES).float()
    counts = torch.clamp(counts, min=1.0)
    weights = 1.0 / counts
    weights /= weights.sum()
    return (weights * NUM_STAGES).float()


def _split_subjects(subjects: list[str], rng: np.random.Generator,
                    val_frac: float, test_frac: float,
                    ) -> tuple[list[str], list[str], list[str]]:
    """Split subject list into train/val/test (no leakage)."""
    perm = rng.permutation(len(subjects))
    n_test = max(1, int(len(subjects) * test_frac))
    n_val = max(1, int(len(subjects) * val_frac))
    test = [subjects[i] for i in perm[:n_test]]
    val = [subjects[i] for i in perm[n_test:n_test + n_val]]
    train = [subjects[i] for i in perm[n_test + n_val:]]
    return train, val, test


def _load_subjects(plan: list[tuple[Path, list[str]]],
                   h5_to_idx: Optional[dict[str, int]] = None,
                   ) -> tuple[Tensor, Tensor, Tensor, list[tuple[int, int]], list[int]]:
    """Load epochs+labels from H5 files into shared-memory tensors.

    Args:
        plan: list of (h5_path, subject_ids) to load
        h5_to_idx: mapping from h5 stem to dataset index (for domain labels)

    Returns:
        epochs: (N, EPOCH_SAMPLES) float16 shared-memory tensor (cast to float32 in Dataset)
        labels: (N,) int64 shared-memory tensor
        dataset_ids: (N,) int64 shared-memory tensor (domain labels)
        ranges: list of (offset, length) per subject
        dataset_n_subjects: list of subject count in source dataset, per range
    """
    all_epochs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_ds_ids: list[np.ndarray] = []
    ranges: list[tuple[int, int]] = []
    dataset_n_subjects: list[int] = []
    offset = 0

    for h5_path, subject_ids in plan:
        n_subs = len(subject_ids)
        ds_idx = h5_to_idx.get(h5_path.stem, 0) if h5_to_idx else 0
        with h5py.File(h5_path, "r") as hf:
            for sid in subject_ids:
                if sid not in hf:
                    continue
                grp = hf[sid]
                assert isinstance(grp, h5py.Group)
                epochs = np.asarray(grp["epochs"])
                labels = np.asarray(grp["labels"])
                all_epochs.append(epochs)
                all_labels.append(labels)
                all_ds_ids.append(np.full(len(epochs), ds_idx, dtype=np.int64))
                ranges.append((offset, len(epochs)))
                dataset_n_subjects.append(n_subs)
                offset += len(epochs)

    if not all_epochs:
        return (torch.empty(0, EPOCH_SAMPLES),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long), [], [])

    # Convert to float16 per-chunk before concatenating (avoids a full
    # float32 intermediate that would double peak RAM ~20 GB -> ~40 GB).
    for i in range(len(all_epochs)):
        all_epochs[i] = all_epochs[i].astype(np.float16)
    epochs_np = np.concatenate(all_epochs)
    del all_epochs
    labels_np = np.concatenate(all_labels).astype(np.int64)
    del all_labels
    ds_ids_np = np.concatenate(all_ds_ids)
    del all_ds_ids

    # Shared-memory tensors: DataLoader workers get a reference, not a pickle copy
    # (Windows spawn can't pickle >2 GB objects)
    epochs_t = torch.from_numpy(epochs_np).share_memory_()
    labels_t = torch.from_numpy(labels_np).share_memory_()
    ds_ids_t = torch.from_numpy(ds_ids_np).share_memory_()
    del epochs_np, labels_np, ds_ids_np

    log.info("Loaded %d epochs (%.1f GB)", offset, epochs_t.nbytes / 1e9)
    return epochs_t, labels_t, ds_ids_t, ranges, dataset_n_subjects


# ── Dataset ───────────────────────────────────────────────────────────

class SleepEpochDataset(Dataset):
    """Per-epoch dataset for encoder training.

    Each sample is (EPOCH_SAMPLES,) with label and dataset_id scalars.
    """

    def __init__(self, epochs: Tensor, labels: Tensor, dataset_ids: Tensor,
                 sample_alpha: float = 0.5) -> None:
        self.epochs = epochs
        self.labels = labels
        self.dataset_ids = dataset_ids
        self.sample_alpha = sample_alpha

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.epochs[idx].float(), self.labels[idx], self.dataset_ids[idx]

    def get_class_weights(self) -> Tensor:
        return _class_weights(self.labels)

    def get_sample_weights(self) -> Tensor:
        """Per-epoch weights for dataset-balanced sampling."""
        if self.sample_alpha <= 0:
            return torch.ones(len(self), dtype=torch.float64)
        ds_counts = torch.bincount(self.dataset_ids).float()
        per_epoch = ds_counts[self.dataset_ids]
        return (1.0 / per_epoch.pow(self.sample_alpha)).double()


class SleepSequenceDataset(Dataset):
    """Sequences of consecutive 30s epochs, preloaded into shared memory.

    Each sample is (seq_len, EPOCH_SAMPLES) with corresponding labels (seq_len,).
    """

    def __init__(self, epochs: Tensor, labels: Tensor,
                 subject_ranges: list[tuple[int, int]],
                 seq_len: int = 20, stride: int = 10,
                 dataset_n_subjects: Optional[list[int]] = None,
                 sample_alpha: float = 0.5) -> None:
        self.seq_len = seq_len
        self.epochs = epochs
        self.labels = labels
        self.sample_alpha = sample_alpha

        # Build sequence index: sequences must not cross subject boundaries
        self.index: list[tuple[int, int]] = []
        seq_ds_n: list[int] = []
        for sub_i, (sub_offset, sub_len) in enumerate(subject_ranges):
            n_ds = dataset_n_subjects[sub_i] if dataset_n_subjects else 1
            if sub_len < seq_len:
                continue
            for start in range(0, sub_len - seq_len + 1, stride):
                self.index.append((sub_offset + start, seq_len))
                seq_ds_n.append(n_ds)
        self._seq_ds_n = torch.tensor(seq_ds_n, dtype=torch.float64)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        start, length = self.index[idx]
        return (self.epochs[start:start + length].float(),
                self.labels[start:start + length].clone(),
                torch.zeros(1))

    def get_class_weights(self) -> Tensor:
        return _class_weights(self.labels)

    def get_sample_weights(self) -> Tensor:
        """Per-sample weights for dataset balance only (class balance is in the loss).

        Weight ~ 1/n^alpha where n = subjects in source dataset.
        alpha=0 ignores dataset size, alpha=1 fully equalizes, alpha=0.5 (default) = sqrt.
        """
        if self.sample_alpha <= 0:
            return torch.ones(len(self.index), dtype=torch.float64)
        return 1.0 / self._seq_ds_n.pow(self.sample_alpha)


# ── DataModule ────────────────────────────────────────────────────────

class SleepDataModule(L.LightningDataModule):
    """Lightning DataModule with subject-level splits.

    Supports single-dataset (dataset_key + acq) or multi-dataset (all cached H5s).
    """

    def __init__(self, h5_path: Optional[Path] = None,
                 dataset_key: str = "ds005555",
                 acq: str = "headband",
                 multi_dataset: bool = False,
                 seq_len: int = 200,
                 stride_train: int = 50,
                 stride_val: int = 200,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 val_frac: float = 0.15,
                 test_frac: float = 0.15,
                 seed: int = 42,
                 use_weighted_sampler: bool = True,
                 sample_alpha: float = 0.5,
                 epoch_mode: bool = False,
                 max_subjects_per_ds: int = 0) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["h5_path"])
        self.h5_path = h5_path or CACHE_DIR / f"{dataset_key}_{acq}.h5"
        self.multi_dataset = multi_dataset
        self.seq_len = seq_len
        self.stride_train = stride_train
        self.stride_val = stride_val
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.seed = seed
        self.use_weighted_sampler = use_weighted_sampler
        self.sample_alpha = sample_alpha
        self.epoch_mode = epoch_mode
        self.max_subjects_per_ds = max_subjects_per_ds

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None
        self.class_weights: Optional[Tensor] = None
        self.n_domains: int = 0

    def setup(self, stage: Optional[str] = None) -> None:
        if self.train_ds is not None:
            return

        h5_paths = (sorted(CACHE_DIR.glob("*.h5")) if self.multi_dataset
                     else [self.h5_path])
        h5_to_idx = {p.stem: i for i, p in enumerate(h5_paths)}
        self.n_domains = len(h5_paths)

        # Collect per-file subject lists
        file_subjects: dict[str, list[str]] = {}
        for p in h5_paths:
            with h5py.File(p, "r") as hf:
                file_subjects[p.stem] = sorted(hf.keys())

        # Per-dataset subject splitting
        rng = np.random.default_rng(self.seed)
        train_plan: list[tuple[Path, list[str]]] = []
        val_plan: list[tuple[Path, list[str]]] = []
        test_plan: list[tuple[Path, list[str]]] = []
        n_train = n_val = n_test = 0

        # ds005555 headband+psg share same subjects -- split jointly
        ds5555 = [p for p in h5_paths if p.stem.startswith("ds005555_")]
        others = [p for p in h5_paths if not p.stem.startswith("ds005555_")]

        cap = self.max_subjects_per_ds

        if ds5555:
            subjects = file_subjects[ds5555[0].stem]
            if cap > 0:
                subjects = subjects[:cap]
            train_subs, val_subs, test_subs = _split_subjects(
                subjects, rng, self.val_frac, self.test_frac)
            for p in ds5555:
                train_plan.append((p, train_subs))
                val_plan.append((p, val_subs))
                test_plan.append((p, test_subs))
            n_train += len(train_subs) * len(ds5555)
            n_val += len(val_subs) * len(ds5555)
            n_test += len(test_subs) * len(ds5555)

        for p in others:
            subjects = file_subjects[p.stem]
            if cap > 0:
                subjects = subjects[:cap]
            train_subs, val_subs, test_subs = _split_subjects(
                subjects, rng, self.val_frac, self.test_frac)
            train_plan.append((p, train_subs))
            val_plan.append((p, val_subs))
            test_plan.append((p, test_subs))
            n_train += len(train_subs)
            n_val += len(val_subs)
            n_test += len(test_subs)

        print(f"Split: {n_train} train, {n_val} val, {n_test} test subjects"
              f" from {len(h5_paths)} dataset(s)")

        # Load into shared-memory tensors
        print("Loading train split...", flush=True)
        tr_ep, tr_lb, tr_ds, tr_rng, tr_dsn = _load_subjects(train_plan, h5_to_idx)
        print("Loading val split...", flush=True)
        va_ep, va_lb, va_ds, va_rng, va_dsn = _load_subjects(val_plan, h5_to_idx)
        print("Loading test split...", flush=True)
        te_ep, te_lb, te_ds, te_rng, te_dsn = _load_subjects(test_plan, h5_to_idx)

        if self.epoch_mode:
            self.train_ds = SleepEpochDataset(
                tr_ep, tr_lb, tr_ds, sample_alpha=self.sample_alpha)
            self.val_ds = SleepEpochDataset(va_ep, va_lb, va_ds, sample_alpha=0.0)
            self.test_ds = SleepEpochDataset(te_ep, te_lb, te_ds, sample_alpha=0.0)
        else:
            self.train_ds = SleepSequenceDataset(
                tr_ep, tr_lb, tr_rng, self.seq_len, self.stride_train,
                dataset_n_subjects=tr_dsn, sample_alpha=self.sample_alpha)
            self.val_ds = SleepSequenceDataset(
                va_ep, va_lb, va_rng, self.seq_len, self.stride_val,
                dataset_n_subjects=va_dsn, sample_alpha=0.0)
            self.test_ds = SleepSequenceDataset(
                te_ep, te_lb, te_rng, self.seq_len, self.stride_val,
                dataset_n_subjects=te_dsn, sample_alpha=0.0)

        self.class_weights = self.train_ds.get_class_weights()
        self._test_ranges = te_rng  # (offset, length) per subject

        n_type = "Epochs" if self.epoch_mode else "Sequences"
        print(f"{n_type}: {len(self.train_ds)} train, {len(self.val_ds)} val, "
              f"{len(self.test_ds)} test")
        print(f"Class weights: {self.class_weights.tolist()}")

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None
        sampler = None
        shuffle = True
        if self.use_weighted_sampler and self.sample_alpha > 0:
            sample_weights = self.train_ds.get_sample_weights()
            sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
            shuffle = False

        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=shuffle,
            sampler=sampler, num_workers=self.num_workers,
            pin_memory=True, persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_ds is not None
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_ds is not None
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

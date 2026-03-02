"""Train SleepStageNet: frozen pretrained encoder + BiGRU + CRF.

Encoder stays frozen by default (--freeze-epochs 999).  Unfreezing at 0.1x LR
is supported but not the current baseline.

Usage:
    python train_model.py                              # default (frozen encoder)
    python train_model.py --encoder-ckpt path/to/ckpt  # custom checkpoint
    python train_model.py --max-subjects 20            # fewer subjects for fast iteration
    python train_model.py --freeze-epochs 3            # unfreeze after 3 epochs
"""

import argparse

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassCohenKappa, MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from config import (
    ENCODER_CKPT_DIR, MODEL_CKPT_DIR, NUM_STAGES, STAGE_NAMES,
    find_best_checkpoint,
)
from dataset import SleepDataModule
from model import SleepStageNet, count_parameters


class SleepStageModule(L.LightningModule):
    """Full model: pretrained encoder + BiGRU + CRF, with phased unfreezing."""

    def __init__(self, encoder_ckpt: str = "",
                 gru_hidden: int = 384,
                 gru_layers: int = 2, dropout: float = 0.5,
                 spectral_dim: int = 128, stft_dim: int = 128,
                 norm_type: str = "group",
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 encoder_lr_factor: float = 0.1,
                 class_weights: Tensor | None = None,
                 freeze_epochs: int = 999,
                 warmup_epochs: int = 3,
                 use_crf: bool = True,
                 ce_weight: float = 1.0) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.lr = lr
        self.weight_decay = weight_decay
        self.encoder_lr_factor = encoder_lr_factor
        self.freeze_epochs = freeze_epochs
        self.warmup_epochs = warmup_epochs
        self.ce_weight = ce_weight
        self.use_crf = use_crf

        self.model = SleepStageNet(
            gru_hidden=gru_hidden,
            gru_layers=gru_layers, dropout=dropout,
            spectral_dim=spectral_dim, stft_dim=stft_dim,
            norm_type=norm_type,
        )

        # CRF layer for structured sequence output
        if use_crf:
            from torchcrf import CRF
            self.crf = CRF(NUM_STAGES, batch_first=True)
        else:
            self.crf = None

        # Load pretrained encoder
        if encoder_ckpt:
            ckpt = torch.load(encoder_ckpt, map_location="cpu",
                              weights_only=False)
            encoder_state = {k.replace("encoder.", ""): v
                             for k, v in ckpt["state_dict"].items()
                             if k.startswith("encoder.")}
            self.model.epoch_encoder.load_state_dict(encoder_state)
            print(f"Loaded encoder from {encoder_ckpt}")

        # Freeze encoder initially
        if freeze_epochs > 0:
            self._set_encoder_frozen(True)

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        # Metrics
        self.train_acc = MulticlassAccuracy(NUM_STAGES)
        self.val_acc = MulticlassAccuracy(NUM_STAGES)
        self.val_kappa = MulticlassCohenKappa(NUM_STAGES)
        self.val_f1 = MulticlassF1Score(NUM_STAGES, average="macro")
        self.val_f1_per = MulticlassF1Score(NUM_STAGES, average=None)
        self.test_acc = MulticlassAccuracy(NUM_STAGES)
        self.test_kappa = MulticlassCohenKappa(NUM_STAGES)
        self.test_f1 = MulticlassF1Score(NUM_STAGES, average="macro")
        self.test_f1_per = MulticlassF1Score(NUM_STAGES, average=None)
        self.test_cm = MulticlassConfusionMatrix(NUM_STAGES)

    def _set_encoder_frozen(self, frozen: bool) -> None:
        for p in self.model.epoch_encoder.parameters():
            p.requires_grad = not frozen
        n = sum(p.numel() for p in self.model.epoch_encoder.parameters())
        print(f"Encoder {'frozen' if frozen else 'unfrozen'} ({n:,} params)")

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.freeze_epochs:
            self._set_encoder_frozen(False)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _compute_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute loss: CRF NLL + optional weighted CE aux, or CE only."""
        if self.crf is not None:
            loss = -self.crf(logits, labels, reduction="token_mean")
            if self.ce_weight > 0:
                flat = logits.reshape(-1, NUM_STAGES)
                flat_labels = labels.reshape(-1)
                loss = loss + self.ce_weight * F.cross_entropy(
                    flat, flat_labels, weight=self.class_weights)
            return loss
        return F.cross_entropy(
            logits.reshape(-1, NUM_STAGES), labels.reshape(-1),
            weight=self.class_weights)

    def _decode(self, logits: Tensor) -> Tensor:
        """Decode predictions: argmax or Viterbi (CRF)."""
        if self.crf is not None:
            paths = self.crf.decode(logits)
            return torch.tensor(paths, device=logits.device,
                                dtype=torch.long).reshape(-1)
        return logits.argmax(dim=-1).reshape(-1)

    def _shared_step(self, batch: tuple) -> tuple[Tensor, Tensor, Tensor]:
        epochs, labels, _ = batch  # (B, S, 3840), (B, S), (B, S)
        logits = self.model(epochs)  # (B, S, C)
        loss = self._compute_loss(logits, labels)
        preds = self._decode(logits)
        return loss, preds, labels.reshape(-1)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        epochs, labels, _ = batch
        logits = self.model(epochs)
        loss = self._compute_loss(logits, labels)
        self.train_acc(logits.argmax(-1).reshape(-1), labels.reshape(-1))
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        loss, preds, labels = self._shared_step(batch)
        self.val_acc(preds, labels)
        self.val_kappa(preds, labels)
        self.val_f1(preds, labels)
        self.val_f1_per(preds, labels)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)
        self.log("val_kappa", self.val_kappa, prog_bar=True)
        self.log("val_f1_macro", self.val_f1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        f1_per = self.val_f1_per.compute()
        for s in range(NUM_STAGES):
            self.log(f"val_f1_{STAGE_NAMES[s]}", f1_per[s])
        # N1 F1 on progress bar for live monitoring
        self.log("val_N1_f1", f1_per[STAGE_NAMES.index("N1")], prog_bar=True)

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        _, preds, labels = self._shared_step(batch)
        self.test_acc(preds, labels)
        self.test_kappa(preds, labels)
        self.test_f1(preds, labels)
        self.test_f1_per(preds, labels)
        self.test_cm(preds, labels)
        self.log("test/acc", self.test_acc)
        self.log("test/kappa", self.test_kappa)
        self.log("test/f1_macro", self.test_f1)

    def on_test_epoch_end(self) -> None:
        f1_per = self.test_f1_per.compute()
        cm = self.test_cm.compute().int()

        print("\nPer-class F1:")
        for s in range(NUM_STAGES):
            print(f"  {STAGE_NAMES[s]:<5} {f1_per[s]:.3f}")

        print("\nConfusion Matrix (rows=true, cols=pred):")
        header = "      " + "".join(f"{STAGE_NAMES[s]:>7}" for s in range(NUM_STAGES))
        print(header)
        for i in range(NUM_STAGES):
            row = f"{STAGE_NAMES[i]:<5} " + "".join(
                f"{cm[i, j]:>7d}" for j in range(NUM_STAGES))
            total = cm[i].sum()
            pct = cm[i, i] / total * 100 if total > 0 else 0
            print(f"{row}  ({pct:.1f}%)")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip torch.compile _orig_mod prefix so checkpoints are portable."""
        checkpoint["state_dict"] = {
            k.replace("._orig_mod.", "."): v
            for k, v in checkpoint["state_dict"].items()
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Re-add _orig_mod prefix if encoder branches are compiled (reverse of save)."""
        enc = self.model.epoch_encoder
        new_state = {}
        for k, v in checkpoint["state_dict"].items():
            if "._orig_mod." not in k:
                for branch in ("spectral", "stft"):
                    prefix = f"model.epoch_encoder.{branch}."
                    if k.startswith(prefix) and hasattr(getattr(enc, branch, None), "_orig_mod"):
                        k = k.replace(prefix, f"{prefix}_orig_mod.", 1)
                        break
            new_state[k] = v
        checkpoint["state_dict"] = new_state

    def configure_optimizers(self):  # type: ignore[override]
        encoder_params = list(self.model.epoch_encoder.parameters())
        encoder_ids = {id(p) for p in encoder_params}
        # self.parameters() includes model + CRF; self.model.parameters() misses CRF
        other_params = [p for p in self.parameters()
                        if id(p) not in encoder_ids]

        optimizer = torch.optim.AdamW([
            {"params": encoder_params,
             "lr": self.lr * self.encoder_lr_factor},
            {"params": other_params, "lr": self.lr},
        ], weight_decay=self.weight_decay)

        max_epochs = self.trainer.max_epochs or 40
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_epochs - self.warmup_epochs),
            eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[self.warmup_epochs])

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SleepStageNet")
    parser.add_argument("--encoder-ckpt", type=str, default="",
                        help="Encoder checkpoint (default: best)")
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr-factor", type=float, default=0.1)
    parser.add_argument("--freeze-epochs", type=int, default=999)
    parser.add_argument("--seq-len", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.005)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--sample-alpha", type=float, default=0.5)
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Cap subjects per dataset (0=all)")
    parser.add_argument("--exp-name", type=str, default="default",
                        help="Experiment name (isolates checkpoints + logs)")
    parser.add_argument("--gru-hidden", type=int, default=384,
                        help="BiGRU hidden size per direction (default: 384)")
    parser.add_argument("--no-crf", action="store_true",
                        help="Disable CRF (CE-only loss)")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the epoch encoder (BiGRU+CRF left uncompiled)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint path (e.g. checkpoints/sleep_model/v10/last.ckpt)")
    parser.add_argument("--model-ckpt", type=str, default="",
                        help="Load full model weights (encoder+BiGRU+CRF) from a prior run checkpoint, "
                             "without restoring trainer state. Use with --freeze-epochs 0 for fine-tuning.")
    args = parser.parse_args()

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    if args.model_ckpt:
        encoder_ckpt = ""  # encoder weights come from --model-ckpt
    else:
        encoder_ckpt = args.encoder_ckpt or find_best_checkpoint(
            ENCODER_CKPT_DIR / args.exp_name, ["val_loss", "val_knn", "val_kappa"])
    print(f"Experiment: {args.exp_name}")
    print(f"Encoder checkpoint: {encoder_ckpt}")

    dm = SleepDataModule(
        dataset_key="all", multi_dataset=True,
        batch_size=args.batch_size, num_workers=4,
        seed=args.seed, sample_alpha=args.sample_alpha,
        seq_len=args.seq_len, stride_train=min(50, args.seq_len),
        stride_val=args.seq_len,
        epoch_mode=False,
        max_subjects_per_ds=args.max_subjects,
    )
    dm.setup()

    module = SleepStageModule(
        encoder_ckpt=encoder_ckpt,
        gru_hidden=args.gru_hidden,
        lr=args.lr, class_weights=dm.class_weights,
        encoder_lr_factor=args.encoder_lr_factor,
        freeze_epochs=args.freeze_epochs,
        warmup_epochs=args.warmup_epochs,
        use_crf=not args.no_crf,
    )
    if args.model_ckpt:
        state = torch.load(args.model_ckpt, map_location="cpu", weights_only=False)
        missing, unexpected = module.load_state_dict(state["state_dict"], strict=False)
        print(f"Loaded full model from {args.model_ckpt}")
        if missing:
            print(f"  Missing keys: {len(missing)}")
        if unexpected:
            print(f"  Unexpected keys (ignored): {len(unexpected)}")

    total_all = sum(p.numel() for p in module.parameters())
    enc_p = sum(p.numel() for p in module.model.epoch_encoder.parameters())
    trainable_p = count_parameters(module)
    print(f"Parameters: {total_all:,} total ({trainable_p:,} trainable)")
    print(f"  Encoder: {enc_p:,}")
    print(f"  GRU+head+CRF: {total_all - enc_p:,}")
    if args.freeze_epochs >= args.max_epochs:
        print("Encoder: permanently frozen")
    else:
        print(f"Phase 1: freeze encoder for {args.freeze_epochs} epochs")
        print(f"Phase 2: unfreeze at {args.encoder_lr_factor}x LR")
    ckpt_dir = MODEL_CKPT_DIR / args.exp_name
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir), filename="{epoch}-{val_kappa:.3f}",
            monitor="val_kappa", mode="max", save_top_k=3, save_last=True),
        EarlyStopping(monitor="val_kappa", mode="max",
                      patience=args.patience, min_delta=args.min_delta),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",  # no AMP: CRF log-sum-exp overflows in fp16
        benchmark=True,
        callbacks=callbacks,
        logger=CSVLogger("logs", name="sleep_model", version=args.exp_name),
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    if args.compile:
        enc = module.model.epoch_encoder
        enc.spectral = torch.compile(enc.spectral)
        enc.stft = torch.compile(enc.stft)
        print("torch.compile applied to spectral + stft branches")

    trainer.fit(module, dm, ckpt_path=args.resume or None)

    # Unwrap compiled branches before test: checkpoint has clean keys but
    # OptimizedModule expects _orig_mod.* keys during load_state_dict.
    if args.compile:
        enc = module.model.epoch_encoder
        if hasattr(enc.spectral, "_orig_mod"):
            enc.spectral = enc.spectral._orig_mod
        if hasattr(enc.stft, "_orig_mod"):
            enc.stft = enc.stft._orig_mod

    if not args.fast_dev_run:
        trainer.test(module, dm, ckpt_path="best")


if __name__ == "__main__":
    main()

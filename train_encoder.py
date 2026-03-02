"""Self-supervised encoder training via reconstruction.

Trains the epoch encoder with reconstruction decoders (spectrogram, waveform, delta).
Fully unsupervised -- no stage labels used.

Usage:
    python train_encoder.py --exp-name v12                                  # full training
    python train_encoder.py --no-augment                                    # disable augmentation
    python train_encoder.py --no-recon                                      # disable reconstruction
"""
import argparse
from pathlib import Path

import lightning as L
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from lightning.pytorch.loggers import CSVLogger
from sklearn.manifold import TSNE
from sklearn.metrics import (
    calinski_harabasz_score, cohen_kappa_score, f1_score,
    pairwise_distances, silhouette_score,
)
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from torch import Tensor


from config import ENCODER_CKPT_DIR, EPOCH_SAMPLES, NUM_STAGES, STAGE_NAMES, TARGET_SFREQ
from dataset import SleepDataModule
from model import (
    BranchDecoder, EpochEncoder,
    SpectrogramDecoder, WaveformDecoder, count_parameters,
)


# -- Training Module --------------------------------------------------------

class EncoderModule(L.LightningModule):
    """Self-supervised encoder with reconstruction decoders. No stage labels used."""

    def __init__(self, spectral_dim: int = 128, stft_dim: int = 128,
                 lr: float = 1e-3, weight_decay: float = 1e-4,
                 warmup_epochs: int = 5,
                 norm_type: str = "group",
                 augment: bool = True,
                 recon_weight: float = 0.5,
                 wave_samples: int = 960,
                 branch_weight: float = 0.3,
                 n_fft: int = 256,
                 hop_length: int = 128) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.augment = augment
        self.recon_weight = recon_weight
        self.branch_weight = branch_weight
        self.wave_samples = wave_samples
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.encoder = EpochEncoder(
            spectral_dim=spectral_dim,
            stft_dim=stft_dim, norm_type=norm_type)

        # Reconstruction decoders (training-only, discarded after training)
        fused_dim = self.encoder.feat_dim
        bw = branch_weight
        if recon_weight > 0:
            self.register_buffer("_stft_window", torch.hann_window(n_fft))
        self.spec_decoder = SpectrogramDecoder(fused_dim) if recon_weight > 0 else None
        self.wave_decoder = (WaveformDecoder(fused_dim, wave_samples)
                             if bw > 0 else None)
        self.delta_decoder = BranchDecoder(128) if bw > 0 else None
        self.highfreq_decoder = BranchDecoder(128) if bw > 0 else None
        self.theta_decoder = BranchDecoder(128) if bw > 0 else None
        self.slowwave_decoder = BranchDecoder(128) if bw > 0 else None
        self.sawtooth_decoder = BranchDecoder(128) if bw > 0 else None

    def _augment_signal(self, x: Tensor) -> Tensor:
        """Mild augmentation: amplitude scaling + relative Gaussian noise."""
        if not self.training or not self.augment:
            return x
        # Random amplitude scaling [0.8, 1.2]
        scale = 0.8 + 0.4 * torch.rand(x.size(0), 1, device=x.device)
        x = x * scale
        # Relative Gaussian noise: x * (1 + N(0, 0.05))
        x = x * (1.0 + 0.05 * torch.randn_like(x))
        return x

    def _compute_wave_target(self, x: Tensor) -> Tensor:
        """Per-epoch z-normalize, downsample to wave_samples."""
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp(min=1e-6)
        normed = (x - mu) / std
        pool = EPOCH_SAMPLES // self.wave_samples  # 4 for 960
        return F.avg_pool1d(normed.unsqueeze(1), pool).squeeze(1)

    def forward(self, x: Tensor) -> Tensor:
        return self.encoder(x)

    def _bandpass_target(self, x: Tensor, lo_hz: float, hi_hz: float,
                         pool_factor: int = 8, rectify: bool = True,
                         spectrum: Tensor | None = None) -> Tensor:
        """Bandpass -> optionally rectify -> z-norm -> downsample -> 480 samples.

        rectify=True: envelope (abs), for highfreq.
        rectify=False: signed waveform, for delta (morphology matters) and theta (sawtooth morphology for REM).
        spectrum: precomputed rfft to avoid redundant FFTs.
        """
        if spectrum is None:
            spectrum = torch.fft.rfft(x.float())
        freq_res = TARGET_SFREQ / x.shape[-1]
        lo_bin = int(lo_hz / freq_res)
        hi_bin = int(hi_hz / freq_res)
        mask = torch.zeros(spectrum.shape[-1], device=x.device)
        mask[lo_bin:hi_bin] = 1.0
        filtered = torch.fft.irfft(spectrum * mask, n=x.shape[-1])
        if rectify:
            filtered = filtered.abs()
        mu = filtered.mean(dim=-1, keepdim=True)
        std = filtered.std(dim=-1, keepdim=True).clamp(min=1e-6)
        normed = (filtered - mu) / std
        return F.avg_pool1d(normed.unsqueeze(1), pool_factor).squeeze(1)

    def _shared_step(self, batch: tuple
                      ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        epochs, labels, _domain_ids = batch
        raw_epochs = epochs  # keep pre-augmentation for recon targets
        epochs = self._augment_signal(epochs)

        # Forward through encoder
        need_branches = (self.delta_decoder is not None
                         or self.highfreq_decoder is not None
                         or self.theta_decoder is not None
                         or self.slowwave_decoder is not None
                         or self.sawtooth_decoder is not None)
        if need_branches:
            fused, feat_s, feat_m, feat_l = self.encoder(
                epochs, return_branches=True)
        else:
            fused = self.encoder(epochs)
            feat_s = feat_m = feat_l = None

        losses: dict[str, Tensor] = {}
        loss_terms: list[Tensor] = []
        bw = self.branch_weight

        # Precompute FFT spectrum once for all bandpass decoders (shared)
        raw_spectrum = (torch.fft.rfft(raw_epochs.float())
                        if need_branches else None)

        # Spectrogram reconstruction
        if self.spec_decoder is not None:
            stft_complex = torch.stft(raw_epochs.float(), self.n_fft, self.hop_length,
                                      window=self._stft_window, return_complex=True)
            # nan_to_num + clamp: prevent fp16 overflow in stft from producing Inf targets
            spec_target = torch.log1p(
                stft_complex.abs().square().nan_to_num(0.0, posinf=1e6)
            ).unsqueeze(1)
            # Cast to float32: fp16 decoder output can overflow (~65k) after many epochs
            losses["spec"] = F.mse_loss(
                self.spec_decoder(fused, spec_target.shape[2:]).float().nan_to_num(0.0),
                spec_target)
            loss_terms.append(self.recon_weight * losses["spec"])

        # Waveform reconstruction
        if self.wave_decoder is not None:
            wave_target = self._compute_wave_target(raw_epochs)
            losses["wave"] = F.mse_loss(
                self.wave_decoder(fused).float().nan_to_num(0.0), wave_target)
            loss_terms.append(bw * losses["wave"])

        # Delta (large CNN branch, bandpass 0.5-4 Hz signed waveform)
        if self.delta_decoder is not None:
            delta_target = self._bandpass_target(
                raw_epochs, 0.5, 4.0, 8, rectify=False, spectrum=raw_spectrum)
            losses["delta"] = F.mse_loss(self.delta_decoder(feat_l), delta_target)
            loss_terms.append(bw * losses["delta"])

        # HighFreq (small CNN branch, 8-45 Hz envelope)
        if self.highfreq_decoder is not None:
            hf_target = self._bandpass_target(
                raw_epochs, 8.0, 45.0, 8, rectify=True, spectrum=raw_spectrum)
            losses["hfreq"] = F.mse_loss(self.highfreq_decoder(feat_s), hf_target)
            loss_terms.append(bw * losses["hfreq"])

        # Theta (medium CNN branch, 4-8 Hz signed)
        if self.theta_decoder is not None:
            theta_target = self._bandpass_target(
                raw_epochs, 4.0, 8.0, 8, rectify=False, spectrum=raw_spectrum)
            losses["theta_m"] = F.mse_loss(self.theta_decoder(feat_m), theta_target)
            loss_terms.append(bw * losses["theta_m"])

        # Slow wave (large CNN branch, 0.5-1 Hz signed)
        if self.slowwave_decoder is not None:
            sw_target = self._bandpass_target(
                raw_epochs, 0.5, 1.0, 8, rectify=False, spectrum=raw_spectrum)
            losses["slowwave"] = F.mse_loss(self.slowwave_decoder(feat_l), sw_target)
            loss_terms.append(bw * losses["slowwave"])

        # Sawtooth (medium CNN branch, 2-6 Hz signed)
        if self.sawtooth_decoder is not None:
            st_target = self._bandpass_target(
                raw_epochs, 2.0, 6.0, 8, rectify=False, spectrum=raw_spectrum)
            losses["sawtooth"] = F.mse_loss(self.sawtooth_decoder(feat_m), st_target)
            loss_terms.append(bw * losses["sawtooth"])

        # Sum all active decoder losses; fused.sum()*0 is a valid zero connected
        # to encoder params (avoids disconnected leaf tensor if no decoders active)
        loss = torch.stack(loss_terms).sum() if loss_terms else fused.sum() * 0
        return loss, labels, fused, losses

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        loss, _, _, losses = self._shared_step(batch)
        self.log("train/loss", loss, prog_bar=True)
        for name, val in losses.items():
            self.log(f"train/{name}", val, prog_bar=(name in ("spec", "wave")))
        return loss

    def on_validation_epoch_start(self) -> None:
        self._val_embs: list[Tensor] = []
        self._val_labels: list[Tensor] = []
        self._val_collected = 0

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        loss, labels, emb, losses = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        for name, val in losses.items():
            self.log(f"val_{name}", val, prog_bar=(name in ("spec", "wave")))
        # Accumulate embeddings for k-NN (cap at ~5000 points)
        if self._val_collected < 5000:
            self._val_embs.append(emb.detach().cpu())
            self._val_labels.append(labels.cpu())
            self._val_collected += emb.size(0)

    def on_validation_epoch_end(self) -> None:
        if not self._val_embs:
            return
        embs = torch.cat(self._val_embs).numpy()
        labels = torch.cat(self._val_labels).numpy()
        dist = pairwise_distances(embs, metric="euclidean")
        knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed")
        preds = cross_val_predict(knn, dist, labels, cv=5)
        kappa = cohen_kappa_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro")
        f1_per = f1_score(labels, preds, average=None, labels=list(range(NUM_STAGES)))
        self.log("val_knn", kappa, prog_bar=True)
        self.log("val_knn_f1", f1, prog_bar=True)
        for s in range(NUM_STAGES):
            self.log(f"val_knn_f1_{STAGE_NAMES[s]}", f1_per[s])
        self.log("val_knn_N1_f1", f1_per[STAGE_NAMES.index("N1")], prog_bar=True)
        sil = silhouette_score(dist, labels, metric="precomputed")
        self.log("val_silhouette", sil, prog_bar=True)
        self.log("val_ch", calinski_harabasz_score(embs, labels))
        del self._val_embs, self._val_labels

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        loss, _, _, _ = self._shared_step(batch)
        self.log("test/loss", loss)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Strip torch.compile _orig_mod prefix so checkpoints are portable."""
        checkpoint["state_dict"] = {
            k.replace("._orig_mod.", "."): v
            for k, v in checkpoint["state_dict"].items()
        }

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Re-add _orig_mod prefix if encoder is compiled (reverse of save)."""
        if hasattr(self.encoder, "_orig_mod"):
            checkpoint["state_dict"] = {
                k.replace("encoder.", "encoder._orig_mod.", 1)
                if k.startswith("encoder.") and "._orig_mod." not in k
                else k: v
                for k, v in checkpoint["state_dict"].items()
            }

    def configure_optimizers(self):  # type: ignore[override]
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        max_epochs = self.trainer.max_epochs or 80
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


@torch.no_grad()
def extract_all_embeddings(module: EncoderModule,
                           dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract embeddings, labels, dataset_ids from entire dataloader."""
    module.eval()
    device = next(module.parameters()).device
    embs, labs, ds_ids = [], [], []
    for batch in dataloader:
        epochs, labels, domain_ids = batch
        emb = module.encoder(epochs.to(device))
        embs.append(emb.cpu().numpy())
        labs.append(labels.numpy())
        ds_ids.append(domain_ids.numpy())
    return np.concatenate(embs), np.concatenate(labs), np.concatenate(ds_ids)


def plot_tsne(embeddings: np.ndarray, labels: np.ndarray,
              dataset_ids: np.ndarray, save_dir: Path,
              max_points: int = 15000) -> None:
    """Generate t-SNE plots colored by stage and by dataset."""
    n = len(embeddings)
    if n > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, max_points, replace=False)
        embeddings, labels, dataset_ids = embeddings[idx], labels[idx], dataset_ids[idx]

    print(f"Running t-SNE on {len(embeddings)} points...", flush=True)
    tsne = TSNE(n_components=2, random_state=42, perplexity=50)
    coords = tsne.fit_transform(embeddings)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: colored by stage
    stage_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
    fig, ax = plt.subplots(figsize=(10, 8))
    for s in range(NUM_STAGES):
        mask = labels == s
        ax.scatter(coords[mask, 0], coords[mask, 1], c=stage_colors[s],
                   label=STAGE_NAMES[s], s=3, alpha=0.4)
    ax.legend(markerscale=5, fontsize=12)
    ax.set_title("t-SNE: Colored by Sleep Stage")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_dir / "tsne_stages.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir / 'tsne_stages.png'}")

    # Plot 2: colored by dataset
    n_ds = dataset_ids.max() + 1
    cmap = plt.colormaps.get_cmap("tab20").resampled(n_ds)
    fig, ax = plt.subplots(figsize=(10, 8))
    for d in range(n_ds):
        mask = dataset_ids == d
        if mask.sum() == 0:
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], c=[cmap(d)],
                   label=f"ds{d}", s=3, alpha=0.4)
    ax.legend(markerscale=5, fontsize=9, ncol=2)
    ax.set_title("t-SNE: Colored by Dataset (should NOT cluster)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(save_dir / "tsne_datasets.png", dpi=150)
    plt.close(fig)
    print(f"  Saved {save_dir / 'tsne_datasets.png'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Self-supervised encoder training")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--sample-alpha", type=float, default=0.5)
    parser.add_argument("--max-subjects", type=int, default=0,
                        help="Cap subjects per dataset (0=all)")
    parser.add_argument("--no-augment", dest="augment", action="store_false",
                        default=True, help="Disable training augmentation")
    parser.add_argument("--no-recon", action="store_true",
                        help="Disable all reconstruction")
    parser.add_argument("--monitor", default="val_loss",
                        choices=["val_loss", "val_knn", "val_knn_f1", "val_silhouette"],
                        help="Metric for early stopping/checkpointing")
    parser.add_argument("--exp-name", type=str, default="v12",
                        help="Experiment name (isolates checkpoints + logs)")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the encoder (requires triton-windows==3.3.1.post21)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from checkpoint path (e.g. checkpoints/encoder/v10/last.ckpt)")
    args = parser.parse_args()
    _bw = 0.0 if args.no_recon else 0.3  # branch decoder weight (0 disables)

    L.seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")

    dm = SleepDataModule(
        dataset_key="all", multi_dataset=True,
        batch_size=args.batch_size, num_workers=2,
        seed=args.seed, sample_alpha=args.sample_alpha,
        epoch_mode=True, max_subjects_per_ds=args.max_subjects,
    )
    dm.setup()

    module = EncoderModule(
        lr=args.lr,
        warmup_epochs=args.warmup_epochs,
        augment=args.augment,
        recon_weight=0.0 if args.no_recon else 0.5,
        branch_weight=_bw,
    )
    enc_p = count_parameters(module.encoder)
    dec_p = count_parameters(module) - enc_p
    print(f"Encoder parameters: {enc_p:,} (output={module.encoder.feat_dim}d)")
    if dec_p:
        print(f"Decoder parameters: {dec_p:,} (training-only)")
    print(f"Total parameters: {count_parameters(module):,}")
    print(f"Datasets: {dm.n_domains}  Experiment: {args.exp_name}")
    print(f"Config: augment={args.augment}, "
          f"monitor={args.monitor}, warmup={args.warmup_epochs}")
    if not args.no_recon:
        print(f"Recon: spec={module.recon_weight}, branch={_bw} "
              "(wave, delta, hfreq, theta, slowwave, sawtooth)")

    ckpt_dir = ENCODER_CKPT_DIR / args.exp_name
    monitor = args.monitor
    monitor_mode = "min" if monitor == "val_loss" else "max"
    ckpt_fn = f"{{epoch}}-{{{monitor}:.3f}}"
    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir), filename=ckpt_fn,
            monitor=monitor, mode=monitor_mode, save_top_k=3, save_last=True),
        EarlyStopping(monitor=monitor, mode=monitor_mode,
                      patience=args.patience, min_delta=0.005),
        RichProgressBar(),
    ]

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto", precision="16-mixed",
        benchmark=True,
        callbacks=callbacks,
        logger=CSVLogger("logs", name="encoder", version=args.exp_name),
        fast_dev_run=args.fast_dev_run,
        log_every_n_steps=50,
        gradient_clip_val=1.0,
    )

    if args.compile:
        module.encoder = torch.compile(module.encoder)
        print("torch.compile applied to encoder")

    trainer.fit(module, dm, ckpt_path=args.resume or None)

    # Unwrap compiled encoder before test: checkpoint has clean keys but
    # OptimizedModule expects _orig_mod.* keys during load_state_dict.
    if args.compile and hasattr(module.encoder, "_orig_mod"):
        module.encoder = module.encoder._orig_mod

    if not args.fast_dev_run:
        trainer.test(module, dm, ckpt_path="best")

        best_ckpt = trainer.checkpoint_callback.best_model_path
        if best_ckpt:
            module = EncoderModule.load_from_checkpoint(best_ckpt)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            module.to(device)

        embs, labs, ds_ids = extract_all_embeddings(module, dm.test_dataloader())
        plot_tsne(embs, labs, ds_ids, save_dir=Path("plots"))

        # Embedding quality metrics
        from eval_embeddings import eval_embeddings, print_results
        results = eval_embeddings(embs, labs, ds_ids)
        print_results(results)


if __name__ == "__main__":
    main()

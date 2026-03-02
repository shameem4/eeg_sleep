"""Sleep stage encoder: five-branch feature extraction + sequence model.

Architecture:
    Input: (batch, 3840) -- 30s epochs at 128Hz
    -> Five-branch encoder:
       - Small CNN (k=50, 3 deep layers): spindles, K-complexes (~8-45 Hz)
       - Medium CNN (k=150, 3 deep layers): theta/alpha morphology (~4-12 Hz)
       - Large CNN (k=400, 3 deep layers): delta/SWA envelope (~0.5-4 Hz)
       - Spectral branch: band powers, ratios, Hjorth, complexity (18 features)
       - STFT branch: time-frequency patches via self-attention
    -> Concatenated 640d fused features (128*3 + 128 + 128)
    -> GroupNorm in CNN, LayerNorm on outputs

Downstream (SleepStageNet):
    -> Spectral + STFT (frozen, mean pooled, 256d; CNN branches skipped)
    -> Bidirectional GRU for sequence modeling
    -> Shared MLP head + CRF
"""
import math

import torch
import torch.nn as nn
from torch import Tensor

from config import EPOCH_SAMPLES, NUM_STAGES, TARGET_SFREQ

_FREQ_RES = TARGET_SFREQ / EPOCH_SAMPLES  # ~0.0333 Hz per bin
_BANDS = [
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta",  16.0, 30.0),
]
_BAND_SLICES = [(int(lo / _FREQ_RES), int(hi / _FREQ_RES)) for _, lo, hi in _BANDS]


class SpectralBranch(nn.Module):
    """Spectral + statistical features via FFT.

    18 features: 5 relative powers + 5 log absolute powers + 4 ratios
    + Hjorth mobility + Hjorth complexity + zero-crossing rate + Petrosian FD.
    """

    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self._n = float(EPOCH_SAMPLES)
        self._log_n = math.log10(EPOCH_SAMPLES)
        self.mlp = nn.Sequential(
            nn.Linear(18, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, 3840) -> (batch, out_dim)."""
        x = x.float()  # fp32 for FFT stability
        eps = 1e-10

        # Band powers via FFT
        spectrum = torch.fft.rfft(x, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        band_powers = [power[:, lo:hi].sum(dim=-1, keepdim=True)
                       for lo, hi in _BAND_SLICES]
        total = sum(band_powers).clamp(min=eps)

        features = []
        for bp in band_powers:
            features.append(bp / total)           # relative power
        for bp in band_powers:
            features.append(torch.log1p(bp))      # log absolute power
        delta, theta, alpha, sigma, beta = band_powers
        features.append(delta / (beta + eps))     # N3 marker
        features.append(theta / (alpha + eps))    # N1 marker
        features.append(sigma / (delta + eps))    # N2 spindle marker
        features.append(alpha / (delta + eps))    # wake marker

        # Hjorth parameters (mobility, complexity)
        dx = x[:, 1:] - x[:, :-1]
        ddx = dx[:, 1:] - dx[:, :-1]
        var_x = x.var(dim=-1, keepdim=True).clamp(min=eps)
        var_dx = dx.var(dim=-1, keepdim=True).clamp(min=eps)
        var_ddx = ddx.var(dim=-1, keepdim=True).clamp(min=eps)
        mobility = torch.sqrt(var_dx / var_x)
        complexity = torch.sqrt(var_ddx / var_dx) / (mobility + eps)
        features.extend([mobility, complexity])

        # Zero-crossing rate + Petrosian fractal dimension
        signs = torch.sign(x)
        zc_count = (signs[:, 1:] != signs[:, :-1]).float().sum(dim=-1, keepdim=True)
        features.append(zc_count / self._n)  # ZCR
        features.append(self._log_n / (self._log_n +
                        torch.log10(self._n / (self._n + 0.4 * zc_count))))  # Petrosian FD

        return self.mlp(torch.cat(features, dim=-1))


class STFTBranch(nn.Module):
    """Time-frequency encoding: STFT -> patch embedding -> self-attention.

    Captures transient spectral dynamics (spindle onset, K-complex shape)
    that fixed-window FFT and raw CNN miss.
    """

    def __init__(self, out_dim: int = 128, n_fft: int = 256,
                 hop_length: int = 128) -> None:
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))
        n_freq = n_fft // 2 + 1  # 129
        self.patch_embed = nn.Linear(n_freq, out_dim)
        n_frames = EPOCH_SAMPLES // hop_length + 1  # 31 (center-padded STFT)
        self.pos_embed = nn.Parameter(torch.randn(1, n_frames, out_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=out_dim, nhead=4, dim_feedforward=out_dim * 2,
            dropout=0.1, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, 3840) -> (batch, out_dim)."""
        x = x.float()
        stft = torch.stft(x, self.n_fft, self.hop_length,
                          window=self.window, return_complex=True)
        power = torch.log1p(stft.abs().square())  # (batch, n_freq, n_frames)
        tokens = self.patch_embed(power.transpose(1, 2))  # (batch, n_frames, dim)
        tokens = tokens + self.pos_embed[:, :tokens.size(1)]
        tokens = self.transformer(tokens)
        return tokens.mean(dim=1)


class EpochEncoder(nn.Module):
    """Five-branch encoder: triple CNN + spectral stats + STFT patches.

    Small CNN (k=50, 3 deep conv): spindles, K-complexes, sharp transients.
    Medium CNN (k=150, 3 deep conv): theta/alpha waveform morphology (~4-12 Hz).
    Large CNN (k=400, 3 deep conv): delta waves, slow oscillations.
    Spectral: band powers, ratios, Hjorth, fractal dimension.
    STFT: time-frequency patches with self-attention.
    GroupNorm in CNN, LayerNorm on branch outputs.
    Output: 640d concatenated features (128*3 + spectral_dim + stft_dim).
    """

    def __init__(self, spectral_dim: int = 128,
                 stft_dim: int = 128, norm_type: str = "group") -> None:
        super().__init__()

        def _norm(ch: int) -> nn.Module:
            if norm_type == "group":
                return nn.GroupNorm(num_groups=min(16, ch), num_channels=ch)
            return nn.BatchNorm1d(ch)

        # Small kernel branch -- fast transients (~8-45 Hz)
        self.small = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=25, padding=25),
            _norm(64), nn.GELU(), nn.MaxPool1d(4), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
        )

        # Medium kernel branch -- theta/alpha morphology (~4-12 Hz)
        self.medium = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=150, stride=25, padding=75),
            _norm(64), nn.GELU(), nn.MaxPool1d(4), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
        )

        # Large kernel branch -- slow oscillations (~0.5-4 Hz)
        self.large = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, padding=200),
            _norm(64), nn.GELU(), nn.MaxPool1d(4), nn.Dropout(0.1),
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(128, 128, kernel_size=7, stride=1, padding=3),
            _norm(128), nn.GELU(), nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.AdaptiveAvgPool1d(1),
        )

        self.norm_small = nn.LayerNorm(128)
        self.norm_medium = nn.LayerNorm(128)
        self.norm_large = nn.LayerNorm(128)
        self.spectral = SpectralBranch(out_dim=spectral_dim)
        self.stft = STFTBranch(out_dim=stft_dim)

        self.feat_dim = 128 * 3 + spectral_dim + stft_dim  # 640

    def forward(self, x: Tensor,
                return_branches: bool = False,
                ) -> Tensor | tuple:
        """x: (batch, 3840) -> (batch, feat_dim).

        If return_branches=True, returns (fused, feat_s, feat_m, feat_l):
            feat_s = small CNN 128d (8-45 Hz transients)
            feat_m = medium CNN 128d (theta 4-8 Hz)
            feat_l = large CNN 128d (delta 0.5-4 Hz)
        """
        spec = self.spectral(x)
        stft = self.stft(x)
        x2 = x.unsqueeze(1)  # (batch, 1, time)
        feat_s = self.norm_small(self.small(x2).squeeze(-1))
        feat_m = self.norm_medium(self.medium(x2).squeeze(-1))
        feat_l = self.norm_large(self.large(x2).squeeze(-1))
        fused = torch.cat([feat_s, feat_m, feat_l, spec, stft], dim=-1)
        if return_branches:
            return fused, feat_s, feat_m, feat_l
        return fused


# -- Sequence Model ------------------------------------------------------------

class SleepStageNet(nn.Module):
    """Full model: encoder (spectral+STFT 256d) + BiGRU + shared head.

    Uses frozen spectral + STFT branches (mean pooled). CNN branches skipped.
    """

    def __init__(self, gru_hidden: int = 384,
                 gru_layers: int = 2, dropout: float = 0.5,
                 spectral_dim: int = 128, stft_dim: int = 128,
                 norm_type: str = "group") -> None:
        super().__init__()
        self.epoch_encoder = EpochEncoder(
            spectral_dim=spectral_dim,
            stft_dim=stft_dim, norm_type=norm_type)

        gru_input = spectral_dim + stft_dim  # 256d
        self.gru = nn.GRU(
            input_size=gru_input, hidden_size=gru_hidden,
            num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=0.3 if gru_layers > 1 else 0.0,
        )
        gru_out = gru_hidden * 2

        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.LayerNorm(gru_out),
            nn.Linear(gru_out, 128), nn.GELU(),
            nn.Linear(128, NUM_STAGES),
        )

    def _encode(self, x: Tensor) -> Tensor:
        """Encode epochs -> BiGRU context.

        Calls spectral + STFT branches (both frozen, mean pooled).
        Skips CNN branches entirely.
        Returns context: (batch, seq_len, gru_hidden*2).
        """
        batch, seq_len, _ = x.shape
        flat = x.reshape(batch * seq_len, -1)
        spec = self.epoch_encoder.spectral(flat)
        stft = self.epoch_encoder.stft(flat)
        gru_in = torch.cat([spec, stft], dim=-1).reshape(batch, seq_len, -1)
        context, _ = self.gru(gru_in)
        return context

    def forward(self, x: Tensor) -> Tensor:
        """x: (batch, seq_len, EPOCH_SAMPLES) -> (batch, seq_len, NUM_STAGES)."""
        return self.head(self._encode(x))


# -- Reconstruction Decoders (training-only) -----------------------------------

class SpectrogramDecoder(nn.Module):
    """Reconstruct log-power spectrogram from fused features (training-only).

    Magnitude-only (no phase) -- aligns with STFTBranch's internal representation
    log1p(|STFT|^2). Halves output dimensionality vs complex reconstruction.
    Spatial geometry: (17, 4) -> 3x upsample -> (136, 32) -> crop to (129, 31).
    """

    def __init__(self, input_dim: int = 640) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 32 * 17 * 4)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),  # (17, 4) -> (34, 8)
            nn.Conv2d(32, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Upsample(scale_factor=2),  # (34, 8) -> (68, 16)
            nn.Conv2d(64, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Upsample(scale_factor=2),  # (68, 16) -> (136, 32)
            nn.Conv2d(32, 1, 3, padding=1),  # -> (1, 136, 32)
        )

    def forward(self, z: Tensor, target_shape: tuple[int, int]) -> Tensor:
        """z: (B, input_dim) -> (B, 1, F, T)."""
        h = self.fc(z).view(-1, 32, 17, 4)
        out = self.net(h)
        return out[:, :, :target_shape[0], :target_shape[1]]


class WaveformDecoder(nn.Module):
    """Reconstruct downsampled normalized waveform from fused features (training-only).

    Output: 960 samples (4x downsampled from 3840).
    """

    def __init__(self, input_dim: int = 640, output_samples: int = 960) -> None:
        super().__init__()
        self.output_samples = output_samples
        self.fc = nn.Linear(input_dim, 64 * 15)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=4),  # 15 -> 60
            nn.Conv1d(64, 32, 5, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Upsample(scale_factor=4),  # 60 -> 240
            nn.Conv1d(32, 16, 5, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Upsample(scale_factor=4),  # 240 -> 960
            nn.Conv1d(16, 1, 5, padding=2),
        )

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, input_dim) -> (B, output_samples)."""
        h = self.fc(z).view(-1, 64, 15)
        return self.net(h).squeeze(1)


class BranchDecoder(nn.Module):
    """Reconstruct bandpassed waveform from a single CNN branch (training-only).

    Used for all branch decoders: delta (large CNN), hfreq (small CNN),
    theta (medium CNN), slowwave (large CNN), sawtooth (medium CNN).
    Direct gradient path to the target branch.
    Output: 480 samples (8x downsampled from 3840).
    """

    def __init__(self, input_dim: int = 128) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 32 * 15)
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=4),  # 15 -> 60
            nn.Conv1d(32, 16, 5, padding=2), nn.GroupNorm(4, 16), nn.GELU(),
            nn.Upsample(scale_factor=4),  # 60 -> 240
            nn.Conv1d(16, 8, 5, padding=2), nn.GroupNorm(4, 8), nn.GELU(),
            nn.Upsample(scale_factor=2),  # 240 -> 480
            nn.Conv1d(8, 1, 5, padding=2),
        )

    def forward(self, z: Tensor) -> Tensor:
        """z: (B, 128) -> (B, 480)."""
        h = self.fc(z).view(-1, 32, 15)
        return self.net(h).squeeze(1)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

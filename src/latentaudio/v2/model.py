# SPDX-License-Identifier: AGPL-3.0-or-later
"""LatentAudio v2 model: spectral encoder + frame-rate ConvNeXt decoder + iSTFT."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsp import InverseSTFT, num_frames

MIN_LOGVAR = -8.0
MAX_LOGVAR = 4.0


@dataclass
class V2Config:
    sample_rate: int = 44100
    length: int = 44100            # samples per sound (1 s)
    latent_dim: int = 128
    # Decoder (the part shipped to the VST)
    n_fft: int = 1024
    hop: int = 256
    dec_dim: int = 384
    dec_layers: int = 8
    dec_mlp: int = 3
    # Encoder (training / analysis only)
    enc_n_fft: int = 1024
    enc_hop: int = 256
    enc_dim: int = 256
    enc_layers: int = 4
    enc_time_steps: int = 24

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "V2Config":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


class ConvNeXtBlock(nn.Module):
    """Depthwise conv + MLP block; optionally FiLM/AdaLN-conditioned on z."""

    def __init__(self, dim: int, mlp: int, cond_dim: Optional[int] = None, layer_scale: float = 0.1):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, elementwise_affine=cond_dim is None)
        self.film = nn.Linear(cond_dim, 2 * dim) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.pw1 = nn.Linear(dim, mlp * dim)
        self.pw2 = nn.Linear(mlp * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), layer_scale))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.dw(x).transpose(1, 2)  # [B, T, C]
        x = self.norm(x)
        if self.film is not None and cond is not None:
            scale, shift = self.film(cond).unsqueeze(1).chunk(2, dim=-1)
            x = x * (1.0 + scale) + shift
        x = self.pw2(F.gelu(self.pw1(x)))
        x = (self.gamma * x).transpose(1, 2)
        return residual + x


class Decoder(nn.Module):
    """z [B, D] -> audio [B, length]. Runs at STFT frame rate, then inverse STFT."""

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        frames = num_frames(cfg.length, cfg.hop)
        bins = cfg.n_fft // 2 + 1
        self.bins = bins

        # One-shots are onset-aligned, so a learned per-frame embedding carries
        # "where in the sound am I" (envelope shape); z says "which sound".
        self.pos = nn.Parameter(torch.randn(1, cfg.dec_dim, frames) * 0.02)
        self.z_proj = nn.Linear(cfg.latent_dim, cfg.dec_dim)
        self.inp = nn.Conv1d(cfg.dec_dim, cfg.dec_dim, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(cfg.dec_dim, cfg.dec_mlp, cond_dim=cfg.latent_dim, layer_scale=1.0 / cfg.dec_layers)
            for _ in range(cfg.dec_layers)
        )
        self.norm = nn.LayerNorm(cfg.dec_dim)
        self.head = nn.Linear(cfg.dec_dim, 2 * bins)
        nn.init.normal_(self.head.weight, std=0.01)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[:bins] = -6.0  # start quiet: exp(-6) ~ -52 dB per bin
        self.istft = InverseSTFT(cfg.n_fft, cfg.hop, cfg.length)

    def spectrum(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Log-magnitude and phase [B, bins, frames]."""
        h = self.inp(self.pos + self.z_proj(z).unsqueeze(-1))
        for block in self.blocks:
            h = block(h, z)
        out = self.head(self.norm(h.transpose(1, 2))).transpose(1, 2)
        log_mag, phase = out[:, : self.bins], out[:, self.bins :]
        return log_mag, phase

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        log_mag, phase = self.spectrum(z)
        # Synthesis always in float32: bf16 overlap-add would add audible noise.
        with torch.autocast(device_type=z.device.type, enabled=False):
            log_mag, phase = log_mag.float(), phase.float()
            mag = torch.exp(torch.clamp(log_mag, max=7.0))
            real = mag * torch.cos(phase)
            imag = mag * torch.sin(phase)
            return self.istft(torch.cat([real, imag], dim=1))


class Encoder(nn.Module):
    """audio [B, length] -> (mu, logvar). Reads a log-magnitude spectrogram."""

    def __init__(self, cfg: V2Config):
        super().__init__()
        self.cfg = cfg
        bins = cfg.enc_n_fft // 2 + 1
        self.register_buffer("window", torch.hann_window(cfg.enc_n_fft), persistent=False)
        self.inp = nn.Conv1d(bins, cfg.enc_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(ConvNeXtBlock(cfg.enc_dim, 3) for _ in range(cfg.enc_layers))
        self.pool = nn.AdaptiveAvgPool1d(cfg.enc_time_steps)
        self.proj = nn.Sequential(
            nn.Linear(cfg.enc_dim * cfg.enc_time_steps, 512),
            nn.GELU(),
        )
        self.mu = nn.Linear(512, cfg.latent_dim)
        self.logvar = nn.Linear(512, cfg.latent_dim)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -3.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        spec = torch.stft(x.float(), self.cfg.enc_n_fft, self.cfg.enc_hop, self.cfg.enc_n_fft,
                          self.window, center=True, return_complex=True)
        feat = (torch.log(spec.abs() + 1e-5) + 5.0) / 5.0  # roughly [-1.3, 1+]
        h = self.inp(feat)
        for block in self.blocks:
            h = block(h)
        h = self.proj(self.pool(h).flatten(1))
        mu = self.mu(h)
        logvar = torch.clamp(self.logvar(h), MIN_LOGVAR, MAX_LOGVAR)
        return mu, logvar


class LatentAudioV2(nn.Module):
    """VAE with the same interface the rest of LatentAudio uses for v1 models."""

    model_type = "v2"

    def __init__(self, cfg: Optional[V2Config] = None):
        super().__init__()
        self.cfg = cfg or V2Config()
        self.latent_dim = self.cfg.latent_dim
        self.waveform_length = self.cfg.length
        self.encoder = Encoder(self.cfg)
        self.decoder = Decoder(self.cfg)

    # --- v1-compatible API -------------------------------------------------
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, skip_prob=None, use_generated_skips=None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decoder(z), mu, logvar, None, None

    def sample_latent(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return self.decoder(mu)

    def check_for_nan(self) -> bool:
        return any(torch.isnan(p).any() for p in self.parameters())

    def reset_nan_parameters(self) -> None:
        with torch.no_grad():
            for p in self.parameters():
                if torch.isnan(p).any():
                    p.copy_(torch.randn_like(p) * 1e-4)

    def num_parameters(self, part: Optional[nn.Module] = None) -> int:
        return sum(p.numel() for p in (part or self).parameters() if p.requires_grad)

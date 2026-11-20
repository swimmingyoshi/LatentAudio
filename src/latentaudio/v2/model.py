# SPDX-License-Identifier: AGPL-3.0-or-later
"""LatentAudio v2 model: spectral encoder + frame-rate ConvNeXt decoder + iSTFT."""

from __future__ import annotations

import math

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
    # Pitch conditioning (melodic instruments). The decoder input becomes
    # [z, p] with p = (midi - pitch_offset) / pitch_scale as the LAST element.
    pitch_conditioned: bool = False
    pitch_offset: float = 60.0
    pitch_scale: float = 24.0
    pitch_features: int = 16

    @property
    def decoder_input_dim(self) -> int:
        return self.latent_dim + (1 if self.pitch_conditioned else 0)

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
        self.pitched = cfg.pitch_conditioned
        cond_dim = cfg.latent_dim + (cfg.pitch_features if self.pitched else 0)
        self.pos = nn.Parameter(torch.randn(1, cfg.dec_dim, frames) * 0.02)
        self.z_proj = nn.Linear(cond_dim, cfg.dec_dim)
        self.inp = nn.Conv1d(cfg.dec_dim, cfg.dec_dim, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList(
            ConvNeXtBlock(cfg.dec_dim, cfg.dec_mlp, cond_dim=cond_dim, layer_scale=1.0 / cfg.dec_layers)
            for _ in range(cfg.dec_layers)
        )
        self.norm = nn.LayerNorm(cfg.dec_dim)
        # Pitched models also output a per-frame "harmonicity" (weight of the harmonic comb).
        self.head = nn.Linear(cfg.dec_dim, 2 * bins + (1 if self.pitched else 0))
        nn.init.normal_(self.head.weight, std=0.01)
        with torch.no_grad():
            self.head.bias.zero_()
            self.head.bias[:bins] = -6.0  # start quiet: exp(-6) ~ -52 dB per bin
        self.istft = InverseSTFT(cfg.n_fft, cfg.hop, cfg.length)

        if self.pitched:
            self.register_buffer("bin_hz", torch.arange(bins, dtype=torch.float32) * cfg.sample_rate / cfg.n_fft,
                                 persistent=False)
            # Seconds at each frame's start; the phase of a steady partial advances by 2*pi*f*hop/sr per frame.
            self.register_buffer("frame_sec", torch.arange(frames, dtype=torch.float32) * cfg.hop / cfg.sample_rate,
                                 persistent=False)
            self.register_buffer("feat_freqs", (2.0 ** torch.arange(cfg.pitch_features // 2, dtype=torch.float32))
                                 * math.pi, persistent=False)

    def split(self, z_in: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decoder input -> (z, normalised pitch [B, 1] or None)."""
        if not self.pitched:
            return z_in, None
        return z_in[:, : self.cfg.latent_dim], z_in[:, self.cfg.latent_dim:self.cfg.latent_dim + 1]

    def condition(self, z: torch.Tensor, p: Optional[torch.Tensor]) -> torch.Tensor:
        if p is None:
            return z
        ang = p * self.feat_freqs  # Fourier features of the pitch
        return torch.cat([z, torch.sin(ang), torch.cos(ang)], dim=1)

    def spectrum(self, z_in: torch.Tensor):
        """Log-magnitude [B, bins, T], phase [B, bins, T] and harmonicity [B, 1, T] (or None)."""
        z, p = self.split(z_in)
        cond = self.condition(z, p)
        h = self.inp(self.pos + self.z_proj(cond).unsqueeze(-1))
        for block in self.blocks:
            h = block(h, cond)
        out = self.head(self.norm(h.transpose(1, 2))).transpose(1, 2)
        log_mag, phase = out[:, : self.bins], out[:, self.bins: 2 * self.bins]
        harm = out[:, 2 * self.bins:] if self.pitched else None
        return log_mag, phase, harm

    def forward(self, z_in: torch.Tensor) -> torch.Tensor:
        log_mag, phase, harm = self.spectrum(z_in)
        # Synthesis always in float32: bf16 overlap-add would add audible noise.
        with torch.autocast(device_type=z_in.device.type, enabled=False):
            log_mag, phase = log_mag.float(), phase.float()
            if self.pitched:
                _, p = self.split(z_in.float())
                midi = p * self.cfg.pitch_scale + self.cfg.pitch_offset
                f0 = 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)               # [B, 1]
                k = torch.clamp(torch.round(self.bin_hz / f0), min=1.0)          # nearest harmonic number
                f_h = k * f0                                                     # [B, bins]
                width = 0.25 * f0 + self.cfg.sample_rate / self.cfg.n_fft        # >= one bin wide
                comb = torch.exp(-torch.square((self.bin_hz - f_h) / width))     # 1 on harmonics, ~0 between
                log_mag = log_mag + harm.float() * (comb.unsqueeze(-1) - 0.5)
                # Phase-vocoder prior: each bin advances like its nearest harmonic.
                phase = phase + 2.0 * math.pi * f_h.unsqueeze(-1) * self.frame_sec
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

    # --- pitch helpers ---------------------------------------------------
    def pitch_to_norm(self, midi: torch.Tensor) -> torch.Tensor:
        return ((midi - self.cfg.pitch_offset) / self.cfg.pitch_scale).reshape(-1, 1)

    def decoder_input(self, z: torch.Tensor, midi: Optional[torch.Tensor] = None) -> torch.Tensor:
        """z [B, D] (+ MIDI pitch [B] for pitched models) -> decoder input."""
        if not self.cfg.pitch_conditioned:
            return z
        if midi is None:
            midi = torch.full((z.shape[0],), self.cfg.pitch_offset, device=z.device)
        return torch.cat([z, self.pitch_to_norm(midi.to(z.dtype))], dim=1)

    def decode(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None,
               midi: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Accepts either a full decoder input (pitch already appended) or a bare latent.
        if self.cfg.pitch_conditioned and z.shape[1] == self.latent_dim:
            z = self.decoder_input(z, midi)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, skip_prob=None, use_generated_skips=None, midi: Optional[torch.Tensor] = None):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decoder(self.decoder_input(z, midi)), mu, logvar, None, None

    def sample_latent(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor, midi: Optional[torch.Tensor] = None) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return self.decoder(self.decoder_input(mu, midi))

    def check_for_nan(self) -> bool:
        return any(torch.isnan(p).any() for p in self.parameters())

    def reset_nan_parameters(self) -> None:
        with torch.no_grad():
            for p in self.parameters():
                if torch.isnan(p).any():
                    p.copy_(torch.randn_like(p) * 1e-4)

    def num_parameters(self, part: Optional[nn.Module] = None) -> int:
        return sum(p.numel() for p in (part or self).parameters() if p.requires_grad)

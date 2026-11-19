# SPDX-License-Identifier: AGPL-3.0-or-later
"""STFT helpers, including an inverse STFT that exports to plain ONNX.

torch.istft has no ONNX equivalent (and the VST's ONNX Runtime 1.16 has no
ISTFT op), so the inverse is written as an inverse real DFT folded into the
weights of a ConvTranspose1d, followed by division by the (constant) window
overlap envelope. Every op used is opset <= 14.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def num_frames(length: int, hop: int) -> int:
    """Frames produced by a centred STFT (torch.stft(center=True))."""
    return length // hop + 1


class InverseSTFT(nn.Module):
    """[B, 2*(n_fft/2+1), T] (real parts, then imaginary parts) -> [B, length]."""

    def __init__(self, n_fft: int, hop: int, length: int):
        super().__init__()
        self.n_fft, self.hop, self.length = n_fft, hop, length
        bins = n_fft // 2 + 1
        window = torch.hann_window(n_fft, periodic=True, dtype=torch.float64)

        # irfft as a linear map: x[n] = sum_k a_k (Re X_k cos(2pi k n/N) - Im X_k sin(2pi k n/N)) / N
        n = torch.arange(n_fft, dtype=torch.float64)
        k = torch.arange(bins, dtype=torch.float64)
        angle = 2.0 * math.pi * k[:, None] * n[None, :] / n_fft
        scale = torch.full((bins, 1), 2.0, dtype=torch.float64)
        scale[0] = 1.0
        scale[-1] = 1.0
        cos_basis = scale * torch.cos(angle) / n_fft
        sin_basis = -scale * torch.sin(angle) / n_fft
        basis = torch.cat([cos_basis, sin_basis], dim=0) * window[None, :]  # [2*bins, n_fft]

        self.ola = nn.ConvTranspose1d(2 * bins, 1, kernel_size=n_fft, stride=hop, bias=False)
        self.ola.weight.data.copy_(basis[:, None, :].float())
        self.ola.weight.requires_grad_(False)

        # Window-squared overlap envelope for the fixed number of frames.
        frames = num_frames(length, hop)
        total = n_fft + hop * (frames - 1)
        env = torch.zeros(total, dtype=torch.float64)
        w2 = window**2
        for t in range(frames):
            env[t * hop : t * hop + n_fft] += w2
        pad = n_fft // 2
        env = env[pad : pad + length].clamp_min(1e-8)
        self.register_buffer("inv_envelope", (1.0 / env).float(), persistent=False)

    def forward(self, spec_ri: torch.Tensor) -> torch.Tensor:
        y = self.ola(spec_ri).squeeze(1)  # [B, total]
        pad = self.n_fft // 2
        y = y[:, pad : pad + self.length]
        return y * self.inv_envelope


def stft_mag(x: torch.Tensor, n_fft: int, hop: int, window: torch.Tensor) -> torch.Tensor:
    """Magnitude STFT [B, bins, frames] (float32 regardless of autocast)."""
    spec = torch.stft(x.float(), n_fft, hop, n_fft, window, center=True, return_complex=True)
    return spec.abs()


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> torch.Tensor:
    """HTK-style triangular mel filters [n_mels, n_fft/2+1] (no librosa dependency)."""
    fmax = fmax or sample_rate / 2
    bins = n_fft // 2 + 1

    def hz_to_mel(f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    mels = torch.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = torch.tensor([mel_to_hz(float(m)) for m in mels])
    freqs = torch.linspace(0, sample_rate / 2, bins)
    fb = torch.zeros(n_mels, bins)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / (mid - lo).clamp_min(1e-6)
        down = (hi - freqs) / (hi - mid).clamp_min(1e-6)
        fb[i] = torch.clamp(torch.minimum(up, down), min=0.0)
    return fb

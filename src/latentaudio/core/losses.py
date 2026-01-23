# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LatentAudio - Direct Neural Audio Generation and Exploration
# Copyright (C) 2026 swimmingyoshi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# losses.py - Audio-specific loss functions with NaN protection
"""Spectral and perceptual loss functions with numerical stability."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from ..config import (
    STFT_QUALITY_RESOLUTIONS,
    STFT_QUALITY_HOPS,
    STFT_QUALITY_WIN_LENGTHS,
    STFT_FAST_RESOLUTIONS,
    STFT_FAST_HOPS,
    STFT_FAST_WIN_LENGTHS,
    STFT_MODE,
)

# Add these safety constants
EPS = 1e-8
SPECTRAL_LOSS_CLIP = 10.0
MAX_LOSS_VALUE = 1e6


class STFTLoss(nn.Module):
    """STFT loss with numerical stability fixes."""

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        window: str = "hann_window",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute STFT loss with Complex Phase awareness and Pre-Emphasis."""

        # Ensure 2D [Batch, Length] for torch.stft
        if x.dim() == 3:
            x = x.squeeze(1)
        if y.dim() == 3:
            y = y.squeeze(1)

        # 0. Pre-Emphasis (High-Frequency Boost for Transient Protection)
        # y_t = x_t - 0.97 * x_{t-1}
        x_pre = x.clone()
        y_pre = y.clone()
        x_pre[:, 1:] = x[:, 1:] - 0.97 * x[:, :-1]
        y_pre[:, 1:] = y[:, 1:] - 0.97 * y[:, :-1]

        # Add small epsilon to prevent silence issues
        x_pre = x_pre + EPS
        y_pre = y_pre + EPS

        try:
            x_stft = torch.stft(
                x_pre,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                return_complex=True,
            )
            y_stft = torch.stft(
                y_pre,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                return_complex=True,
            )
        except RuntimeError as e:
            # Fallback if STFT fails
            print(f"STFT failed: {e}, returning zero loss")
            return torch.tensor(0.0, device=x.device), torch.tensor(0.0, device=x.device)

        # 1. Magnitude Loss (Volume/Tone)
        x_mag = torch.sqrt(torch.clamp(x_stft.real**2 + x_stft.imag**2, min=0.0) + EPS)
        y_mag = torch.sqrt(torch.clamp(y_stft.real**2 + y_stft.imag**2, min=0.0) + EPS)

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.clamp(
            torch.norm(y_mag, p="fro"), min=EPS
        )
        log_mag_loss = F.l1_loss(torch.log(y_mag + EPS), torch.log(x_mag + EPS))

        # 2. Complex Loss (Phase/Timing)
        # This penalizes differences in the complex plane directly
        complex_loss = F.l1_loss(x_stft.real, y_stft.real) + F.l1_loss(x_stft.imag, y_stft.imag)

        # Combined Spectral Loss
        # We return (Spectral Convergence, Log-Magnitude + Complex Phase)
        return sc_loss, log_mag_loss + 0.5 * complex_loss


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT loss with stability improvements."""

    def __init__(
        self,
        fft_sizes: Optional[List[int]] = None,
        hop_sizes: Optional[List[int]] = None,
        win_lengths: Optional[List[int]] = None,
        window: str = "hann_window",
        mode: str = "fast",
    ):
        super().__init__()

        # Use configuration values from config.py
        if mode == "fast":
            default_fft = STFT_FAST_RESOLUTIONS  # Use config: [512]
            default_hop = STFT_FAST_HOPS  # Use config: [128]
            default_win = STFT_FAST_WIN_LENGTHS  # Use config: [512]
        else:
            default_fft = STFT_QUALITY_RESOLUTIONS
            default_hop = STFT_QUALITY_HOPS
            default_win = STFT_QUALITY_WIN_LENGTHS

        fft_sizes = fft_sizes if fft_sizes is not None else default_fft
        hop_sizes = hop_sizes if hop_sizes is not None else default_hop
        win_lengths = win_lengths if win_lengths is not None else default_win

        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_losses = nn.ModuleList(
            [STFTLoss(fs, hs, wl, window) for fs, hs, wl in zip(fft_sizes, hop_sizes, win_lengths)]
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute MR-STFT loss with averaging."""
        sc_loss = torch.tensor(0.0, device=x.device)
        log_mag_loss = torch.tensor(0.0, device=x.device)
        valid_losses = 0

        for loss_fn in self.stft_losses:
            sc, lm = loss_fn(x, y)

            # Only accumulate if not NaN
            if not (torch.isnan(sc) or torch.isinf(sc)):
                sc_loss += sc
                valid_losses += 1
            if not (torch.isnan(lm) or torch.isinf(lm)):
                log_mag_loss += lm

        # Average only over valid losses
        if valid_losses > 0:
            sc_loss /= valid_losses
            log_mag_loss /= valid_losses

        return sc_loss, log_mag_loss

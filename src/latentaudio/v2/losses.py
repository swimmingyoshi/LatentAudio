# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training losses and evaluation metrics for LatentAudio v2."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dsp import mel_filterbank, stft_mag

DEFAULT_RESOLUTIONS: Tuple[Tuple[int, int], ...] = ((2048, 512), (1024, 256), (512, 128), (256, 64), (128, 32))
LOG_EPS = 1e-5


class MultiResolutionSTFTLoss(nn.Module):
    """Spectral convergence + log-magnitude L1, averaged over resolutions.

    All STFT math runs in float32 even under autocast.
    """

    def __init__(self, resolutions: Sequence[Tuple[int, int]] = DEFAULT_RESOLUTIONS):
        super().__init__()
        self.resolutions = list(resolutions)
        for i, (n_fft, _) in enumerate(self.resolutions):
            self.register_buffer(f"win{i}", torch.hann_window(n_fft), persistent=False)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        sc_total = y_hat.new_zeros((), dtype=torch.float32)
        mag_total = y_hat.new_zeros((), dtype=torch.float32)
        for i, (n_fft, hop) in enumerate(self.resolutions):
            win = getattr(self, f"win{i}")
            m_hat = stft_mag(y_hat, n_fft, hop, win)
            m = stft_mag(y, n_fft, hop, win)
            # per-example spectral convergence, then mean
            num = torch.linalg.vector_norm((m - m_hat).flatten(1), dim=1)
            den = torch.linalg.vector_norm(m.flatten(1), dim=1).clamp_min(1e-4)
            sc_total = sc_total + (num / den).mean()
            mag_total = mag_total + F.l1_loss(torch.log(m_hat + LOG_EPS), torch.log(m + LOG_EPS))
        n = len(self.resolutions)
        return {"sc": sc_total / n, "log_mag": mag_total / n}


class MelLoss(nn.Module):
    """L1 between log-mel spectrograms (perceptual weighting toward lows/mids)."""

    def __init__(self, sample_rate: int, n_fft: int = 2048, hop: int = 256, n_mels: int = 128):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.register_buffer("fb", mel_filterbank(sample_rate, n_fft, n_mels), persistent=False)

    def log_mel(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.matmul(self.fb, stft_mag(x, self.n_fft, self.hop, self.window)) + LOG_EPS)

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self.log_mel(y_hat), self.log_mel(y))


class EnvelopeLoss(nn.Module):
    """L1 between log-RMS energy envelopes at several window sizes.

    Per-bin spectral losses give quiet low-frequency tails (808 sustains) almost
    no weight; this compares energy over time regardless of frequency, so a
    missing or too-short tail is penalised in proportion to how long it lasts.
    """

    def __init__(self, windows: Sequence[Tuple[int, int]] = ((1024, 256), (4096, 1024)), floor: float = 1e-4):
        super().__init__()
        self.windows = list(windows)
        self.floor = floor

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        total = y_hat.new_zeros((), dtype=torch.float32)
        for win, hop in self.windows:
            a = F.avg_pool1d(y_hat.float().pow(2).unsqueeze(1), win, hop).squeeze(1)
            b = F.avg_pool1d(y.float().pow(2).unsqueeze(1), win, hop).squeeze(1)
            total = total + F.l1_loss(0.5 * torch.log(a + self.floor ** 2), 0.5 * torch.log(b + self.floor ** 2))
        return total / len(self.windows)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns (kl used for the loss, raw kl) summed over dims, mean over batch.

    Free bits are applied per dimension on the batch-averaged KL, so a dimension
    is only penalised once it carries more than `free_bits` nats.
    """
    kl_dim = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())  # [B, D]
    raw = kl_dim.sum(1).mean()
    if free_bits > 0:
        kl = torch.clamp(kl_dim.mean(0), min=free_bits).sum()
    else:
        kl = raw
    return kl, raw


# ---------------------------------------------------------------------------
# Evaluation metrics (numpy-friendly, shared by v1/v2 comparisons)

@torch.no_grad()
def spectral_metrics(y_hat: torch.Tensor, y: torch.Tensor, sample_rate: int) -> Dict[str, List[float]]:
    """Per-example metrics. Lower is better except where noted.

    * lsd_db      : multi-resolution log-spectral distance in dB (RMS over bins/frames)
    * mel_db      : mean absolute log-mel difference in dB
    * loudness_db : |RMS level difference| in dB
    * env_db      : mean |difference| of the energy envelope (46 ms windows) in dB,
                    after matching overall level - i.e. attack/decay/tail shape
    """
    y_hat = y_hat.float()
    y = y.float()
    floor = -80.0  # dB re full scale: differences below this are inaudible, don't count them
    lsd = torch.zeros(y.shape[0], device=y.device)
    for n_fft, hop in ((2048, 512), (512, 128)):
        win = torch.hann_window(n_fft, device=y.device)
        # normalise by window energy so a full-scale sine reads ~0 dB
        scale = win.sum() / 2
        a = (20 * torch.log10(stft_mag(y_hat, n_fft, hop, win) / scale + 1e-12)).clamp_min(floor)
        b = (20 * torch.log10(stft_mag(y, n_fft, hop, win) / scale + 1e-12)).clamp_min(floor)
        lsd += ((a - b) ** 2).mean((1, 2)).sqrt() / 2

    mel = MelLoss(sample_rate).to(y.device)
    scale = mel.window.sum() / 2
    ma = (20 * torch.log10(torch.matmul(mel.fb, stft_mag(y_hat, mel.n_fft, mel.hop, mel.window)) / scale + 1e-12)).clamp_min(floor)
    mb = (20 * torch.log10(torch.matmul(mel.fb, stft_mag(y, mel.n_fft, mel.hop, mel.window)) / scale + 1e-12)).clamp_min(floor)
    mel_db = (ma - mb).abs().mean((1, 2))

    def rms_db(x):
        return 20 * torch.log10(x.pow(2).mean(1).sqrt() + 1e-6)

    loud = (rms_db(y_hat) - rms_db(y)).abs()

    gain = (y.pow(2).mean(1, keepdim=True).sqrt() / y_hat.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-8))
    ea = 10 * torch.log10(F.avg_pool1d((y_hat * gain).pow(2).unsqueeze(1), 2048, 512).squeeze(1) + 1e-6)
    eb = 10 * torch.log10(F.avg_pool1d(y.pow(2).unsqueeze(1), 2048, 512).squeeze(1) + 1e-6)
    env = (ea - eb).abs().mean(1)
    return {"lsd_db": lsd.tolist(), "mel_db": mel_db.tolist(), "loudness_db": loud.tolist(), "env_db": env.tolist()}

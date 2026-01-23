# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LatentAudio - Direct Neural Audio Generation and Exploration
# Copyright (C) 2024 LatentAudio Team
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
# simple_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List

from ..config import LATENT_DIM, LEAKY_RELU_SLOPE
from ..logging import logger

# CRITICAL: Anti-collapse constants
MIN_LOGVAR = -6.0  # Prevents variance collapse (exp(-6) ≈ 0.0025)
MAX_LOGVAR = 2.0  # Prevents explosion (exp(2) ≈ 7.4)


class SimpleEncoder(nn.Module):
    """Encoder with ANTI-COLLAPSE measures - proper variance initialization and bounds."""

    def __init__(self, waveform_length: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim

        # Calculate conv output dimensions dynamically
        h = waveform_length
        strides = [4, 4, 4, 4]
        for s in strides:
            h = (h + 2 * 4 - 9) // s + 1  # padding=4, kernel=9
        self.conv_out_len = h

        # Conv stack with skip connection points
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(32),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, 9, stride=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, 9, stride=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Dropout(0.15),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 256, 9, stride=4, padding=4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.15),
        )

        # Calculate flattened size dynamically
        flattened_size = 256 * self.conv_out_len

        # Shared bottleneck
        self.bottleneck = nn.Sequential(
            nn.Linear(flattened_size, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )

        # CRITICAL: Separate networks for mu and logvar
        # This prevents them from interfering with each other during training
        self.fc_mu = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, latent_dim)
        )

        self.fc_logvar = nn.Sequential(
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Linear(256, latent_dim)
        )

        logger.info(
            f"SimpleEncoder: {waveform_length} -> {latent_dim}D (conv_out={self.conv_out_len}, flat={flattened_size})"
        )

        # CRITICAL: Initialize logvar network to output reasonable variance
        self._init_logvar_network()

    def _init_logvar_network(self):
        """Initialize logvar network to prevent collapse during early training."""
        with torch.no_grad():
            # Initialize final layer bias to -1.0 (variance ≈ 0.37)
            # This gives the model a reasonable starting point
            if hasattr(self.fc_logvar[-1], "bias") and self.fc_logvar[-1].bias is not None:
                self.fc_logvar[-1].bias.fill_(-1.0)
                logger.info("Initialized logvar bias to -1.0 (variance ≈ 0.37)")

            # Small weights for logvar network to prevent large initial swings
            for layer in self.fc_logvar:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=0.3)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Forward through convs, storing skip connections
        h1 = self.conv1(x)
        h2 = self.conv2(h1)
        h3 = self.conv3(h2)
        h4 = self.conv4(h3)

        # Store skips (excluding final layer)
        skips = [h1, h2, h3]

        # Bottleneck
        h = h4.view(h4.size(0), -1)
        h = self.bottleneck(h)

        # CRITICAL: Get mu and logvar from SEPARATE networks
        mu_raw = self.fc_mu(h)
        logvar_raw = self.fc_logvar(h)

        # CRITICAL: Enforce bounds AFTER network output
        # This allows gradients to flow while preventing collapse/explosion
        mu = torch.clamp(mu_raw, -5.0, 5.0)  # Wider bounds than before
        logvar = torch.clamp(logvar_raw, MIN_LOGVAR, MAX_LOGVAR)  # Hard floor!

        return mu, logvar, skips


class SkipGenerator(nn.Module):
    """Deep Residual Skip Generator for Phase 7 Transparency."""

    def __init__(self, latent_dim: int, skip_specs: List[Tuple[int, int]]):
        super().__init__()
        self.generators = nn.ModuleList()

        for ch, length in skip_specs:
            # Each skip branch gets a mini-residual network
            self.generators.append(
                nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, ch * length),
                    nn.Unflatten(1, (ch, length)),
                    # Residual Texture Refinement
                    nn.Conv1d(ch, ch, kernel_size=7, padding=3, groups=ch, bias=False),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(ch, ch, kernel_size=1, bias=False),
                    nn.GroupNorm(1, ch),
                )
            )

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        return [gen(z) for gen in self.generators]


class SimpleDecoder(nn.Module):
    """Decoder with skip connections and adaptive sizing."""

    def __init__(
        self,
        latent_dim: int,
        waveform_length: int,
        conv_out_len: int,
        skip_shapes: List[Tuple[int, int]],
    ):
        super().__init__()
        self.waveform_length = waveform_length
        self.conv_out_len = conv_out_len

        # Matching bottleneck expansion
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(512, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(2048, 256 * conv_out_len),
            nn.LeakyReLU(0.2),
        )

        # Skip generator for hallucinations
        self.skip_generator = SkipGenerator(latent_dim, skip_shapes)

        # Skip connection adapters (1x1 convs to adjust channels)

        self.skip_adapter1 = nn.Conv1d(128, 128, 1, bias=False)  # For conv3 skip
        self.skip_adapter2 = nn.Conv1d(64, 64, 1, bias=False)  # For conv2 skip
        self.skip_adapter3 = nn.Conv1d(32, 32, 1, bias=False)  # For conv1 skip

        # 1. Smoothly stretch 4x (Layer 1)
        self.deconv1 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(256, 128, kernel_size=9, padding=4, bias=False),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, 128, affine=True),
        )

        # 2. Smoothly stretch 4x (Layer 2)
        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(256, 64, kernel_size=9, padding=4, bias=False),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, 64, affine=True),
        )

        # 3. Smoothly stretch 4x (Layer 3)
        self.deconv3 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(128, 32, kernel_size=9, padding=4, bias=False),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, 32, affine=True),
        )

        # 4. Smoothly stretch 4x (Layer 4)
        self.deconv4 = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="linear", align_corners=False),
            nn.Conv1d(64, 16, kernel_size=9, padding=4, bias=False),
            nn.LeakyReLU(0.2),
            nn.GroupNorm(8, 16, affine=True),
        )
        # Final conv to waveform
        self.final_conv = nn.Conv1d(16, 1, 7, padding=3, bias=False)

        logger.info(
            f"SimpleDecoder: {latent_dim}D -> {waveform_length} (with skip connections, bias=False)"
        )

    def forward(
        self,
        z: torch.Tensor,
        skips: Optional[List[torch.Tensor]] = None,
        use_skips: bool = True,
        use_generated_skips: bool = False,
    ) -> torch.Tensor:
        # Expand from latent
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.conv_out_len)

        # Choose skip source
        if use_generated_skips:
            curr_skips = self.skip_generator(z)
        elif skips is not None and use_skips:
            curr_skips = skips[
                ::-1
            ]  # Reverse encoder skips (shallow->deep) to decoder order (deep->shallow)
        else:
            curr_skips = []

        # Decoder layer 1
        h = self.deconv1(h)

        # Inject skip 1 (Deepest - l3)
        if len(curr_skips) >= 1:
            skip = self.skip_adapter1(curr_skips[0])
            # Align lengths
            if h.size(2) != skip.size(2):
                skip = F.interpolate(skip, size=h.size(2), mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
        else:
            h = torch.cat([h, torch.zeros_like(h)], dim=1)

        # Decoder layer 2 (l2)
        h = self.deconv2(h)

        if len(curr_skips) >= 2:
            skip = self.skip_adapter2(curr_skips[1])
            if h.size(2) != skip.size(2):
                skip = F.interpolate(skip, size=h.size(2), mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
        else:
            h = torch.cat([h, torch.zeros_like(h)], dim=1)

        # Decoder layer 3 (Shallowest - l1)
        h = self.deconv3(h)

        if len(curr_skips) >= 3:
            skip = self.skip_adapter3(curr_skips[2])
            if h.size(2) != skip.size(2):
                skip = F.interpolate(skip, size=h.size(2), mode="linear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
        else:
            h = torch.cat([h, torch.zeros_like(h)], dim=1)

        # Decoder layer 4
        h = self.deconv4(h)

        # Final conv
        out = self.final_conv(h)
        out = out.squeeze(1).unsqueeze(1)  # Ensure [B, 1, L]

        # Hard Slicing ensures EXACT output length without blurry averaging
        out = out[:, :, : self.waveform_length]
        out = out.squeeze(1)

        # Phase 7: Tanh + -1dB Headroom Buffer (0.9 multiplier)
        # This prevents the signal from grinding against the digital ceiling
        out = 0.9 * torch.tanh(out)

        return out


class SimpleFastVAE(nn.Module):
    """VAE with ANTI-COLLAPSE measures - prevents posterior collapse."""

    def __init__(self, latent_dim: int = LATENT_DIM, waveform_length: int = 44100):
        super().__init__()
        self.latent_dim = latent_dim
        self.waveform_length = waveform_length

        self.encoder = SimpleEncoder(waveform_length, latent_dim)

        # Calculate skip shapes for SkipGenerator
        # Mirroring SimpleEncoder logic
        def get_len(l):
            return (l + 2 * 4 - 9) // 4 + 1

        l1 = get_len(waveform_length)
        l2 = get_len(l1)
        l3 = get_len(l2)

        # skip_shapes should be DEEPEST first for the SkipGenerator
        skip_shapes = [(128, l3), (64, l2), (32, l1)]

        self.decoder = SimpleDecoder(
            latent_dim, waveform_length, self.encoder.conv_out_len, skip_shapes
        )

        # Improved initialization
        self.apply(self._init_weights)

        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"SimpleFastVAE: {total_params:,} parameters ({total_params/1e6:.1f}M)")
        logger.info(
            "✓ Anti-collapse measures: MIN_LOGVAR=-6.0, separate μ/σ networks, logvar bias=-1.0"
        )

    def _init_weights(self, m):
        """Improved weight initialization."""
        if isinstance(m, nn.Linear):
            # Regular linear layers
            nn.init.orthogonal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            # Convolutions: orthogonal with moderate gain
            nn.init.orthogonal_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Extra conservative for final layer
        if m == self.decoder.final_conv:
            logger.debug("Conservative initialization on final conv")
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar, _ = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        # Use generated skips for generation
        return self.decoder(
            z, skips, use_skips=skips is not None, use_generated_skips=skips is None
        )

    def forward(
        self,
        x: torch.Tensor,
        skip_prob: Optional[float] = None,
        use_generated_skips: Optional[bool] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[List[torch.Tensor]],
        Optional[List[torch.Tensor]],
    ]:
        # Decide whether to use generated skips this batch
        if use_generated_skips is None:
            use_generated_skips = False

        # Encode with skip connections
        mu, logvar, skips = self.encoder(x)

        # Reparameterize
        z = self.reparameterize(mu, logvar)

        # We always want the generated skips for the consistency loss during training
        gen_skips = self.decoder.skip_generator(z) if self.training else None

        # Decode using skip connections
        reconstructed = self.decoder(
            z, skips, use_skips=True, use_generated_skips=use_generated_skips
        )

        return reconstructed, mu, logvar, skips, gen_skips

    def sample_latent(self, batch_size: int = 1, device: str = "cpu") -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct without reparameterization (use mean)."""
        with torch.no_grad():
            mu, _, skips = self.encoder(x)
            return self.decoder(mu, skips, use_skips=True)

    def check_for_nan(self) -> bool:
        """Check if model has any NaN parameters."""
        for param in self.parameters():
            if torch.isnan(param).any():
                return True
        return False

    def reset_nan_parameters(self):
        """Reset any NaN parameters to small random values."""
        for param in self.parameters():
            if torch.isnan(param).any():
                with torch.no_grad():
                    param.copy_(torch.randn_like(param) * 0.0001)

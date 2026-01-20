# vae.py - PROPERLY FIXED with calculated skip shapes
"""Convolutional Variational Autoencoder with Skip Connections and Residual Blocks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np
from ..config import (
    LATENT_DIM, ENCODER_CONV_CHANNELS, ENCODER_CONV_KERNEL_SIZES,
    ENCODER_CONV_STRIDES, ENCODER_FC_LAYERS, DECODER_CONV_CHANNELS,
    DECODER_CONV_KERNEL_SIZES, DECODER_CONV_STRIDES, DECODER_FC_LAYERS,
    DECODER_CONV_PADDING, LEAKY_RELU_SLOPE, ENCODER_DROPOUT_RATE, 
    DECODER_DROPOUT_RATE, SKIP_CHANNELS, SKIP_DROPOUT_PROB
)
from ..logging import logger

class ResBlock1d(nn.Module):
    """1D Residual Block for capturing temporal features."""
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(8, channels) if channels % 8 == 0 else nn.GroupNorm(1, channels),
            nn.LeakyReLU(LEAKY_RELU_SLOPE),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(8, channels) if channels % 8 == 0 else nn.GroupNorm(1, channels)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SkipGenerator(nn.Module):
    """Deep Residual Skip Generator for Phase 7 Transparency."""

    def __init__(self, latent_dim: int, skip_specs: List[Tuple[int, int]]):
        super().__init__()
        self.skip_specs = skip_specs
        self.generators = nn.ModuleList()

        for ch, length in skip_specs:
            self.generators.append(
                nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, ch * length),
                    nn.Unflatten(1, (ch, length)),
                    # Texture Refinement
                    nn.Conv1d(ch, ch, kernel_size=7, padding=3, groups=ch, bias=False),
                    nn.LeakyReLU(0.2),
                    nn.Conv1d(ch, ch, kernel_size=1, bias=False),
                    nn.Tanh()
                )
            )

    def forward(self, z: torch.Tensor) -> List[torch.Tensor]:
        return [gen(z) for gen in self.generators]


class Encoder(nn.Module):
    """Convolutional Encoder with Skip Connection points."""
    def __init__(self, waveform_length: int, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.waveform_length = waveform_length
        self.latent_dim = latent_dim

        self.blocks = nn.ModuleList()
        in_ch = 1
        
        # Track skip shapes as we build
        self.skip_shapes = []
        h_len = waveform_length
        
        # Convolutional Layers
        for out_ch, k, s in zip(ENCODER_CONV_CHANNELS, ENCODER_CONV_KERNEL_SIZES, ENCODER_CONV_STRIDES):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, k, stride=s, padding=k//2),
                nn.GroupNorm(min(8, out_ch), out_ch),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                ResBlock1d(out_ch),
                nn.Dropout(ENCODER_DROPOUT_RATE)
            ))
            
            # Calculate output length
            padding = k // 2
            h_len = (h_len + 2*padding - k) // s + 1
            self.skip_shapes.append((out_ch, h_len))
            
            in_ch = out_ch
        
        # Remove last skip (it's the bottleneck input)
        self.skip_shapes = self.skip_shapes[:-1]
        
        self.bottleneck_len = 16 
        self.bottleneck_ch = in_ch
        self.pool = nn.AdaptiveAvgPool1d(self.bottleneck_len)
        
        flattened_size = self.bottleneck_len * self.bottleneck_ch
        
        # FC bottleneck
        fc_layers = []
        curr_in = flattened_size
        for out_f in ENCODER_FC_LAYERS:
            fc_layers.extend([
                nn.Linear(curr_in, out_f),
                nn.LeakyReLU(LEAKY_RELU_SLOPE),
                nn.Dropout(ENCODER_DROPOUT_RATE)
            ])
            curr_in = out_f
        
        self.fc_common = nn.Sequential(*fc_layers)
        self.fc_mu = nn.Linear(curr_in, latent_dim)
        self.fc_logvar = nn.Linear(curr_in, latent_dim)

        logger.debug(f"Encoder: {waveform_length} -> {latent_dim}D")
        logger.debug(f"Skip shapes: {self.skip_shapes}")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        skips = []
        h = x
        for block in self.blocks:
            h = block(h)
            skips.append(h)
        
        skips.pop()  # Remove bottleneck input
        
        h = self.pool(h)
        h_flat = h.view(h.size(0), -1)
        h_fc = self.fc_common(h_flat)
        
        mu = self.fc_mu(h_fc)
        logvar = self.fc_logvar(h_fc)
        
        return mu, logvar, skips


class DecoderBlock(nn.Module):
    """Unified Decoder Block handling Skip Connections."""
    def __init__(self, in_c: int, skip_c: int, out_c: int, k: int, s: int, p: int):
        super().__init__()
        self.skip_c = skip_c
        self.upsample = nn.ConvTranspose1d(in_c + skip_c, out_c, k, stride=s, padding=p, bias=False)
        self.res = ResBlock1d(out_c)
        self.act = nn.LeakyReLU(LEAKY_RELU_SLOPE)
        self.dropout = nn.Dropout(DECODER_DROPOUT_RATE)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None, skip_prob: float = 0.0) -> torch.Tensor:
        if self.skip_c > 0:
            if skip is None:
                skip = torch.zeros(x.size(0), self.skip_c, x.size(2), device=x.device)
            else:
                if self.training and torch.rand(1).item() < skip_prob:
                    skip = torch.zeros_like(skip)
                
                # Align lengths if needed
                if x.size(2) != skip.size(2):
                    x = F.interpolate(x, size=skip.size(2), mode='linear', align_corners=False)
            
            x = torch.cat([x, skip], dim=1)
        
        x = self.upsample(x)
        x = self.res(x)
        x = self.act(x)
        x = self.dropout(x)
        return x


class Decoder(nn.Module):
    """Convolutional Decoder with Skip Generator."""
    def __init__(self, latent_dim: int, waveform_length: int, bottleneck_len: int, 
                 bottleneck_ch: int, encoder_skip_shapes: List[Tuple[int, int]]):
        super().__init__()
        self.latent_dim = latent_dim
        self.waveform_length = waveform_length
        self.bottleneck_len = bottleneck_len
        self.bottleneck_ch = bottleneck_ch

        # FC upsampling
        fc_layers = []
        curr_in = latent_dim
        for out_f in reversed(DECODER_FC_LAYERS):
            fc_layers.extend([
                nn.Linear(curr_in, out_f),
                nn.LeakyReLU(LEAKY_RELU_SLOPE)
            ])
            curr_in = out_f
        
        self.fc_upsample = nn.Sequential(
            *fc_layers,
            nn.Linear(curr_in, bottleneck_len * bottleneck_ch),
            nn.LeakyReLU(LEAKY_RELU_SLOPE)
        )

        # Use encoder's actual skip shapes (reversed for decoder)
        skip_specs = encoder_skip_shapes[::-1]
        
        logger.debug(f"Decoder will use skip shapes (reversed from encoder):")
        for i, (ch, length) in enumerate(skip_specs):
            logger.debug(f"  Decoder block {i} expects skip: {ch} channels x {length} length")
        
        # Create skip generator with EXACT encoder shapes
        self.skip_generator = SkipGenerator(latent_dim, skip_specs)

        # Decoder blocks
        self.up_blocks = nn.ModuleList()
        channels = DECODER_CONV_CHANNELS
        kernels = DECODER_CONV_KERNEL_SIZES
        strides = DECODER_CONV_STRIDES
        paddings = DECODER_CONV_PADDING

        for i in range(len(channels) - 1):
            in_c = channels[i]
            out_c = channels[i+1]
            k = kernels[i]
            s = strides[i]
            p = paddings[i]
            
            # Skip channels must match encoder
            skip_c = skip_specs[i][0] if i < len(skip_specs) else 0
            self.up_blocks.append(DecoderBlock(in_c, skip_c, out_c, k, s, p))

        self.final_conv = nn.Conv1d(channels[-1], 1, kernel_size=7, padding=3, bias=False)
        
        logger.debug(f"Decoder: {latent_dim} -> {waveform_length} (bias=False)")

    def forward(self, z: torch.Tensor, skips: Optional[List[torch.Tensor]] = None, 
                skip_prob: float = 0.0, use_generated_skips: bool = False) -> torch.Tensor:
        h = self.fc_upsample(z)
        h = h.view(h.size(0), self.bottleneck_ch, self.bottleneck_len)

        # Choose skip source
        if use_generated_skips:
            curr_skips = self.skip_generator(z)
        elif skips is not None:
            curr_skips = skips[::-1]  # Reverse to match decoder order
        else:
            curr_skips = []

        for i, block in enumerate(self.up_blocks):
            skip = curr_skips[i] if i < len(curr_skips) else None
            h = block(h, skip, skip_prob)
        
        # Phase 7: Tanh + -1dB Headroom Buffer
        out = 0.9 * torch.tanh(self.final_conv(h))
        out = out.squeeze(1)
        
        if out.size(1) > self.waveform_length:
            out = out[:, :self.waveform_length]
        elif out.size(1) < self.waveform_length:
            out = F.pad(out, (0, self.waveform_length - out.size(1)))
            
        return out


class UnconditionalVAE(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, waveform_length: Optional[int] = None, 
                 skip_dropout_prob: float = SKIP_DROPOUT_PROB):
        super().__init__()
        if waveform_length is None:
            raise ValueError("waveform_length must be specified")
        
        self.latent_dim = latent_dim
        self.waveform_length = waveform_length
        self.skip_dropout_prob = skip_dropout_prob
        
        # Build encoder first to get skip shapes
        self.encoder = Encoder(waveform_length, latent_dim)
        
        # Build decoder with encoder's skip shapes
        self.decoder = Decoder(
            latent_dim, waveform_length, 
            self.encoder.bottleneck_len, 
            self.encoder.bottleneck_ch,
            self.encoder.skip_shapes  # Pass actual shapes!
        )
        
        self.apply(self._init_weights)
        logger.info(f"VAE initialized: {latent_dim}D latent, {waveform_length} samples")


    def check_for_nan(self):
        """Check if model has any NaN parameters."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"NaN detected in parameter: {name}")
                return True
        return False

    def reset_nan_parameters(self):
        """Reset any NaN parameters to small random values."""
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"Resetting NaN parameter: {name}")
                with torch.no_grad():
                    param.copy_(torch.randn_like(param) * 0.01)

    def _init_weights(self, m):
        """FIXED: More conservative weight initialization."""
        if isinstance(m, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
            # Use smaller gain to prevent gradient explosion
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # REDUCED from 0.8
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Special handling for final layer - CRITICAL FIX
        if m == self.decoder.final_conv:
            print("Applying extra-conservative init to final conv")
            # Even more conservative initialization
            nn.init.xavier_uniform_(m.weight, gain=0.01)  # Very small!
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        # Initialize batch/group norm properly
        if isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        if m == self.decoder.final_conv:
            logger.debug("Silent initialization on final layer")
            nn.init.constant_(m.weight, 1e-5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, logvar, _ = self.encoder(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, skip_prob: Optional[float] = None, 
                use_generated_skips: Optional[bool] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        if skip_prob is None:
            skip_prob = self.skip_dropout_prob
        
        # Decide whether to use generated skips this batch (if not explicitly told)
        if use_generated_skips is None:
            use_generated_skips = False # Default to False, generator will override

        mu, logvar, skips = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        
        # We always want the generated skips for the consistency loss during training
        gen_skips = self.decoder.skip_generator(z) if self.training else None
        
        # Pass use_generated_skips to decoder
        reconstructed = self.decoder(z, skips=skips, skip_prob=skip_prob, use_generated_skips=use_generated_skips)
        
        return reconstructed, mu, logvar, skips, gen_skips

    def sample_latent(self, batch_size: int = 1, device: str = 'cpu') -> torch.Tensor:
        return torch.randn(batch_size, self.latent_dim, device=device)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Use generated skips for generation
        return self.decoder(z, skips=None, use_generated_skips=True)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            mu, _, skips = self.encoder(x)
            return self.decoder(mu, skips=skips, skip_prob=0.0)
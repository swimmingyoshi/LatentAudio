#!/usr/bin/env python3
"""
deep_clean_v2.py - Signal-Aware Spectral Refinement.
Uses energy-gating to preserve long 808 tails while aggressively 
scrubbing static from silent regions and pruning latent entropy.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from latentaudio.core.generator import AdvancedAudioGenerator
from latentaudio.types import TrainingConfig, TrainingMetrics
import latentaudio.core.training as training_module
from latentaudio.core.losses import MultiResolutionSTFTLoss, STFTLoss
import latentaudio.config
from latentaudio.logging import logger
import soundfile as sf

# ============================================================================
# 1. SIGNAL-AWARE LOSS COMPONENTS
# ============================================================================

class EnergyGatedLoss(torch.nn.Module):
    """Applies dynamic penalty based on input signal level."""
    def __init__(self, silence_threshold_db: float = -60.0):
        super().__init__()
        self.threshold = 10**(silence_threshold_db / 20.0)

    def forward(self, reconstructed, original):
        # Create a mask where the original signal is "silent"
        # original shape: [B, 1, L] or [B, L]
        with torch.no_grad():
            # Use a small windowed absolute max to smooth the gate
            abs_orig = original.abs()
            if abs_orig.dim() == 3:
                pooled = F.max_pool1d(abs_orig, kernel_size=101, stride=1, padding=50)
            else:
                pooled = F.max_pool1d(abs_orig.unsqueeze(1), kernel_size=101, stride=1, padding=50).squeeze(1)
            
            is_silent = (pooled < self.threshold).float()
        
        mse = (reconstructed - original)**2
        # Apply 50x penalty to silent regions to kill "fuzz" and "ringing"
        # Apply 1x penalty to active regions to preserve 808 tails
        weighted_mse = mse * (1.0 + is_silent * 49.0)
        return weighted_mse.mean()

class HighFreqScrubLoss(torch.nn.Module):
    """Penalizes high-frequency 'fuzz' that shouldn't exist in kicks."""
    def __init__(self, sample_rate=44100, cutoff_hz=14000):
        super().__init__()
        self.n_fft = 1024
        # Calculate which bins are "high frequency"
        freqs = torch.linspace(0, sample_rate/2, self.n_fft//2 + 1)
        self.register_buffer('hf_mask', (freqs > cutoff_hz).float().view(1, -1, 1))

    def forward(self, reconstructed):
        # Push energy in the 14kHz+ range toward zero
        # This removes the "digital distortion" texture
        if reconstructed.dim() == 2:
            x = reconstructed
        else:
            x = reconstructed.squeeze(1)
            
        stft = torch.stft(x, n_fft=self.n_fft, return_complex=True, pad_mode='constant')
        mag = torch.abs(stft)
        hf_energy = (mag * self.hf_mask).mean()
        return hf_energy

# ============================================================================
# 2. MONKEY-PATCH LOGIC
# ============================================================================

def create_signal_aware_loss(gate_fn, hf_fn):
    """Returns a signal-aware monkey-patchable loss function."""
    
    original_compute_fn = training_module.compute_vae_loss

    def signal_aware_compute_vae_loss(
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kl: float = 1.0,
        config = None,
        mrstft = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # 1. Gated Waveform Loss (Respects tails, kills silence noise)
        waveform_loss = gate_fn(reconstructed, original)
        
        # 2. HF Scrubbing (Kills digital "fuzz" distortion)
        hf_loss = hf_fn(reconstructed)
        
        # 3. Spectral Loss (Standard)
        if mrstft is not None:
            sc_loss, mag_loss = mrstft(reconstructed, original)
            spectral_loss = sc_loss + mag_loss
        else:
            spectral_loss = torch.tensor(0.0, device=reconstructed.device)

        # 4. KL and Skip (Stability)
        base_losses = original_compute_fn(
            reconstructed, original, mu, logvar, 
            beta_kl=beta_kl, config=config, 
            compute_stft=False, **kwargs
        )
        
        # FINAL WEIGHTING
        # - Huge weight on Spectral for texture (15.0)
        # - High weight on Gated Waveform for silence (10.0)
        # - Moderate HF scrub (5.0)
        # - High Beta (passed in from config) to prune latent space
        total_loss = (
            waveform_loss * 10.0 + 
            spectral_loss * 15.0 + 
            hf_loss * 5.0 +
            beta_kl * base_losses['kl_loss'] + 
            base_losses['skip_loss'] * 0.1 # Almost ignore skip internal state
        )

        return {
            'total_loss': total_loss,
            'recon_loss': spectral_loss,
            'kl_loss': base_losses['kl_loss'],
            'waveform_loss': waveform_loss,
            'spectral_loss': spectral_loss,
            'skip_loss': base_losses['skip_loss'],
            'diversity_loss': hf_loss # Hijack for display
        }
    
    return signal_aware_compute_vae_loss

# ============================================================================
# 3. EXECUTION ENGINE
# ============================================================================

def signal_aware_clean(model_path, data_dir, output_path, epochs=100):
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    gen = AdvancedAudioGenerator()
    resume_state = gen.load_model(model_path)
    device = gen.device
    
    # Setup Surgical Components
    gate_loss = EnergyGatedLoss(silence_threshold_db=-55.0)
    hf_scrub = HighFreqScrubLoss(gen.config.sample_rate)
    
    print("--- Applying Signal-Aware Monkey-Patch ---")
    training_module.compute_vae_loss = create_signal_aware_loss(gate_loss, hf_scrub)
    
    # Configuration
    latentaudio.config.LOG_AUDIO_INTERVAL = 10
    
    def cleanup_callback(epoch, loss, metrics_dict):
        if epoch % 10 == 0:
            if gen.training_logger:
                gen.model.eval()
                with torch.no_grad():
                    fixed_x = torch.tensor(np.array(samples[0:1])).to(device)
                    recon, _, _, _, _ = gen.model(fixed_x)
                    gen.training_logger.log_audio('CleanV2/Recon', recon, gen.config.sample_rate, epoch)
                    gen.log_mel_spectrogram('CleanV2/Recon', recon.cpu().numpy().squeeze(), epoch)
                    
                    z = gen.model.sample_latent(1, str(device))
                    genned = gen.model.decode(z)
                    gen.training_logger.log_audio('CleanV2/Random', genned, gen.config.sample_rate, epoch)
                gen.model.train()

    # CRITICAL: High Beta (0.02) to prune the "noisy" latent dimensions
    clean_config = TrainingConfig(
        epochs=epochs,
        batch_size=resume_state.get('batch_size', 64),
        learning_rate=1e-6, # Slightly higher than before to allow weights to shift
        beta_kl=0.03,       # FORCE latent compression to drop static
        weight_decay=0.0001,
        kl_warmup_epochs=0,
        use_amp=False,
        callback=cleanup_callback
    )
    
    # Lock architecture
    latentaudio.config.SKIP_DROPOUT_PROB = 0.0
    latentaudio.config.SKIP_DROPOUT_SCHEDULE = {k: 0.0 for k in latentaudio.config.SKIP_DROPOUT_SCHEDULE}
    latentaudio.config.SKIP_SWAP_SCHEDULE = {k: 0.0 for k in latentaudio.config.SKIP_SWAP_SCHEDULE}
    
    samples = gen.load_dataset(data_dir)
    
    print(f"Starting Signal-Aware Deep Clean (Beta=0.03)...")
    gen.train(
        samples, 
        config=clean_config, 
        resume_from=resume_state,
        log_dir=Path("logs/signal_aware_clean")
    )
    
    # Save
    gen.save_model(output_path)
    print(f"Cleaned model saved: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Signal-Aware Deep Clean.")
    parser.add_argument("model", help="Path to input .pth model")
    parser.add_argument("--data", default="kicks_processed", help="Path to training data")
    parser.add_argument("--out", default="Models/1kNewKicks_SignalCleaned.pth", help="Path to save cleaned model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of cleaning epochs")
    
    args = parser.parse_args()
    signal_aware_clean(args.model, args.data, args.out, args.epochs)

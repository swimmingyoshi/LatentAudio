#!/usr/bin/env python3
"""
deep_clean_v3.py - The "No-Cheat" Fidelity Bridge.
Forces the model to use hallucinated skips during training to eliminate the 
discrepancy between TensorBoard previews and actual generative quality.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from latentaudio.core.generator import AdvancedAudioGenerator
from latentaudio.types import TrainingConfig, TrainingMetrics
import latentaudio.core.training as training_module
from latentaudio.core.losses import MultiResolutionSTFTLoss
import latentaudio.config
from latentaudio.logging import logger

# ============================================================================
# 1. MONKEY-PATCH: FORCE GENERATED SKIPS
# ============================================================================

def patch_model_forward(model):
    """Overrides the model's forward pass to ALWAYS use generated skips."""
    original_forward = model.forward

    def no_cheat_forward(x, skip_prob=None, use_generated_skips=None):
        # FORCE use_generated_skips to True for the reconstruction pass
        # This means the model must learn to make the latent-generated skips
        # sound as good as the encoder ones.
        return original_forward(x, skip_prob=skip_prob, use_generated_skips=True)
    
    model.forward = no_cheat_forward
    print("✅ Model forward pass patched: 'Cheating' disabled.")

# ============================================================================
# 2. MONKEY-PATCH: HIGH-FIDELITY LOSS
# ============================================================================

def create_no_cheat_loss():
    """Returns a loss function that prioritizes skip-alignment and spectral clarity."""
    
    original_compute_fn = training_module.compute_vae_loss

    def no_cheat_compute_vae_loss(
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        beta_kl: float = 1.0,
        config = None,
        mrstft = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        
        # Use the original logic but with specific weights
        losses = original_compute_fn(
            reconstructed, original, mu, logvar, 
            beta_kl=beta_kl, config=config, mrstft=mrstft, **kwargs
        )
        
        # RE-WEIGHTING FOR V3:
        # - High Skip Weight (20.0): Force the hallucinated skips to match reality
        # - High Spectral Weight (15.0): Clean up the fuzz
        # - Standard Waveform Weight: Maintain the thump
        
        total_loss = (
            losses['waveform_loss'] * 5.0 + 
            losses['spectral_loss'] * 15.0 + 
            losses['kl_loss'] * beta_kl + 
            losses['skip_loss'] * 20.0
        )
        
        losses['total_loss'] = total_loss
        return losses
    
    return no_cheat_compute_vae_loss

# ============================================================================
# 3. EXECUTION ENGINE
# ============================================================================

def deep_clean_v3(model_path, data_dir, output_path, epochs=100):
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    gen = AdvancedAudioGenerator()
    resume_state = gen.load_model(model_path)
    device = gen.device
    
    # Apply Patches
    patch_model_forward(gen.model)
    training_module.compute_vae_loss = create_no_cheat_loss()
    
    # Configuration
    latentaudio.config.LOG_AUDIO_INTERVAL = 10
    
    def cleanup_callback(epoch, loss, metrics_dict):
        if epoch % latentaudio.config.LOG_AUDIO_INTERVAL == 0:
            if gen.training_logger:
                gen.model.eval()
                with torch.no_grad():
                    # In V3, model.forward already uses generated skips, 
                    # so this sample will be a TRUE representation of generative quality.
                    fixed_x = torch.tensor(np.array(samples[0:1])).to(device)
                    recon, _, _, _, _ = gen.model(fixed_x)
                    
                    gen.training_logger.log_audio('TrueFidelity/Recon', recon, gen.config.sample_rate, epoch)
                    gen.log_mel_spectrogram('TrueFidelity/Recon', recon.cpu().numpy().squeeze(), epoch)
                    
                    z = gen.model.sample_latent(1, str(device))
                    genned = gen.model.decode(z)
                    gen.training_logger.log_audio('TrueFidelity/Random', genned, gen.config.sample_rate, epoch)
                gen.model.train()

    # V3 Settings: Moderate Beta to keep it clean, Low LR for refinement
    clean_config = TrainingConfig(
        epochs=epochs,
        batch_size=resume_state.get('batch_size', 64),
        learning_rate=2e-6, # Slightly higher than V2 to allow the Skip Generator to learn
        beta_kl=0.01,       # Balance between detail and cleanliness
        weight_decay=0.0001,
        kl_warmup_epochs=0,
        use_amp=False,
        callback=cleanup_callback
    )
    
    # Lock architecture (deterministic mode)
    latentaudio.config.SKIP_DROPOUT_PROB = 0.0
    latentaudio.config.SKIP_DROPOUT_SCHEDULE = {k: 0.0 for k in latentaudio.config.SKIP_DROPOUT_SCHEDULE}
    
    samples = gen.load_dataset(data_dir)
    
    print(f"Starting V3 'True Fidelity' Clean (No-Cheating Mode)...")
    gen.train(
        samples, 
        config=clean_config, 
        resume_from=resume_state,
        log_dir=Path("logs/v3_no_cheat_clean")
    )
    
    # Save
    gen.save_model(output_path)
    print(f"V3 Cleaned model saved: {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V3 No-Cheat Deep Clean.")
    parser.add_argument("model", help="Path to input .pth model")
    parser.add_argument("--data", default="kicks_processed", help="Path to training data")
    parser.add_argument("--out", default="Models/1kNewKicks_V3_FINAL.pth", help="Path to save cleaned model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of cleaning epochs")
    
    args = parser.parse_args()
    deep_clean_v3(args.model, args.data, args.out, args.epochs)

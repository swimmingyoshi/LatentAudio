#!/usr/bin/env python3
"""
polish_model.py - High-Fidelity Refinement Script for LatentAudio.
Targeted at cleaning up reconstruction artifacts after the latent space is organized.
"""

import os
import torch
import numpy as np
from pathlib import Path
from latentaudio.core.generator import AdvancedAudioGenerator
from latentaudio.types import GeneratorConfig, TrainingConfig
from latentaudio.logging import logger

def polish(model_path, data_dir, output_path, epochs=100):
    model_path = Path(model_path)
    data_dir = Path(data_dir)
    
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    # 1. Initialize Generator and Load Model
    print(f"--- Starting Polish Phase for {model_path.name} ---")
    # We load with the existing config from the checkpoint
    generator = AdvancedAudioGenerator()
    resume_state = generator.load_model(model_path)
    
    # 2. Configure "Polish" settings
    # - Low Beta (focus on reconstruction)
    # - No skip dropout (maximum fidelity)
    # - Low fixed learning rate
    # - Disable AMP for absolute precision if needed (optional)
    polish_config = TrainingConfig(
        epochs=epochs,
        batch_size=resume_state.get('batch_size', 64),
        learning_rate=1e-5,  # Micro-learning rate
        beta_kl=0.001,       # Reduced KL pressure
        weight_decay=0.0001, # Minimal regularization
        kl_warmup_epochs=0,  # No warmup
        waveform_loss_weight=15.0, # Slight boost to waveform MSE
        spectral_loss_weight=3.0,  # Boost to spectral detail
        use_amp=False        # Maximum numerical precision
    )
    
    # Override dynamic settings in the core training logic for this run
    # (Note: skip_dropout is handled by the training loop using config)
    # We will monkey-patch the schedule to force 0.0 dropout
    import latentaudio.config
    latentaudio.config.SKIP_DROPOUT_SCHEDULE = {
        'phase1_epochs': 0, 'phase1_dropout': 0.0,
        'phase2_epochs': 0, 'phase2_dropout': 0.0,
        'phase3_dropout': 0.0
    }
    latentaudio.config.SKIP_SWAP_SCHEDULE = {
        'phase1_epochs': 0, 'phase1_swap': 0.0,
        'phase2_epochs': 0, 'phase2_swap': 0.0,
        'phase3_swap': 0.0
    }

    # 3. Load Dataset
    print(f"Loading data from {data_dir}...")
    samples = generator.load_dataset(data_dir)
    
    # 4. Run Refinement
    print("Beginning polish training...")
    generator.train(
        samples, 
        config=polish_config, 
        resume_from=resume_state,
        log_dir=Path("logs/polish_run")
    )
    
    # 5. Save Final Result
    print(f"Polish complete. Saving to {output_path}")
    generator.save_model(output_path)
    print("✨ Polish Phase Successful! ✨")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Polish a trained LatentAudio model.")
    parser.add_argument("model", help="Path to input .pth model")
    parser.add_argument("--data", default="kicks_processed", help="Path to training data")
    parser.add_argument("--out", default="Models/1kNewKicks_Polished.pth", help="Path to save polished model")
    parser.add_argument("--epochs", type=int, default=150, help="Number of polish epochs")
    
    args = parser.parse_args()
    polish(args.model, args.data, args.out, args.epochs)

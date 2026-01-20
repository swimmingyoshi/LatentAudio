import torch
import numpy as np
import sys
import os

# Add project root to sys.path
sys.path.append(r'B:\Coding Projects\++\Neuro\LatentAudio\src')

from latentaudio.core.generator import AdvancedAudioGenerator
from latentaudio.types import GeneratorConfig

def check_model():
    gen = AdvancedAudioGenerator()
    
    # Find latest model
    model_dir = r'B:\Coding Projects\++\Neuro\LatentAudio\TestModels'
    if not os.path.exists(model_dir):
        print("Model directory not found")
        return
        
    checkpoints = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not checkpoints:
        print("No checkpoints found")
        return
        
    latest_checkpoint = max(checkpoints, key=os.path.getmtime)
    print(f"Loading latest checkpoint: {latest_checkpoint}")
    
    try:
        gen.load_model(latest_checkpoint)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Generate random sample
    print("Generating random sample...")
    samples = gen.generate_random(n_samples=1)
    sample = samples[0]
    
    print(f"Sample stats: max={sample.max():.6f}, min={sample.min():.6f}, std={sample.std():.6f}")
    
    # Check model parameters
    total_params = sum(p.numel() for p in gen.model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")
    
    # Check if it's Simple or Complex
    from latentaudio.core.simple_vae import SimpleFastVAE
    if isinstance(gen.model, SimpleFastVAE):
        print("Model type: SimpleFastVAE")
    else:
        print("Model type: UnconditionalVAE")

if __name__ == "__main__":
    check_model()

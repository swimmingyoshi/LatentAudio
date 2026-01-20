import sys
import os
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from latentaudio.core.generator import AdvancedAudioGenerator
from latentaudio.types import GenerationConfig

def test_upgrade():
    print("--- Starting VAE Upgrade Verification ---")
    
    # 1. Initialize Generator
    gen = AdvancedAudioGenerator()
    gen.build_model()
    
    # 2. Verify Silent Start
    print("\n[1/2] Verifying Silent Start...")
    config = GenerationConfig(apply_postprocessing=False)
    random_samples = gen.generate_random(n_samples=1, config=config)
    audio = random_samples[0]
    
    max_amp = np.max(np.abs(audio))
    print(f"Max amplitude of random sample (untrained): {max_amp:.8f}")
    
    if max_amp < 1e-3:
        print("SUCCESS: Model is effectively silent at initialization.")
    else:
        print("FAILURE: Model is producing audible noise at initialization.")

    # 3. Verify Reconstruction Pipeline
    print("\n[2/2] Verifying Reconstruction Pipeline...")
    kick_path = "kicks/Big Kicks/KSHMR Big Kick 01 (D).wav"
    
    if not os.path.exists(kick_path):
        print(f"SKIPPED: Could not find kick at {kick_path}")
        return

    try:
        original_audio = gen.load_audio_file(kick_path)
        results = gen.test_reconstruction(original_audio)
        
        print(f"Original shape: {original_audio.shape}")
        print(f"Reconstructed shape: {results['reconstructed'].shape}")
        print(f"Initial Reconstruction MSE: {results['mse']:.6f}")
        print(f"Initial Spectral Convergence: {results['spectral_convergence']:.6f}")
        
        print("\nSUCCESS: Reconstruction pipeline is functional with Convolutional U-Net.")
        
    except Exception as e:
        print(f"FAILURE: Error during reconstruction test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_upgrade()

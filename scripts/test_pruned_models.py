import torch
import time
import sys
import os
from pathlib import Path

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), "src"))

from latentaudio.core.generator import AdvancedAudioGenerator

MODELS = {
    "FP32": r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_PRUNED_FP32.pth",
    "FP16": r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_PRUNED_FP16.pth"
}

def test_inference():
    # Force fresh init
    gen = AdvancedAudioGenerator()
    
    for name, path in MODELS.items():
        print(f"\n{'='*20} Testing {name} Model {'='*20}")
        
        # 1. Load Speed
        start_load = time.perf_counter()
        try:
            # We use the generator's load_model which handles architecture setup
            gen.load_model(path)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue
        end_load = time.perf_counter()
        print(f"Load Time: {end_load - start_load:.2f}s")
        
        # 2. Performance (GPU)
        # Note: AdvancedAudioGenerator.load_model sets gen.device and gen.model
        z = torch.randn(1, gen.config.latent_dim).to(gen.device)
        
        # For FP16, we need to ensure the model and input are both half-precision
        if name == "FP16":
            gen.model.half()
            z = z.half()
        else:
            gen.model.float()
            z = z.float()
            
        gen.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = gen.model.decode(z)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_inf = time.perf_counter()
            iters = 100
            for _ in range(iters):
                _ = gen.model.decode(z)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            latency = (time.perf_counter() - start_inf) / iters
            print(f"Inference Latency: {latency*1000:.2f}ms")
            print(f"Real-time Factor: {1.0/latency:.1f}x")

        # 3. Memory usage (approximate)
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024**2)
            print(f"Approx. VRAM Usage: {vram:.2f} MB")

if __name__ == "__main__":
    test_inference()

import argparse
import torch
import time
import sys
from pathlib import Path

# Ensure we can import from src regardless of the current working directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentaudio.core.generator import AdvancedAudioGenerator


def test_inference(MODELS: dict):
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
    parser = argparse.ArgumentParser(
        description="Benchmark load time and decode latency of pruned checkpoints (from prune_model.py)."
    )
    parser.add_argument("--fp32", type=Path, help="Path to the *_PRUNED_FP32.pth checkpoint")
    parser.add_argument("--fp16", type=Path, help="Path to the *_PRUNED_FP16.pth checkpoint")
    args = parser.parse_args()
    models = {name: str(p) for name, p in (("FP32", args.fp32), ("FP16", args.fp16)) if p}
    if not models:
        parser.error("pass at least one of --fp32 / --fp16")
    test_inference(models)

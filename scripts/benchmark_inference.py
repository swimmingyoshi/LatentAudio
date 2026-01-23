import torch
import time
import os
import sys

# Ensure we can import from src
sys.path.append(os.path.join(os.getcwd(), "src"))

from latentaudio.core.generator import AdvancedAudioGenerator

MODEL_PATH = r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_MAPPED.pth"

def benchmark():
    print(f"Loading model from {MODEL_PATH}...")
    load_start = time.perf_counter()
    
    # Initialize generator
    gen = AdvancedAudioGenerator()
    try:
        checkpoint = gen.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    load_end = time.perf_counter()
    
    print(f"Model loaded in {load_end - load_start:.2f}s")
    
    # Calculate true model size (weights only)
    param_count = sum(p.numel() for p in gen.model.parameters())
    weight_size_mb = sum(p.numel() * p.element_size() for p in gen.model.parameters()) / (1024 * 1024)
    print(f"Model Parameters: {param_count:,}")
    print(f"Weight Size (In-Memory): {weight_size_mb:.2f} MB")
    
    # Prepare latent vector
    latent_dim = gen.config.latent_dim
    print(f"Latent Dimension: {latent_dim}")
    z = torch.randn(1, latent_dim).to(gen.device)
    
    # GPU Benchmark
    print("\n--- GPU Benchmark ---")
    if torch.cuda.is_available():
        # Warmup
        for _ in range(10):
            _ = gen.model.decode(z)
        torch.cuda.synchronize()
        
        start = time.perf_counter()
        iterations = 100
        for _ in range(iterations):
            _ = gen.model.decode(z)
        torch.cuda.synchronize()
        gpu_time = (time.perf_counter() - start) / iterations
        print(f"GPU Latency: {gpu_time*1000:.2f} ms")
        print(f"GPU RTF: {1.0/gpu_time:.1f}x (Real-time Factor)")
    else:
        print("CUDA not available.")

    # CPU Benchmark
    print("\n--- CPU Benchmark ---")
    gen.model.to('cpu')
    z_cpu = z.to('cpu')
    
    # Warmup
    for _ in range(3):
        _ = gen.model.decode(z_cpu)
        
    start = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        _ = gen.model.decode(z_cpu)
    cpu_time = (time.perf_counter() - start) / iterations
    
    print(f"CPU Latency: {cpu_time*1000:.2f} ms")
    print(f"CPU RTF: {1.0/cpu_time:.1f}x (Real-time Factor)")

if __name__ == "__main__":
    benchmark()

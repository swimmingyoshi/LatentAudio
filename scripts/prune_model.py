import torch
import os
from pathlib import Path

def prune_model():
    source = Path(r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_MAPPED.pth")
    target_fp32 = Path(r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_PRUNED_FP32.pth")
    target_fp16 = Path(r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_PRUNED_FP16.pth")

    if not source.exists():
        print(f"Error: Source model not found at {source}")
        return

    print(f"Loading heavy checkpoint: {source.name}...")
    # Load to CPU to avoid VRAM issues during pruning
    ckpt = torch.load(str(source), map_location="cpu", weights_only=False)

    # Keys to keep for inference and exploration (Sound Map)
    inference_keys = [
        'model_state', 'model_type', 'sample_rate', 'duration', 
        'waveform_length', 'latent_dim', 'explorer_data', 'metadata'
    ]

    # Create the base pruned dictionary (FP32)
    pruned_ckpt = {k: ckpt[k] for k in inference_keys if k in ckpt}
    
    # --- Save FP32 Version ---
    print(f"Saving FP32 pruned model to {target_fp32.name}...")
    torch.save(pruned_ckpt, str(target_fp32))
    
    # --- Prepare and Save FP16 Version ---
    print("Converting model weights to float16...")
    # Copy to avoid modifying the fp32 dictionary in memory
    import copy
    fp16_ckpt = copy.deepcopy(pruned_ckpt)
    
    # Convert all model tensors to half precision
    for k, v in fp16_ckpt['model_state'].items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            fp16_ckpt['model_state'][k] = v.half()
    
    # If explorer_data contains tensors (latents), convert them too
    if 'explorer_data' in fp16_ckpt and fp16_ckpt['explorer_data'] is not None:
        if 'cached_latents' in fp16_ckpt['explorer_data']:
            latents = fp16_ckpt['explorer_data']['cached_latents']
            if isinstance(latents, torch.Tensor):
                fp16_ckpt['explorer_data']['cached_latents'] = latents.half()
            elif hasattr(latents, 'astype'): # numpy
                import numpy as np
                fp16_ckpt['explorer_data']['cached_latents'] = latents.astype(np.float16)

    print(f"Saving FP16 pruned model to {target_fp16.name}...")
    torch.save(fp16_ckpt, str(target_fp16))

    # Verification of sizes
    s_size = source.stat().st_size / (1024**3)
    f32_size = target_fp32.stat().st_size / (1024**2)
    f16_size = target_fp16.stat().st_size / (1024**2)

    print("\n" + "="*40)
    print("PRUNING COMPLETE")
    print("="*40)
    print(f"Original:  {s_size:.2f} GB")
    print(f"FP32:      {f32_size:.2f} MB")
    print(f"FP16:      {f16_size:.2f} MB")
    print("="*40)
    print("Original file preserved.")

if __name__ == "__main__":
    prune_model()

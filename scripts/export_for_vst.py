import torch
import torch.nn as nn
import numpy as np
import json
import os
from pathlib import Path
from sklearn.decomposition import PCA

# Ensure we can import from src
import sys
sys.path.append(os.path.join(os.getcwd(), "src"))

from latentaudio.core.generator import AdvancedAudioGenerator
from latentaudio.core.simple_vae import SimpleFastVAE

SOURCE_MODEL = r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_PRUNED_FP16.pth"
# Updated EXPORT_DIR to point to the NeuroClone VST source folder
EXPORT_DIR = Path(r"B:\Coding Projects\++\NeuroClone\LatentAudioVST\Source\VST_Export")
EXPORT_DIR.mkdir(exist_ok=True, parents=True)

def export():
    print(f"Loading pruned model: {SOURCE_MODEL}...")
    ckpt = torch.load(SOURCE_MODEL, map_location="cpu", weights_only=False)
    
    latent_dim = ckpt.get('latent_dim', 128)
    waveform_length = ckpt.get('waveform_length', 44100)
    
    # 1. --- ISOLATE DECODER ---
    print("Initializing SimpleFastVAE to extract Decoder structure...")
    full_model = SimpleFastVAE(latent_dim=latent_dim, waveform_length=waveform_length)
    full_model.load_state_dict(ckpt['model_state'])
    decoder = full_model.decoder
    decoder.eval()

    # Create a wrapper to force use_generated_skips=True
    class VSTDecoderWrapper(nn.Module):
        def __init__(self, target_decoder):
            super().__init__()
            self.decoder = target_decoder
            
        def forward(self, z):
            # Force texture generation (hallucination) ON
            return self.decoder(z, use_generated_skips=True)
            
    vst_model = VSTDecoderWrapper(decoder)
    vst_model.eval()

    # Initialize map_path to None to avoid unbound error if no latents found
    map_path = None

    # 2. --- EXPORT TO ONNX ---
    print("Exporting Decoder (with Texture Generator) to ONNX...")
    onnx_path = EXPORT_DIR / "latent_decoder.onnx"
    dummy_input = torch.randn(1, latent_dim)
    
    # We use dynamic axes so the VST can technically batch if needed
    torch.onnx.export(
        vst_model,
        (dummy_input,),
        str(onnx_path),
        input_names=['latent_input'],
        output_names=['audio_output'],
        dynamic_axes={'latent_input': {0: 'batch_size'}, 'audio_output': {0: 'batch_size'}},
        opset_version=14
    )
    print(f"DONE: ONNX model saved to {onnx_path.name}")

    # 3. --- PROJECT SOUND MAP (2D) ---
    print("Projecting Sound Map to 2D (matching GUI)...")
    explorer_data = ckpt.get('explorer_data', {})
    latents_list = explorer_data.get('cached_latents')
    labels = explorer_data.get('cached_labels', [])

    if latents_list:
        latents_np = np.array(latents_list)
        
        # Use LatentProjector to match Python GUI (UMAP/t-SNE)
        from latentaudio.explorer.projector import LatentProjector
        # Initialize projector (LatentProjector defaults to UMAP if available, else t-SNE)
        projector = LatentProjector(latent_dim=latent_dim, method='auto')
        
        print(f"Fitting projector on {len(latents_np)} points...")
        # Fit and transform
        projector.fit(latents_np)
        coords_2d = projector.transform(latents_np)
        
        # Normalize coordinates to [-1, 1] for the VST
        c_min, c_max = coords_2d.min(axis=0), coords_2d.max(axis=0)
        c_range = c_max - c_min
        # Avoid divide by zero if range is 0
        c_range[c_range == 0] = 1.0 
        
        coords_normalized = 2 * (coords_2d - c_min) / c_range - 1
        
        map_points = []
        n_labels = len(labels) if labels is not None else 0
        for i in range(len(coords_normalized)):
            label = labels[i] if (labels is not None and i < n_labels) else f"Sample_{i}"
            map_points.append({
                "label": label,
                "x": float(coords_normalized[i, 0]),
                "y": float(coords_normalized[i, 1]),
                "latent": latents_list[i] # The full 128D vector
            })
            
        map_data = {
            "latent_dim": latent_dim,
            "sample_rate": ckpt.get('sample_rate', 44100),
            "points": map_points,
            # Note: PCA components removed as UMAP/t-SNE are non-linear
        }
        
        map_path = EXPORT_DIR / "sound_map.json"
        with open(map_path, 'w') as f:
            json.dump(map_data, f)
        print(f"DONE: Sound Map (2D) saved to {map_path.name}")
    else:
        print("WARNING: No latents found in explorer_data to project.")

    # 4. --- SAVE DECODER WEIGHTS (PYTORCH) ---
    decoder_path = EXPORT_DIR / "decoder_weights.pth"
    # Keep it FP16 for the VST
    decoder_state = {k.replace('decoder.', ''): v.half() for k, v in ckpt['model_state'].items() if k.startswith('decoder.')}
    torch.save(decoder_state, str(decoder_path))
    print(f"DONE: Decoder-only weights saved to {decoder_path.name}")

    print("\n" + "="*40)
    print("VST EXPORT COMPLETE")
    print(f"Directory: {EXPORT_DIR}")
    if map_path and map_path.exists():
        print(f"ONNX Model: {onnx_path.stat().st_size / 1024**2:.1f} MB")
        print(f"JSON Map: {map_path.stat().st_size / 1024:.1f} KB")
    print("="*40)

if __name__ == "__main__":
    export()

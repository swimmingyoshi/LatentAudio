"""
PTH2PT.py - Robust Export from Full Checkpoint
"""

import torch
import torch.nn as nn
import sys
import os
from pathlib import Path

# 1. Setup Paths
SRC_PATH = r"B:\Coding Projects\++\NeuroClone\LatentAudio\src"
MODEL_PATH = r"B:\Coding Projects\++\Neuro\LatentAudio\TestModels\VVV-Y1_PRUNED_FP32.pth"
EXPORT_DIR = r"B:\Coding Projects\++\NeuroClone\LatentAudioVST\Source\VST_Export"
EXPORT_PATH = os.path.join(EXPORT_DIR, "decoder_model.pt")

# Add src to python path so we can import the architecture
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from latentaudio.core.simple_vae import SimpleFastVAE
    print(f"Successfully imported SimpleFastVAE from {SRC_PATH}")
except ImportError as e:
    print(f"ERROR: Could not import SimpleFastVAE. Check path: {SRC_PATH}")
    print(e)
    sys.exit(1)

# Wrapper to force generated skips (Hallucination Mode)
class VSTDecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, z):
        # Force texture generation (hallucination) ON by passing use_generated_skips=True
        return self.decoder(z, use_generated_skips=True)

# Create a trace-friendly version of the decoder
class TraceFriendlyDecoder(nn.Module):
    """Decoder without dynamic size checking for better TorchScript compatibility."""

    def __init__(self, decoder):
        super().__init__()
        # Copy all layers from the original decoder
        self.fc = decoder.fc
        self.skip_generator = decoder.skip_generator
        self.skip_adapter1 = decoder.skip_adapter1
        self.skip_adapter2 = decoder.skip_adapter2
        self.skip_adapter3 = decoder.skip_adapter3
        self.deconv1 = decoder.deconv1
        self.deconv2 = decoder.deconv2
        self.deconv3 = decoder.deconv3
        self.deconv4 = decoder.deconv4
        self.final_conv = decoder.final_conv
        self.waveform_length = decoder.waveform_length
        self.conv_out_len = decoder.conv_out_len

    def forward(self, z):
        # Force use_generated_skips=True (no conditional logic)
        # Expand from latent
        h = self.fc(z)
        h = h.view(h.size(0), 256, self.conv_out_len)

        # Generate skips (always use generated skips for VST)
        curr_skips = self.skip_generator(z)

        # Decoder layer 1 - always concatenate (no size checking)
        h = self.deconv1(h)
        # Use the first generated skip (deepest)
        skip1 = self.skip_adapter1(curr_skips[0])
        # Assume sizes match (they should in this architecture)
        h = torch.cat([h, skip1], dim=1)

        # Decoder layer 2
        h = self.deconv2(h)
        skip2 = self.skip_adapter2(curr_skips[1])
        h = torch.cat([h, skip2], dim=1)

        # Decoder layer 3
        h = self.deconv3(h)
        skip3 = self.skip_adapter3(curr_skips[2])
        h = torch.cat([h, skip3], dim=1)

        # Decoder layer 4
        h = self.deconv4(h)

        # Final conv
        out = self.final_conv(h)
        out = out.squeeze(1).unsqueeze(1)

        # Hard Slicing ensures EXACT output length
        out = out[:, :, :self.waveform_length]
        out = out.squeeze(1)

        # Tanh + headroom buffer
        out = 0.9 * torch.tanh(out)

        return out

def main():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Loading checkpoint: {MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return

    # Load full checkpoint
    # weights_only=False because we are loading a full dictionary structure, not just state_dict
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    # Extract config
    latent_dim = ckpt.get('latent_dim', 128)
    waveform_length = ckpt.get('waveform_length', 44100)
    print(f"Model Config: latent_dim={latent_dim}, waveform_length={waveform_length}")

    # Initialize Model
    print("Initializing SimpleFastVAE...")
    full_model = SimpleFastVAE(latent_dim=latent_dim, waveform_length=waveform_length)

    # Load state dict
    if 'model_state' in ckpt:
        print("Loading state_dict from 'model_state' key...")
        full_model.load_state_dict(ckpt['model_state'])
    else:
        print("Loading state_dict directly...")
        full_model.load_state_dict(ckpt)

    full_model.eval()

    # Extract Decoder
    print("Extracting decoder...")
    decoder = full_model.decoder

    # Create trace-friendly version
    print("Creating trace-friendly decoder...")
    trace_decoder = TraceFriendlyDecoder(decoder)
    trace_decoder.eval()

    # Create Dummy Input
    dummy_input = torch.randn(1, latent_dim)

    # Verify Inference
    print("Verifying inference in Python...")
    with torch.no_grad():
        try:
            output = vst_model(dummy_input)
            print(f"Inference Successful!")
            print(f"Input shape: {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Output stats: min={output.min():.4f}, max={output.max():.4f}, mean={output.mean():.4f}")

            if output.shape[-1] != waveform_length:
                print(f"WARNING: Output length {output.shape[-1]} != expected {waveform_length}")
        except Exception as e:
            print(f"ERROR during Python inference: {e}")
            return

    # Verify with different inputs
    print("Testing with multiple inputs...")
    test_inputs = [
        torch.randn(1, latent_dim),
        torch.zeros(1, latent_dim),
        torch.ones(1, latent_dim) * 0.5
    ]

    for i, test_input in enumerate(test_inputs):
        try:
            with torch.no_grad():
                test_output = vst_model(test_input)
                print(f"Test {i+1}: OK - shape {test_output.shape}")
        except Exception as e:
            print(f"Test {i+1} FAILED: {e}")
            return

    # Try tracing first
    print("Tracing model with torch.jit.trace...")
    traced_model = None
    try:
        traced_model = torch.jit.trace(vst_model, dummy_input)
        print("Tracing successful.")
    except Exception as e:
        print(f"Tracing failed: {e}")
        print("Falling back to torch.jit.script...")
        try:
            traced_model = torch.jit.script(vst_model)
            print("Script compilation successful.")
        except Exception as e2:
            print(f"Script compilation also failed: {e2}")
            return

    # Verify traced model works
    print("Verifying traced model...")
    try:
        with torch.no_grad():
            traced_output = traced_model(dummy_input)
            print(f"Traced model inference OK - shape: {traced_output.shape}")

            # Compare with original
            original_output = vst_model(dummy_input)
            diff = torch.abs(traced_output - original_output).mean()
            print(f"Output difference (traced vs original): {diff:.6f}")

            if diff > 1e-5:
                print("WARNING: Large difference between traced and original model!")
    except Exception as e:
        print(f"ERROR: Traced model verification failed: {e}")
        return

    # Save
    print(f"Saving to {EXPORT_PATH}...")
    os.makedirs(EXPORT_DIR, exist_ok=True)
    traced_model.save(EXPORT_PATH)

    # Verify File
    if os.path.exists(EXPORT_PATH):
        size_mb = os.path.getsize(EXPORT_PATH) / (1024 * 1024)
        print(f"SUCCESS! Saved decoder_model.pt ({size_mb:.2f} MB)")
    else:
        print("ERROR: File was not saved.")

if __name__ == "__main__":
    main()

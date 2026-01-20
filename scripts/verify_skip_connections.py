#!/usr/bin/env python3
"""
verify_skip_connections.py - Verify skip connection shapes in VAE
This script comprehensively tests the skip connection architecture.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from latentaudio.config import (
    LATENT_DIM, WAVEFORM_LENGTH, ENCODER_CONV_CHANNELS, 
    ENCODER_CONV_KERNEL_SIZES, ENCODER_CONV_STRIDES,
    DECODER_CONV_CHANNELS, DECODER_CONV_KERNEL_SIZES, 
    DECODER_CONV_STRIDES, DECODER_CONV_PADDING
)
from latentaudio.core.vae import UnconditionalVAE

def calculate_encoder_shapes(waveform_length):
    """Calculate expected encoder output shapes manually."""
    print("\n" + "=" * 80)
    print("MANUAL CALCULATION - Encoder Forward Pass")
    print("=" * 80)
    
    h_len = waveform_length
    shapes = []
    
    for i, (out_ch, k, s) in enumerate(zip(ENCODER_CONV_CHANNELS, 
                                           ENCODER_CONV_KERNEL_SIZES, 
                                           ENCODER_CONV_STRIDES)):
        padding = k // 2
        h_len = (h_len + 2*padding - k) // s + 1
        shapes.append((out_ch, h_len))
        print(f"Encoder block {i}: in_ch={ENCODER_CONV_CHANNELS[i-1] if i>0 else 1}, "
              f"out_ch={out_ch}, kernel={k}, stride={s}, padding={padding}")
        print(f"  -> Output shape: ({out_ch}, {h_len})")
    
    return shapes, h_len

def calculate_decoder_shapes(start_length):
    """Calculate expected decoder output shapes manually."""
    print("\n" + "=" * 80)
    print("MANUAL CALCULATION - Decoder Forward Pass")
    print("=" * 80)
    
    h_len = start_length
    shapes = []
    
    for i, (out_ch, k, s, p) in enumerate(zip(DECODER_CONV_CHANNELS, 
                                               DECODER_CONV_KERNEL_SIZES, 
                                               DECODER_CONV_STRIDES,
                                               DECODER_CONV_PADDING)):
        h_len = (h_len - 1) * s - 2*p + k
        shapes.append((out_ch, h_len))
        print(f"Decoder block {i}: in_ch={DECODER_CONV_CHANNELS[i]}, "
              f"out_ch={DECODER_CONV_CHANNELS[i+1] if i+1 < len(DECODER_CONV_CHANNELS) else 64}, "
              f"kernel={k}, stride={s}, padding={p}")
        print(f"  -> Output shape: ({DECODER_CONV_CHANNELS[i+1] if i+1 < len(DECODER_CONV_CHANNELS) else 64}, {h_len})")
    
    return shapes, h_len

def verify_model():
    """Comprehensive verification of skip connections."""
    
    print("\n" + "=" * 80)
    print("SKIP CONNECTION VERIFICATION")
    print("=" * 80)
    print(f"Waveform Length: {WAVEFORM_LENGTH}")
    print(f"Latent Dim: {LATENT_DIM}")
    
    # Manual calculation
    encoder_shapes, bottleneck_len = calculate_encoder_shapes(WAVEFORM_LENGTH)
    
    # Remove last shape (bottleneck input)
    encoder_skip_shapes = encoder_shapes[:-1]
    print(f"\nEncoder produces {len(encoder_skip_shapes)} skip connections:")
    for i, (ch, length) in enumerate(encoder_skip_shapes):
        print(f"  Skip {i}: ({ch}, {length})")
    
    # Calculate decoder
    decoder_shapes, final_len = calculate_decoder_shapes(bottleneck_len)
    print(f"\nDecoder has {len(DECODER_CONV_CHANNELS) - 1} upsampling blocks")
    
    # Build and test actual model
    print("\n" + "=" * 80)
    print("ACTUAL MODEL TEST")
    print("=" * 80)
    
    device = 'cpu'
    model = UnconditionalVAE(latent_dim=LATENT_DIM, waveform_length=WAVEFORM_LENGTH).to(device)
    
    # Test encoder
    x = torch.randn(2, WAVEFORM_LENGTH).to(device)
    print(f"\nInput shape: {x.shape}")
    
    with torch.no_grad():
        mu, logvar, skips = model.encoder(x)
        
    print(f"\nEncoder output:")
    print(f"  mu shape: {mu.shape}")
    print(f"  logvar shape: {logvar.shape}")
    print(f"  Number of skips: {len(skips)}")
    
    print(f"\nEncoder skip shapes (actual):")
    for i, skip in enumerate(skips):
        print(f"  Skip {i}: {skip.shape}")
    
    # Test SkipGenerator
    z = torch.randn(2, LATENT_DIM).to(device)
    with torch.no_grad():
        generated_skips = model.decoder.skip_generator(z)
    
    print(f"\nSkipGenerator output (for generation):")
    for i, skip in enumerate(generated_skips):
        print(f"  Generated skip {i}: {skip.shape}")
    
    # Test reconstruction (using encoder skips)
    with torch.no_grad():
        reconstructed, mu_rec, logvar_rec = model(x, skip_prob=0.0)
    
    print(f"\nReconstruction output: {reconstructed.shape}")
    
    # Test generation (using generated skips)
    with torch.no_grad():
        generated = model.decoder(z, use_generated_skips=True)
    
    print(f"Generation output: {generated.shape}")
    
    # Verify skip shapes match (decoder reverses encoder skips)
    print("\n" + "=" * 80)
    print("VERIFICATION RESULTS")
    print("=" * 80)
    
    # Decoder reverses encoder skips for upsampling
    reversed_encoder_skips = skips[::-1]
    
    match = True
    if len(generated_skips) != len(reversed_encoder_skips):
        print(f"[X] SKIP COUNT MISMATCH: generator={len(generated_skips)}, encoder_reversed={len(reversed_encoder_skips)}")
        match = False
    else:
        print(f"[OK] Skip count matches: {len(generated_skips)}")
        print(f"[INFO] Decoder uses encoder skips in REVERSED order")
        
        for i, (enc_skip_rev, gen_skip) in enumerate(zip(reversed_encoder_skips, generated_skips)):
            if enc_skip_rev.shape == gen_skip.shape:
                print(f"[OK] Skip {i} matches: {gen_skip.shape} (encoder skip {len(skips)-1-i} reversed)")
            else:
                print(f"[X] Skip {i} MISMATCH: encoder_rev={enc_skip_rev.shape}, generator={gen_skip.shape}")
                match = False
    
    # Check if decoder blocks can use these skips
    print("\nDecoder block skip requirements:")
    for i, block in enumerate(model.decoder.up_blocks):
        expected_channels = model.decoder.skip_generator.skip_specs[i][0]
        expected_length = model.decoder.skip_generator.skip_specs[i][1]
        print(f"  Block {i}: expects ({block.skip_c} channels, length calculated)")
        print(f"    SkipGenerator provides: ({expected_channels} channels, {expected_length} length)")
    
    if match:
        print("\n" + "=" * 80)
        print("[OK] ALL SKIP CONNECTIONS ARE CORRECT!")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("[X] SKIP CONNECTION MISMATCH DETECTED!")
        print("=" * 80)
        return False

if __name__ == "__main__":
    success = verify_model()
    sys.exit(0 if success else 1)

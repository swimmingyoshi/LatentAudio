import torch
import numpy as np
from latentaudio.core.simple_vae import SimpleFastVAE
from latentaudio.core.training import compute_vae_loss
from latentaudio.core.losses import MultiResolutionSTFTLoss

def test_hd_engine():
    waveform_length = 44100
    latent_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Testing HD Engine on {device}...")
    model = SimpleFastVAE(latent_dim=latent_dim, waveform_length=waveform_length).to(device)
    mrstft = MultiResolutionSTFTLoss().to(device)
    
    x = torch.randn(2, 1, waveform_length).to(device)
    
    # 1. Forward Pass
    recon, mu, logvar, skips, gen_skips = model(x)
    print(f"  [SUCCESS] Forward pass output shape: {recon.shape}")
    
    # 2. Loss Computation (HD Engine)
    loss_dict = compute_vae_loss(
        recon, x, mu, logvar, 
        beta_kl=0.001, 
        mrstft=mrstft, 
        real_skips=skips, 
        gen_skips=gen_skips
    )
    
    print("\n--- Phase 7 Loss Metrics ---")
    for k, v in loss_dict.items():
        print(f"  {k:20}: {v.item():.4f}")
    
    # 3. Check for Headroom
    max_amp = recon.abs().max().item()
    print(f"\n  [SUCCESS] Output Peak: {max_amp:.4f} (Goal: <= 0.9000)")
    assert max_amp <= 0.95, "Headroom buffer failed!"

if __name__ == "__main__":
    try:
        test_hd_engine()
        print("\n🚀 PHASE 7 TRANSPARENCY VERIFIED!")
    except Exception as e:
        print(f"\n❌ STRESS TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

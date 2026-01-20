import torch
from latentaudio.core.simple_vae import SimpleFastVAE
from latentaudio.core.vae import UnconditionalVAE

def test_models():
    waveform_length = 44100
    latent_dim = 128
    
    print("Testing SimpleFastVAE...")
    simple_model = SimpleFastVAE(latent_dim=latent_dim, waveform_length=waveform_length)
    x = torch.randn(2, 1, waveform_length)
    recon, mu, logvar, skips, gen_skips = simple_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Recon shape: {recon.shape}")
    assert recon.shape == (2, waveform_length)
    print("  ✓ SimpleFastVAE shape correct")
    
    print("\nTesting UnconditionalVAE...")
    complex_model = UnconditionalVAE(latent_dim=latent_dim, waveform_length=waveform_length)
    recon, mu, logvar, skips, gen_skips = complex_model(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Recon shape: {recon.shape}")
    assert recon.shape == (2, waveform_length)
    print("  ✓ UnconditionalVAE shape correct")

if __name__ == "__main__":
    try:
        test_models()
        print("\n🚀 ALL MODELS VERIFIED!")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

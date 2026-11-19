"""Tests for the LatentAudio v2 model (run: pytest tests/test_v2.py)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentaudio.v2.dsp import InverseSTFT, num_frames  # noqa: E402
from latentaudio.v2.losses import MultiResolutionSTFTLoss, kl_divergence, spectral_metrics  # noqa: E402
from latentaudio.v2.model import LatentAudioV2, V2Config  # noqa: E402

SMALL = V2Config(length=8192, latent_dim=16, n_fft=512, hop=128, dec_dim=64, dec_layers=2,
                 enc_n_fft=512, enc_hop=128, enc_dim=32, enc_layers=1, enc_time_steps=8)


@pytest.mark.parametrize("n_fft,hop,length", [(1024, 256, 44100), (512, 128, 8192), (2048, 512, 22050)])
def test_inverse_stft_matches_torch_istft(n_fft, hop, length):
    torch.manual_seed(0)
    x = torch.randn(2, length)
    window = torch.hann_window(n_fft)
    spec = torch.stft(x, n_fft, hop, n_fft, window, center=True, return_complex=True)
    assert spec.shape[-1] == num_frames(length, hop)
    ours = InverseSTFT(n_fft, hop, length)(torch.cat([spec.real, spec.imag], dim=1))
    ref = torch.istft(spec, n_fft, hop, n_fft, window, center=True, length=length)
    assert torch.allclose(ours, ref, atol=1e-4), (ours - ref).abs().max()
    assert torch.allclose(ours, x, atol=1e-4)  # perfect reconstruction


def test_model_shapes_and_api():
    model = LatentAudioV2(SMALL).eval()
    x = torch.randn(3, SMALL.length) * 0.1
    recon, mu, logvar, skips, gen_skips = model(x)
    assert recon.shape == x.shape and mu.shape == (3, 16) and logvar.shape == (3, 16)
    assert skips is None and gen_skips is None
    assert model.decode(model.sample_latent(2)).shape == (2, SMALL.length)
    assert torch.isfinite(recon).all()
    # starts quiet, not a wall of noise
    assert recon.abs().max() < 0.5


def test_one_training_step_reduces_loss():
    torch.manual_seed(0)
    model = LatentAudioV2(SMALL)
    loss_fn = MultiResolutionSTFTLoss(((512, 128), (256, 64)))
    x = torch.sin(torch.linspace(0, 400, SMALL.length)).repeat(4, 1) * 0.5
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(25):
        y, mu, logvar, _, _ = model(x)
        out = loss_fn(y, x)
        loss = out["sc"] + out["log_mag"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.8


def test_kl_free_bits():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)
    kl, raw = kl_divergence(mu, logvar, free_bits=0.1)
    assert raw.item() == pytest.approx(0.0)
    assert kl.item() == pytest.approx(0.8)  # every dim floored at 0.1


def test_metrics_zero_for_identical_and_level_sensitive():
    x = torch.randn(2, 8192) * 0.1
    m = spectral_metrics(x, x, 44100)
    assert max(m["lsd_db"]) < 1e-3 and max(m["mel_db"]) < 1e-3
    m2 = spectral_metrics(x * 0.5, x, 44100)
    assert m2["loudness_db"][0] == pytest.approx(6.02, abs=0.05)


def test_onnx_export_matches_torch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnx = pytest.importorskip("onnx")
    from latentaudio.v2.export import export_decoder_onnx

    model = LatentAudioV2(SMALL).eval()
    path = export_decoder_onnx(model, tmp_path / "dec.onnx")
    proto = onnx.load(str(path))
    assert proto.ir_version <= 9
    assert max(o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")) <= 19

    z = torch.randn(2, SMALL.latent_dim)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run(["audio_output"], {"latent_input": z.numpy()})[0]
    ref = model.decode(z).detach().numpy()
    assert out.shape == (2, SMALL.length)
    np.testing.assert_allclose(out, ref, atol=1e-4)


@pytest.mark.parametrize("path,expected", [
    ("KSHMR_Live_Instruments/KSHMR_Violin_01_100_C#.wav", "strings"),     # file name beats folder
    ("INSTRUMENT SOUNDS/BELLS/BELL - BUCKET.wav", "bell"),                # "strum" is not in "instrument"
    ("Pokemon_Cries/051 - Dugtrio/PLAY_PV_0051.wav", "other"),            # "gtr" is not in "Dugtrio"
    ("Acoustic_Drums/KSHMR_Acoustic_Tom_01_Low_B.wav", "perc"),           # "acoustic" is not guitar
    ("Guitar/KSHMR_Acoustic_Guitar_Chords_01_G#m.wav", "guitar"),
    ("Kicks/KSHMR Big Kick 01 (D).wav", "kick"),
    ("Hats/KSHMR_Closed_Hat_03.wav", "hihat"),
])
def test_category_classifier(path, expected):
    from latentaudio.v2.categories import classify

    assert classify(path) == expected


def test_loop_detection():
    from latentaudio.v2.categories import is_loop

    assert is_loop(Path("Guitar Loops 112BPM/x.wav"))
    assert is_loop(Path("Fills/KSHMR_Acoustic_Fill_01_90.wav"))
    assert is_loop(Path("Phrases/Lead [122bpm Cm].wav"))
    assert is_loop(Path("Kicks/KSHMR Big Kick 01 (D).wav")) is None

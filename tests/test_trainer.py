"""Trainer checkpoints/resume and the dataset cache."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latentaudio.v2.data import load_folder_cached  # noqa: E402
from latentaudio.v2.model import V2Config  # noqa: E402
from latentaudio.v2.trainer import TrainConfig, Trainer, load_v2  # noqa: E402

CFG = V2Config(length=8192, latent_dim=8, n_fft=512, hop=128, dec_dim=32, dec_layers=1,
               enc_n_fft=512, enc_hop=128, enc_dim=16, enc_layers=1, enc_time_steps=8)
X = (np.random.default_rng(0).normal(size=(6, 8192)) * 0.1).astype(np.float32)


def trainer(out, epochs, **kw):
    return Trainer(CFG, TrainConfig(epochs=epochs, batch_size=3, eval_every=2, **kw), out, device="cpu",
                   log=lambda s: None)


def test_resume_matches_uninterrupted_run(tmp_path):
    full = trainer(tmp_path / "full", 4)
    full.fit(X)
    trainer(tmp_path / "cut", 4).fit(X, should_stop=iter([False, True]).__next__)
    cut = trainer(tmp_path / "cut", 4)
    cut.fit(X, resume=True)
    assert [r["epoch"] for r in cut.history] == [1, 2, 3, 4]
    for a, b in zip(full.ema.parameters(), cut.ema.parameters()):
        assert torch.allclose(a, b, atol=1e-5)


def test_resume_rejects_different_settings(tmp_path):
    trainer(tmp_path, 2).fit(X)
    with pytest.raises(ValueError, match="Cannot resume"):
        trainer(tmp_path, 4, lr=1e-3).fit(X, resume=True)


def test_early_stop_writes_this_runs_best_and_real_epoch(tmp_path):
    trainer(tmp_path, 2).fit(X)
    old = (tmp_path / "best.pt").read_bytes()
    trainer(tmp_path, 10).fit(X, should_stop=lambda: True)
    assert (tmp_path / "best.prev.pt").read_bytes() == old
    for name in ("best.pt", "last.pt"):
        assert torch.load(tmp_path / name, weights_only=True)["epoch"] == 1
    load_v2(tmp_path / "best.pt")


def test_cache_rebuilt_when_data_or_length_changes(tmp_path):
    sf = pytest.importorskip("soundfile")
    for d in ("a", "b"):
        (tmp_path / d).mkdir()
        sf.write(tmp_path / d / f"{d}.wav", np.sin(np.arange(9000) * 0.1) * 0.5, 44100)
    cache = tmp_path / "c.npz"
    assert load_folder_cached(tmp_path / "a", 4096, cache)[1] == ["a"]
    assert load_folder_cached(tmp_path / "a", 4096, cache)[1] == ["a"]
    assert load_folder_cached(tmp_path / "b", 4096, cache)[1] == ["b"]
    assert load_folder_cached(tmp_path / "b", 8192, cache)[0].shape == (1, 8192)

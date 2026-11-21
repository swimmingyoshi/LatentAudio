# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export a v2 model to a LatentAudioVST model folder (ONNX decoder + sound map)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from .model import LatentAudioV2

ONNX_OPSET = 17  # ONNX Runtime 1.16 (the VST's) supports opset <= 19 / IR <= 9


class _DecoderForExport(nn.Module):
    def __init__(self, model: LatentAudioV2):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, latent_input: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_input)


def export_decoder_onnx(model: LatentAudioV2, path: Path) -> Path:
    import onnx

    model = model.cpu().eval()
    dummy = torch.zeros(1, model.cfg.decoder_input_dim)
    torch.onnx.export(
        _DecoderForExport(model), (dummy,), str(path),
        input_names=["latent_input"], output_names=["audio_output"],
        dynamic_axes={"latent_input": {0: "batch_size"}, "audio_output": {0: "batch_size"}},
        opset_version=ONNX_OPSET, dynamo=False, do_constant_folding=True,
    )
    proto = onnx.load(str(path))
    if proto.ir_version > 9:  # keep it loadable by ORT 1.16
        proto.ir_version = 8
        onnx.save(proto, str(path))
    onnx.checker.check_model(proto)
    return path


def project_2d(latents: np.ndarray, method: str = "auto", seed: int = 42) -> np.ndarray:
    """2D layout for the sound map, normalised to [-1, 1]."""
    coords = None
    if method in ("auto", "umap") and len(latents) >= 10:
        try:
            import umap

            coords = umap.UMAP(n_neighbors=min(15, len(latents) - 1), min_dist=0.1,
                               random_state=seed).fit_transform(latents)
        except Exception:
            coords = None
    if coords is None:
        centered = latents - latents.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    lo, hi = coords.min(0), coords.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return 2 * (coords - lo) / rng - 1


@torch.no_grad()
def encode_dataset(model: LatentAudioV2, audio: np.ndarray, device: str = "cpu") -> np.ndarray:
    model = model.to(device).eval()
    out = []
    for i in range(0, len(audio), 32):
        mu, _ = model.encoder(torch.from_numpy(audio[i:i + 32]).to(device))
        out.append(mu.float().cpu().numpy())
    return np.concatenate(out)


def export_vst_folder(model: LatentAudioV2, audio: np.ndarray, labels: List[str], out_dir: Path,
                      device: str = "cpu", method: str = "auto", midi: Optional[np.ndarray] = None,
                      sky_map: bool = True) -> dict:
    """Writes latent_decoder.onnx + sound_map.json into out_dir.

    Pitched models: each point's latent gets its normalised pitch appended (so the
    plugin's latent size equals the ONNX input size) and the map describes where
    the pitch lives, so the plugin can replace it with a played MIDI note.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    latents = encode_dataset(model, audio, device)
    coords = project_2d(latents, method)  # layout by timbre only
    cfg = model.cfg
    if cfg.pitch_conditioned:
        if midi is None:
            raise ValueError("pitched model export needs midi per sound")
        latents = np.concatenate([latents, ((midi - cfg.pitch_offset) / cfg.pitch_scale)[:, None]], axis=1)
    points = [{"label": labels[i], "x": float(coords[i, 0]), "y": float(coords[i, 1]),
               "latent": [round(float(v), 6) for v in latents[i]]} for i in range(len(latents))]
    if cfg.pitch_conditioned:
        for i, p in enumerate(points):
            p["midi"] = round(float(midi[i]), 2)
    sound_map = {"latent_dim": cfg.decoder_input_dim, "sample_rate": cfg.sample_rate,
                 "model": "latentaudio-v2", "points": points}
    if cfg.pitch_conditioned:
        # latent[index] = (midi - offset) / scale
        sound_map["pitch"] = {"index": cfg.latent_dim, "offset": cfg.pitch_offset, "scale": cfg.pitch_scale}
    if sky_map:
        # Even layout, smooth field, star colours/sizes and the nebula (see mapfield.py).
        from .mapfield import apply_sky_map, build_sky_map

        decoder = model.decoder.to(device).eval()

        @torch.no_grad()
        def decode(z, batch=32):
            return np.concatenate([decoder(torch.as_tensor(z[i:i + batch], dtype=torch.float32, device=device))
                                   .float().cpu().numpy() for i in range(0, len(z), batch)])

        extras = build_sky_map(coords, latents, decode, cfg.sample_rate, timbre_dims=cfg.latent_dim)
        sound_map = apply_sky_map(sound_map, extras)
    (out_dir / "sound_map.json").write_text(json.dumps(sound_map))
    onnx_path = export_decoder_onnx(model, out_dir / "latent_decoder.onnx")
    return {"onnx_mb": onnx_path.stat().st_size / 1e6, "points": len(points)}

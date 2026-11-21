# SPDX-License-Identifier: AGPL-3.0-or-later
"""Smooth latent field for the sound map.

The plugin used to blend the 5 nearest sounds by inverse distance. That has two
problems: the result jumps whenever the set of 5 nearest changes (creases), and
averaging latents shrinks them (the space between sounds sounds dull).

Here the map is a continuous field  f(x, y) -> latent  sampled on a grid that the
plugin reads bicubically:

  f = m + r
  m  wide Gaussian (Nadaraya-Watson) blend: fills every spot, including gaps
  r  Gaussian kernel ridge fit of the residual z_i - m(x_i): makes f pass
     (almost) through every real sound, decays to 0 away from the data

Optionally the field is rescaled to the local typical latent norm, which undoes
the shrinkage of averaging.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------------- layout
def layout_2d(latents: np.ndarray, method: str = "auto", seed: int = 42, spread: float = 0.5) -> np.ndarray:
    """2D layout normalised to [-1, 1].

    UMAP keeps similar sounds together. `spread` (0..1) then evens out the density
    by blending each axis with its rank (dense clumps open up, order is kept), so
    the map uses the whole square instead of a few tight islands.
    """
    coords = None
    n = len(latents)
    if method in ("auto", "umap") and n >= 10:
        try:
            import umap

            coords = umap.UMAP(n_neighbors=min(30, n - 1), min_dist=0.5, spread=1.0,
                               random_state=seed).fit_transform(latents)
        except Exception:
            coords = None
    if coords is None:
        centered = latents - latents.mean(0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ vt[:2].T
    coords = _normalise(coords)
    if spread > 0 and n > 3:
        ranks = np.argsort(np.argsort(coords, axis=0), axis=0) / (n - 1) * 2 - 1
        coords = _normalise((1 - spread) * coords + spread * ranks)
    return coords


def _normalise(c: np.ndarray) -> np.ndarray:
    lo, hi = c.min(0), c.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return 2 * (c - lo) / rng - 1


# ----------------------------------------------------------------------------- old blend
def idw_knn(coords: np.ndarray, latents: np.ndarray, q: np.ndarray, k: int = 5) -> np.ndarray:
    """The plugin's previous interpolation (inverse distance squared over the k nearest)."""
    d2 = ((q[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1)[:, :k]
    dk = np.take_along_axis(d2, idx, 1)
    out = np.empty((len(q), latents.shape[1]), np.float64)
    for i in range(len(q)):
        if dk[i, 0] < 1e-10:
            out[i] = latents[idx[i, 0]]
            continue
        w = 1.0 / dk[i]
        out[i] = (w[:, None] * latents[idx[i]]).sum(0) / w.sum()
    return out


# ----------------------------------------------------------------------------- field
@dataclass
class FieldConfig:
    res: int = 128            # grid cells per axis
    bandwidth: float = 1.0    # residual kernel width, x median distance to the 5th neighbour
    base_width: float = 4.0   # baseline blend width, x the residual width
    ridge: float = 0.05       # > 0: slight smoothing instead of exact interpolation
    norm_fix: bool = False    # rescale to the local typical latent norm
    norm_dims: Optional[int] = None  # leading dims that are timbre (others, e.g. pitch, untouched)
    mode: str = "shepard"     # "shepard": inverse-distance blend over all sounds with a smooth
                              #   taper (no nearest-k cutoff, so no creases); "krr": kernel fit


class LatentField:
    def __init__(self, coords: np.ndarray, latents: np.ndarray, cfg: FieldConfig = FieldConfig()):
        self.cfg = cfg
        self.x = np.asarray(coords, np.float64)
        self.z = np.asarray(latents, np.float64)
        n = len(self.x)
        d2 = _sqdist(self.x, self.x)
        np.fill_diagonal(d2, np.inf)
        k = min(5, n - 1)
        kth = np.sqrt(np.sort(d2, axis=1)[:, k - 1]) if n > 1 else np.array([0.5])
        self.h = max(1e-3, float(np.median(kth)) * cfg.bandwidth)
        self.H = self.h * cfg.base_width
        np.fill_diagonal(d2, 0.0)

        self.nd = cfg.norm_dims or self.z.shape[1]
        self.norms = np.linalg.norm(self.z[:, :self.nd], axis=1)
        if cfg.mode == "shepard":
            return
        base = self._baseline(d2)
        K = np.exp(-d2 / (2 * self.h ** 2))
        K[np.diag_indices_from(K)] += cfg.ridge
        self.alpha = np.linalg.solve(K, self.z - base)

    def _shepard(self, d2: np.ndarray) -> np.ndarray:
        # w = taper / (d^2 + eps^2): like the old 1/d^2 blend near the sounds, but every
        # sound's weight fades smoothly with distance instead of dropping out at k = 5.
        eps2 = (0.15 * self.h) ** 2
        logw = -d2 / (2 * self.H ** 2) - np.log(d2 + eps2)
        logw -= logw.max(1, keepdims=True)
        w = np.exp(logw)
        return (w @ self.z) / w.sum(1, keepdims=True)

    def _baseline(self, d2: np.ndarray, values: Optional[np.ndarray] = None) -> np.ndarray:
        values = self.z if values is None else values
        logw = -d2 / (2 * self.H ** 2)
        logw -= logw.max(1, keepdims=True)  # stable far from the data
        w = np.exp(logw)
        return (w @ values) / w.sum(1, keepdims=True)

    def __call__(self, q: np.ndarray, chunk: int = 4096) -> np.ndarray:
        q = np.asarray(q, np.float64)
        out = []
        for i in range(0, len(q), chunk):
            d2 = _sqdist(q[i:i + chunk], self.x)
            if self.cfg.mode == "shepard":
                f = self._shepard(d2)
            else:
                f = self._baseline(d2) + np.exp(-d2 / (2 * self.h ** 2)) @ self.alpha
            if self.cfg.norm_fix:
                target = self._baseline(d2, self.norms[:, None])[:, 0]
                cur = np.linalg.norm(f[:, :self.nd], axis=1)
                f[:, :self.nd] *= (target / np.maximum(cur, 1e-6))[:, None]
            out.append(f)
        return np.concatenate(out)

    def grid(self, bounds=None) -> dict:
        """Field sampled on a res x res grid over the map bounds (row = y, column = x)."""
        r = self.cfg.res
        if bounds is None:
            (x0, y0), (x1, y1) = self.x.min(0), self.x.max(0)
        else:
            x0, x1, y0, y1 = bounds
        gx, gy = np.meshgrid(np.linspace(x0, x1, r), np.linspace(y0, y1, r))
        vals = self(np.stack([gx.ravel(), gy.ravel()], 1)).reshape(r, r, -1)
        return {"res": r, "min_x": float(x0), "max_x": float(x1), "min_y": float(y0), "max_y": float(y1),
                "values": vals.astype(np.float32)}


def _sqdist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T)


# ----------------------------------------------------------------------------- sampling
def _catmull_rom(t: np.ndarray):
    t2, t3 = t * t, t * t * t
    return np.stack([-0.5 * t3 + t2 - 0.5 * t,
                     1.5 * t3 - 2.5 * t2 + 1.0,
                     -1.5 * t3 + 2.0 * t2 + 0.5 * t,
                     0.5 * t3 - 0.5 * t2], -1)


def sample_grid(g: dict, q: np.ndarray) -> np.ndarray:
    """Bicubic (Catmull-Rom) sampling, clamped at the edges - exactly what the plugin does."""
    v = g["values"]
    r = g["res"]
    u = (q[:, 0] - g["min_x"]) / max(1e-9, g["max_x"] - g["min_x"]) * (r - 1)
    w = (q[:, 1] - g["min_y"]) / max(1e-9, g["max_y"] - g["min_y"]) * (r - 1)
    u, w = np.clip(u, 0, r - 1), np.clip(w, 0, r - 1)
    iu, iw = np.floor(u).astype(int), np.floor(w).astype(int)
    wu, ww = _catmull_rom(u - iu), _catmull_rom(w - iw)
    out = np.zeros((len(q), v.shape[2]))
    for a in range(4):
        rows = np.clip(iw - 1 + a, 0, r - 1)
        for b in range(4):
            cols = np.clip(iu - 1 + b, 0, r - 1)
            out += (ww[:, a] * wu[:, b])[:, None] * v[rows, cols]
    return out


def grid_to_json(g: dict) -> dict:
    """Map-file block: float16 little-endian, row-major [y][x][dim], base64."""
    data = g["values"].astype("<f2").tobytes()
    return {"res": g["res"], "dim": int(g["values"].shape[2]),
            "min_x": g["min_x"], "max_x": g["max_x"], "min_y": g["min_y"], "max_y": g["max_y"],
            "format": "f16", "data": base64.b64encode(data).decode("ascii")}


def grid_from_json(block: dict) -> dict:
    raw = np.frombuffer(base64.b64decode(block["data"]), "<f2").astype(np.float32)
    r, d = block["res"], block["dim"]
    return {"res": r, "min_x": block["min_x"], "max_x": block["max_x"],
            "min_y": block["min_y"], "max_y": block["max_y"], "values": raw.reshape(r, r, d)}


# ----------------------------------------------------------------------------- sky map
def describe(audio: np.ndarray, sr: int) -> np.ndarray:
    """Per-sound raw descriptors: [log spectral centroid, level dB, ring length s]."""
    audio = np.asarray(audio, np.float64)
    n = audio.shape[1]
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(audio * win, axis=1)) ** 2
    f = np.fft.rfftfreq(n, 1 / sr)
    cent = (spec * f).sum(1) / np.maximum(spec.sum(1), 1e-12)
    level = 10 * np.log10(np.maximum((audio ** 2).mean(1), 1e-12))
    hop = 512
    frames = n // hop
    env = (audio[:, :frames * hop].reshape(len(audio), frames, hop) ** 2).mean(2)
    env_db = 10 * np.log10(np.maximum(env, 1e-12))
    above = env_db > env_db.max(1, keepdims=True) - 40
    last = frames - 1 - np.argmax(above[:, ::-1], axis=1)
    length = (last + 1) * hop / sr
    return np.stack([np.log2(np.maximum(cent, 20.0)), level, length], 1)


def rank_scale(ref: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Position of each value in the reference distribution, 0..1 (per column)."""
    out = np.empty_like(values, dtype=np.float64)
    for c in range(values.shape[1]):
        s = np.sort(ref[:, c])
        out[:, c] = np.searchsorted(s, values[:, c]) / max(1, len(s))
    return np.clip(out, 0, 1)


def build_sky_map(coords: np.ndarray, latents: np.ndarray, decode, sample_rate: int,
                  timbre_dims: Optional[int] = None, cfg: FieldConfig = FieldConfig(),
                  sky_res: int = 48, layout: bool = True, spread: float = 0.5) -> dict:
    """Everything the plugin's sky map needs, as sound-map entries.

    latents: decoder inputs per sound (pitch included for pitched models, at the end).
    decode:  f(latents [B, D]) -> audio [B, T] (numpy).
    Returns {"coords", "field", "sky", "brightness", "length"}.
    """
    latents = np.asarray(latents, np.float64)
    nd = timbre_dims or latents.shape[1]
    if layout:
        coords = layout_2d(latents[:, :nd], spread=spread)
    field = LatentField(coords, latents, FieldConfig(**{**cfg.__dict__, "norm_dims": nd}))
    grid = field.grid((-1.0, 1.0, -1.0, 1.0) if layout else None)

    # Descriptors of the sounds themselves (decoded, so they match the sky's scale).
    point_desc = describe(decode(latents), sample_rate)
    gx, gy = np.meshgrid(np.linspace(grid["min_x"], grid["max_x"], sky_res),
                         np.linspace(grid["min_y"], grid["max_y"], sky_res))
    q = np.stack([gx.ravel(), gy.ravel()], 1)
    sky_desc = describe(decode(sample_grid(grid, q)), sample_rate)
    point_scaled = rank_scale(point_desc, point_desc)
    sky_scaled = rank_scale(point_desc, sky_desc).reshape(sky_res, sky_res, 3)
    # Light blur: neighbouring cells are separate decodes, so their descriptors are noisy.
    from scipy.ndimage import gaussian_filter

    sky_scaled = np.stack([gaussian_filter(sky_scaled[..., c], sigma=sky_res / 40.0, mode="nearest")
                           for c in range(3)], -1)
    sky = {"res": sky_res, "min_x": grid["min_x"], "max_x": grid["max_x"],
           "min_y": grid["min_y"], "max_y": grid["max_y"], "values": sky_scaled.astype(np.float32)}
    return {"coords": coords, "field": grid_to_json(grid), "sky": grid_to_json(sky),
            "brightness": point_scaled[:, 0], "level": point_scaled[:, 1], "length": point_scaled[:, 2]}


def apply_sky_map(sound_map: dict, extras: dict) -> dict:
    for i, p in enumerate(sound_map["points"]):
        p["x"], p["y"] = round(float(extras["coords"][i, 0]), 5), round(float(extras["coords"][i, 1]), 5)
        p["brightness"] = round(float(extras["brightness"][i]), 3)
        p["length"] = round(float(extras["length"][i]), 3)
    sound_map["field"] = extras["field"]
    sound_map["sky"] = extras["sky"]
    sound_map["map_version"] = 2
    return sound_map

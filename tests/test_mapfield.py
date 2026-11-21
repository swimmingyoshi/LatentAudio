import base64
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latentaudio.v2.mapfield import (FieldConfig, LatentField, apply_sky_map, build_sky_map,  # noqa: E402
                                     grid_from_json, grid_to_json, idw_knn, layout_2d, sample_grid)


def _data(n=60, d=6, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (n, 2))
    z = np.concatenate([np.sin(3 * xy), np.cos(2 * xy), xy ** 2, ], 1)[:, :d] + 0.05 * rng.normal(size=(n, d))
    return xy, z


def test_field_is_close_to_each_sound():
    xy, z = _data()
    f = LatentField(xy, z, FieldConfig())
    err = np.abs(f(xy) - z).mean()
    assert err < 0.25 * np.abs(z - z.mean(0)).mean()


def test_field_has_no_creases_but_old_blend_does():
    xy, z = _data(n=80)
    t = np.linspace(0, 1, 4001)[:, None]
    line = np.array([-0.9, -0.3]) * (1 - t) + np.array([0.9, 0.4]) * t
    jump = lambda v: np.abs(np.diff(v, axis=0)).max()  # noqa: E731
    f = LatentField(xy, z, FieldConfig())
    g = f.grid((-1, 1, -1, 1))
    smooth = sample_grid(g, line)
    old = idw_knn(xy, z, line)
    # the old 5-nearest blend steps whenever the neighbour set changes
    assert jump(smooth) < 0.5 * jump(old)


def test_grid_sampling_hits_nodes_and_clamps():
    xy, z = _data()
    g = LatentField(xy, z, FieldConfig(res=16)).grid((-1, 1, -1, 1))
    nodes = np.array([[-1.0, -1.0], [1.0, 1.0], [-1 + 2 * 5 / 15, -1 + 2 * 9 / 15]])
    got = sample_grid(g, nodes)
    np.testing.assert_allclose(got[0], g["values"][0, 0], atol=1e-5)
    np.testing.assert_allclose(got[1], g["values"][15, 15], atol=1e-5)
    np.testing.assert_allclose(got[2], g["values"][9, 5], atol=1e-5)
    np.testing.assert_allclose(sample_grid(g, np.array([[5.0, -5.0]]))[0], g["values"][0, 15], atol=1e-5)


def test_json_round_trip_is_standard_base64_f16():
    xy, z = _data()
    g = LatentField(xy, z, FieldConfig(res=8)).grid()
    block = grid_to_json(g)
    raw = base64.b64decode(block["data"], validate=True)
    assert len(raw) == 8 * 8 * z.shape[1] * 2
    back = grid_from_json(block)
    np.testing.assert_allclose(back["values"], g["values"], atol=2e-3, rtol=2e-3)


def test_layout_fills_the_square():
    _, z = _data(n=200, d=6)
    c = layout_2d(z, method="pca", spread=0.5)
    assert c.shape == (200, 2) and np.allclose(c.min(0), -1) and np.allclose(c.max(0), 1)


def test_build_sky_map_with_fake_decoder():
    xy, z = _data(n=40, d=5)
    z = np.concatenate([z, np.full((40, 1), 0.1)], 1)  # a pitch element at the end

    def decode(lat):
        t = np.arange(4410) / 44100
        f0 = 200 + 400 * (np.tanh(lat[:, :1]) + 1)
        return (np.sin(2 * np.pi * f0 * t) * np.exp(-t * (2 + 3 * np.abs(lat[:, 1:2])))).astype(np.float32)

    extras = build_sky_map(xy, z, decode, 44100, timbre_dims=5, cfg=FieldConfig(res=16), sky_res=8)
    assert extras["field"]["dim"] == 6 and extras["sky"]["dim"] == 3
    assert 0 <= extras["brightness"].min() and extras["brightness"].max() <= 1
    sm = apply_sky_map({"points": [{"latent": list(r)} for r in z]}, extras)
    assert sm["map_version"] == 2 and "brightness" in sm["points"][0] and -1 <= sm["points"][0]["x"] <= 1

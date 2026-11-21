"""Measure how good a sound map is between and around the sounds.

  python scripts/eval_map.py --model D:/DSTORAGE/AudioModels/StringsModel_v2p

Compares the plugin's old interpolation (5-nearest inverse-distance blend on the
current layout) with the smooth latent field (mapfield.py), optionally on a new
layout. Everything is decoded with the exported ONNX decoder.

* held-out   : leave real sounds out of the map, predict them from their position,
               decode, compare with the real sound's own decode (mel dB, lower = better).
               This asks "does an empty spot sound like what belongs there?"
* smoothness : walk straight lines across the map; mel change per step. Reports the
               median step and the worst jumps (p99 / median: lower = fewer creases).
* dullness   : midpoints between neighbouring sounds vs. the sounds themselves:
               level and brightness (spectral centroid) loss in dB / % (0 = none).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latentaudio.v2.losses import spectral_metrics  # noqa: E402
from latentaudio.v2.mapfield import (FieldConfig, LatentField, idw_knn, layout_2d,  # noqa: E402
                                     sample_grid)


class Decoder:
    def __init__(self, onnx_path: Path):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def __call__(self, z: np.ndarray, batch: int = 24) -> np.ndarray:
        out = [self.s.run(None, {"latent_input": z[i:i + batch].astype(np.float32)})[0]
               for i in range(0, len(z), batch)]
        return np.concatenate(out)


def mel_db(a: np.ndarray, b: np.ndarray, sr: int) -> np.ndarray:
    return np.asarray(spectral_metrics(torch.from_numpy(a), torch.from_numpy(b), sr)["mel_db"])


def centroid_level(y: np.ndarray, sr: int):
    spec = np.abs(np.fft.rfft(y * np.hanning(y.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(y.shape[1], 1 / sr)
    cent = (spec * f).sum(1) / np.maximum(spec.sum(1), 1e-12)
    level = 10 * np.log10(np.maximum((y ** 2).mean(1), 1e-12))
    return cent, level


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--holdout", type=int, default=120)
    ap.add_argument("--lines", type=int, default=10)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--pairs", type=int, default=80)
    ap.add_argument("--res", type=int, nargs="+", default=[128])
    ap.add_argument("--bandwidth", type=float, nargs="+", default=[1.0])
    ap.add_argument("--spread", type=float, nargs="+", default=[0.5])
    ap.add_argument("--modes", nargs="+", default=["shepard"])
    ap.add_argument("--base-width", type=float, default=4.0)
    ap.add_argument("--norm-fix", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    smap = json.loads((args.model / "sound_map.json").read_text())
    sr = smap.get("sample_rate", 44100)
    pts = smap["points"]
    Z = np.array([p["latent"] for p in pts], np.float64)
    old_xy = np.array([[p["x"], p["y"]] for p in pts], np.float64)
    pitch = smap.get("pitch")
    pidx = pitch["index"] if pitch else None
    timbre = np.delete(Z, pidx, axis=1) if pitch else Z
    nd = timbre.shape[1]
    dec = Decoder(args.model / "latent_decoder.onnx")
    rng = np.random.default_rng(args.seed)
    n = len(Z)
    print(f"{args.model.name}: {n} sounds, latent {Z.shape[1]}{' (pitched)' if pitch else ''}")

    layouts = {"old layout": old_xy}
    for sp in args.spread:
        layouts[f"new layout s{sp}"] = layout_2d(timbre, spread=sp)

    def with_pitch(zt, midi):
        """Timbre latents -> decoder input (fixed pitch for pitched models)."""
        if not pitch:
            return zt
        p = np.full((len(zt), 1), (midi - pitch["offset"]) / pitch["scale"]) if np.isscalar(midi) \
            else ((np.asarray(midi) - pitch["offset"]) / pitch["scale"])[:, None]
        return np.concatenate([zt[:, :pidx], p, zt[:, pidx:]], 1)

    midis = np.array([p.get("midi", 60.0) for p in pts]) if pitch else None

    # methods: name -> (coords, predictor(train_idx) -> f(q))
    def idw_on(xy):
        return lambda tr: (lambda q: idw_knn(xy[tr], timbre[tr], q))

    def field_on(xy, res, bw, mode):
        cfg = FieldConfig(res=res, bandwidth=bw, norm_fix=args.norm_fix, mode=mode, base_width=args.base_width)

        def make(tr):
            f = LatentField(xy[tr], timbre[tr], cfg)
            g = f.grid((-1, 1, -1, 1))
            return lambda q: sample_grid(g, q)
        return make

    methods = {}
    for lname, xy in layouts.items():
        methods[f"IDW-5 {lname}"] = (xy, idw_on(xy))
        for mode in args.modes:
            for res in args.res:
                for bw in args.bandwidth:
                    methods[f"{mode} r{res} bw{bw} {lname}"] = (xy, field_on(xy, res, bw, mode))

    # ---- held-out
    ho = rng.choice(n, size=min(args.holdout, n // 5), replace=False)
    tr = np.setdiff1d(np.arange(n), ho)
    ref = dec(with_pitch(timbre[ho], midis[ho] if pitch else None))
    # smoothness lines (between random sounds) + neighbour midpoints, per layout
    line_ends = rng.choice(n, size=(args.lines, 2))
    t = np.linspace(0, 1, args.steps)[:, None]

    print(f"\n{'method':36s} {'held-out mel':>12s} {'step mel':>9s} {'crease':>6s} "
          f"{'mid level':>9s} {'mid bright':>10s}")
    for name, (xy, make) in methods.items():
        f_ho = make(tr)
        pred = dec(with_pitch(f_ho(xy[ho]), midis[ho] if pitch else None))
        ho_mel = mel_db(pred, ref, sr).mean()

        f_all = make(np.arange(n))
        steps, crease = [], []
        for a, b in line_ends:
            q = xy[a] * (1 - t) + xy[b] * t
            y = dec(with_pitch(f_all(q), 60.0))
            st = mel_db(y[1:], y[:-1], sr)
            steps.append(st)
            # a crease is a step much bigger than its neighbours (steep-but-smooth is not)
            pad = np.pad(st, 2, mode="edge")
            nb = np.median(np.stack([pad[0:-4], pad[1:-3], pad[3:-1], pad[4:]]), 0)
            crease.append(st / np.maximum(nb, 0.05))
        steps = np.concatenate(steps)
        med = float(np.median(steps))
        jumps = float(np.percentile(np.concatenate(crease), 99))

        # neighbouring pairs (in this layout): midpoint vs ends
        d2 = ((xy[:, None] - xy[None]) ** 2).sum(-1)
        np.fill_diagonal(d2, np.inf)
        a = rng.choice(n, size=args.pairs, replace=False)
        b = np.argsort(d2[a], 1)[:, rng.integers(0, 3)]
        ends = dec(with_pitch(np.concatenate([f_all(xy[a]), f_all(xy[b])]), 60.0))
        mids = dec(with_pitch(f_all((xy[a] + xy[b]) / 2), 60.0))
        ce, le = centroid_level(ends, sr)
        cm, lm = centroid_level(mids, sr)
        lvl = float(np.mean(lm - (le[:args.pairs] + le[args.pairs:]) / 2))
        bright = float(np.mean(cm / ((ce[:args.pairs] + ce[args.pairs:]) / 2) - 1) * 100)
        print(f"{name:36s} {ho_mel:10.2f}dB {med:7.2f}dB {jumps:6.1f} {lvl:+8.2f}dB {bright:+9.1f}%", flush=True)


if __name__ == "__main__":
    main()

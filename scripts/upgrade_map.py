"""Upgrade an exported model folder to the sky map (no retraining).

  python scripts/upgrade_map.py D:/DSTORAGE/AudioModels/StringsModel_v2p [more folders...]

Adds a new even layout, the smooth latent field (no creases between sounds),
per-sound brightness / length (star colour / size) and the nebula grid the
plugin draws. The previous map is kept as sound_map.v1.json.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from latentaudio.v2.mapfield import FieldConfig, apply_sky_map, build_sky_map  # noqa: E402


def onnx_decoder(path: Path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def decode(z, batch=24):
        z = np.asarray(z, np.float32)
        return np.concatenate([sess.run(None, {"latent_input": z[i:i + batch]})[0]
                               for i in range(0, len(z), batch)])
    return decode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", type=Path, nargs="+")
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--bandwidth", type=float, default=1.0)
    ap.add_argument("--spread", type=float, default=0.5)
    ap.add_argument("--keep-layout", action="store_true", help="Keep the current dot positions")
    args = ap.parse_args()

    for folder in args.folders:
        t0 = time.time()
        path = folder / "sound_map.json"
        smap = json.loads(path.read_text())
        backup = folder / "sound_map.v1.json"
        if not backup.exists() and smap.get("map_version", 1) < 2:
            shutil.copy2(path, backup)
        z = np.array([p["latent"] for p in smap["points"]], np.float64)
        coords = np.array([[p["x"], p["y"]] for p in smap["points"]], np.float64)
        pitch = smap.get("pitch")
        nd = pitch["index"] if pitch else z.shape[1]
        if pitch and pitch["index"] != z.shape[1] - 1:
            raise SystemExit(f"{folder}: pitch must be the last latent element")
        extras = build_sky_map(coords, z, onnx_decoder(folder / "latent_decoder.onnx"),
                               smap.get("sample_rate", 44100), timbre_dims=nd,
                               cfg=FieldConfig(res=args.res, bandwidth=args.bandwidth),
                               layout=not args.keep_layout, spread=args.spread)
        path.write_text(json.dumps(apply_sky_map(smap, extras)))
        print(f"{folder.name}: {len(z)} sounds, map {path.stat().st_size / 1e6:.1f} MB, "
              f"{time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()

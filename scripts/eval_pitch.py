"""Measure how in-tune an exported model is.

  python scripts/eval_pitch.py --model D:/.../GuitarModel_v2p --manifest D:/.../guitar/manifest.json

* own pitch:   decode each sound's stored latent; compare the detected pitch with the
               sound's real pitch (from the map for pitched models, else the manifest).
* transposed:  pitched models only - request the same timbre at other notes and check
               the detected pitch follows (what Key Track needs).
"""

import argparse
import json
from pathlib import Path

import numpy as np

SR = 44100


def detect_midi(y):
    import librosa

    seg = y[int(0.03 * SR): int(1.03 * SR)]
    f0, voiced, _ = librosa.pyin(seg, fmin=35, fmax=2100, sr=SR, frame_length=4096, hop_length=1024)
    f0 = f0[voiced & np.isfinite(f0)]
    return 69 + 12 * np.log2(np.median(f0) / 440.0) if f0.size >= 3 else np.nan


def summarize(name, err):
    err = np.asarray([e for e in err if np.isfinite(e)])
    if not len(err):
        print(f"  {name}: no pitched output detected")
        return
    print(f"  {name:22s} n={len(err):3d}  median error {np.median(err):.2f} st   "
          f"within 0.5 st {np.mean(err < 0.5) * 100:3.0f}%   off >1 st {np.mean(err > 1) * 100:3.0f}%")


def main():
    import onnxruntime as ort

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="Dataset manifest (needed for unpitched models)")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--val-labels", type=Path, help="Only evaluate these labels (held-out set)")
    args = ap.parse_args()

    smap = json.loads((args.model / "sound_map.json").read_text())
    pitch = smap.get("pitch")
    points = smap["points"]
    if args.val_labels:
        keep = set(json.loads(args.val_labels.read_text()))
        points = [p for p in points if p["label"] in keep]
    truth = {}
    if pitch:
        truth = {p["label"]: p["midi"] for p in points}
    elif args.manifest:
        for c in json.loads(args.manifest.read_text())["clips"]:
            if "midi" in c:
                truth[c["file"].rsplit(".", 1)[0]] = c["midi"] + c.get("cents", 0) / 100
    points = [p for p in points if p["label"] in truth]
    rng = np.random.default_rng(0)
    points = [points[i] for i in rng.choice(len(points), min(args.n, len(points)), replace=False)]

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    sess = ort.InferenceSession(str(args.model / "latent_decoder.onnx"), opts, providers=["CPUExecutionProvider"])

    def decode(lat):
        return sess.run(None, {"latent_input": np.asarray(lat, np.float32)[None]})[0][0]

    print(f"{args.model.name}: {'pitch-conditioned' if pitch else 'unconditioned'}, {len(points)} sounds")
    summarize("own pitch", [abs(detect_midi(decode(p["latent"])) - truth[p["label"]]) for p in points])

    if pitch:
        idx, off, scale = pitch["index"], pitch["offset"], pitch["scale"]
        for shift in (-7, -3, 4, 7):
            err = []
            for p in points[: max(20, args.n // 2)]:
                target = truth[p["label"]] + shift
                lat = list(p["latent"])
                lat[idx] = (target - off) / scale
                err.append(abs(detect_midi(decode(lat)) - target))
            summarize(f"transposed {shift:+d} st", err)


if __name__ == "__main__":
    main()

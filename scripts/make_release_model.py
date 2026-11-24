"""Copy a VST model folder for sharing, without sample names.

Labels become "<name> 001" (plus the note for pitched models); audio, positions and
the sky map are unchanged. Only latent_decoder.onnx and sound_map.json are written
(never sound_map.v1.json, which still holds the original names).

Example:
  python scripts/make_release_model.py D:/Models/KickModel_v2 release/KickModel_v2 --name Kick
"""

import argparse
import json
import shutil
from pathlib import Path

NOTES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
TOP_KEYS = ("latent_dim", "sample_rate", "model", "points", "pitch", "field", "sky", "map_version")
POINT_KEYS = ("x", "y", "latent", "midi", "brightness", "length")  # label is replaced


def anonymize(sound_map: dict, name: str) -> dict:
    """Keeps only known keys, so nothing unexpected can leak through."""
    out = {k: sound_map[k] for k in TOP_KEYS if k in sound_map}
    width = max(3, len(str(len(sound_map["points"]))))
    points = []
    for i, p in enumerate(sound_map["points"], 1):
        label = f"{name} {i:0{width}d}"
        if "midi" in p:
            m = round(p["midi"])
            label += f" ({NOTES[m % 12]}{m // 12 - 1})"
        points.append({"label": label, **{k: p[k] for k in POINT_KEYS if k in p}})
    out["points"] = points
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--name", required=True, help='Label prefix, e.g. "Kick"')
    args = ap.parse_args()
    if args.dst.exists():
        ap.error(f"{args.dst} already exists")

    original = json.loads((args.src / "sound_map.json").read_text(encoding="utf-8"))
    release = anonymize(original, args.name)
    args.dst.mkdir(parents=True)
    (args.dst / "sound_map.json").write_text(json.dumps(release), encoding="utf-8")
    shutil.copyfile(args.src / "latent_decoder.onnx", args.dst / "latent_decoder.onnx")
    print(f"{len(release['points'])} sounds -> {args.dst}")


if __name__ == "__main__":
    main()

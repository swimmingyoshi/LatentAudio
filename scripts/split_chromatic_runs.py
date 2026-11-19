"""Split chromatic-run recordings (e.g. University of Iowa MIS) into single labelled notes.

File names carry the range, e.g. "Guitar.mf.sulA.C4E4.aif" = C4..E4 played one note at a
time. A file is only used when the number of detected notes equals the range size AND
each note's detected pitch is within a semitone of its label, so nothing is mislabelled.

  python scripts/split_chromatic_runs.py --input .../iowa_guitar/raw --out .../iowa_guitar/notes
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

SR = 44100
RANGE = re.compile(r"\.([A-G]b?\d)([A-G]b?\d)?\.(?:mono|stereo|aif)", re.I)


def note_range(name):
    import librosa

    m = RANGE.search(name)
    if not m:
        return None
    lo = librosa.note_to_midi(m.group(1))
    hi = librosa.note_to_midi(m.group(2)) if m.group(2) else lo
    return list(range(lo, hi + 1))


def segment(y, expected, min_gap_s=2.0):
    """Note starts: the `expected` strongest onsets at least min_gap_s apart.

    Plucked notes ring into each other, so silence splitting is unreliable; small
    handling noises right after a pluck are weaker than the plucks and get dropped.
    """
    import librosa

    hop = 512
    env = librosa.onset.onset_strength(y=y, sr=SR, hop_length=hop)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=SR, hop_length=hop, backtrack=False)
    if len(frames) < expected:
        return None, None
    chosen = []
    for f in sorted(frames, key=lambda f: -env[f]):
        if all(abs(int(f) - int(c)) * hop / SR >= min_gap_s for c in chosen):
            chosen.append(f)
        if len(chosen) == expected:
            break
    if len(chosen) != expected:
        return None, None
    starts = librosa.onset.onset_backtrack(np.array(sorted(chosen)), env) * hop
    ends = list(starts[1:]) + [len(y)]
    return list(zip(starts, ends)), "onsets"


def main():
    import librosa
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prefix", default="iowa")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report, written = [], 0
    for path in sorted(p for p in args.input.rglob("*") if p.suffix.lower() in (".aif", ".aiff", ".wav")
                       and not p.name.startswith("._") and "__MACOSX" not in p.parts):
        notes = note_range(path.name)
        if not notes:
            report.append({"file": path.name, "status": "no range in name"})
            continue
        y, _ = librosa.load(str(path), sr=SR, mono=True)
        iv, top_db = segment(y, len(notes))
        if iv is None:
            report.append({"file": path.name, "status": f"note count mismatch (expected {len(notes)})"})
            continue

        # Each note runs until the next note starts (keeps its natural decay).
        starts = [a for a, _ in iv] + [len(y)]
        ok, clips = True, []
        for k, midi in enumerate(notes):
            seg = y[starts[k]: starts[k + 1]]
            f0, voiced, _ = librosa.pyin(seg[: SR], fmin=60, fmax=1500, sr=SR, frame_length=4096, hop_length=1024)
            f0 = f0[voiced & np.isfinite(f0)]
            detected = 69 + 12 * np.log2(np.median(f0) / 440.0) if f0.size else None
            if detected is None or abs(detected - midi) > 1.0:
                ok = False
                break
            clips.append((midi, seg))
        if not ok:
            report.append({"file": path.name, "status": f"pitch check failed at note {k} ({librosa.midi_to_note(midi, unicode=False)})"})
            continue

        parts = path.stem.split(".")  # Guitar.mf.sulA.C4E4
        stem = "_".join(p for p in parts[:3])
        for midi, seg in clips:
            note = librosa.midi_to_note(midi, unicode=False).replace("#", "s")
            sf.write(str(args.out / f"{args.prefix}_{stem}_{note}.wav"), seg.astype(np.float32), SR, subtype="PCM_24")
            written += 1
        report.append({"file": path.name, "status": "ok", "notes": len(clips), "top_db": top_db})

    (args.out / "split_report.json").write_text(json.dumps(report, indent=1))
    ok = sum(r["status"] == "ok" for r in report)
    print(f"{written} notes from {ok}/{len(report)} files")
    for r in report:
        if r["status"] != "ok":
            print(f"  skipped {r['file']}: {r['status']}")


if __name__ == "__main__":
    main()

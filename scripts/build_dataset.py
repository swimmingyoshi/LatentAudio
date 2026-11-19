"""Collect one-shots of one instrument category from sample libraries into a training set.

  # what's in my libraries?
  python scripts/build_dataset.py --source B:/MUSIC_SAMPLES --list

  # build a guitar set: 2 s clips, pitch-detected, near-duplicates removed
  python scripts/build_dataset.py --source B:/MUSIC_SAMPLES --category guitar \
      --seconds 2.0 --pitch --out D:/DSTORAGE/Datasets/guitar

For each file: skip loops/phrases and long files, load mono 44.1 kHz, trim leading
silence, crop/pad to --seconds (fading out if cut), peak-normalise to -1 dBFS.
Writes the clips (same folder structure as the source, prefixed by the source name)
plus manifest.json with source path, duration, detected pitch and skip reasons.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentaudio.v2.categories import AUDIO_EXTENSIONS, GROUPS, classify, is_loop  # noqa: E402

SR = 44100


def scan(sources):
    for root in sources:
        for path in root.rglob("*"):
            if path.suffix.lower() in AUDIO_EXTENSIONS and path.is_file():
                yield root, path


def duration_of(path):
    import soundfile as sf

    try:
        return sf.info(str(path)).duration
    except Exception:
        try:
            import librosa

            return librosa.get_duration(path=str(path))
        except Exception:
            return None


def list_inventory(sources, max_seconds):
    stats = collections.defaultdict(lambda: {"files": 0, "oneshots": 0, "packs": collections.Counter()})
    for root, path in scan(sources):
        rel = path.relative_to(root)
        s = stats[classify(str(rel))]
        s["files"] += 1
        s["packs"][f"{root.name}/{rel.parts[0]}" if len(rel.parts) > 1 else root.name] += 1
        if not is_loop(rel):
            d = duration_of(path)
            if d is not None and 0.05 <= d <= max_seconds:
                s["oneshots"] += 1
    print(f"{'category':10s} {'files':>6s} {'one-shots':>9s}  main sources")
    for cat, s in sorted(stats.items(), key=lambda kv: -kv[1]["oneshots"]):
        packs = ", ".join(f"{k} ({v})" for k, v in s["packs"].most_common(3))
        print(f"{cat:10s} {s['files']:6d} {s['oneshots']:9d}  {packs}")


def process(path, seconds, peak=0.89, trim_db=50.0):
    import librosa

    y, _ = librosa.load(str(path), sr=SR, mono=True, duration=seconds + 3.0)
    if y.size == 0 or not np.isfinite(y).all() or np.abs(y).max() < 1e-4:
        return None, "silent/unreadable"
    level = np.abs(y).max()
    start = np.nonzero(np.abs(y) > level * 10 ** (-trim_db / 20))[0][0]
    y = y[max(0, start - 32):]                     # keep a few samples before the onset
    n = int(seconds * SR)
    if y.size >= n:
        y = y[:n].copy()
        fade = int(0.05 * SR)
        y[-fade:] *= np.linspace(1.0, 0.0, fade) ** 2
    else:
        y = np.pad(y, (0, n - y.size))
    return (y * peak / np.abs(y).max()).astype(np.float32), None


def detect_pitch(y):
    """Median f0 over the first second after the attack. Returns (hz, voiced_fraction)."""
    import librosa

    seg = y[int(0.03 * SR): int(1.03 * SR)]
    f0, voiced, _ = librosa.pyin(seg, fmin=35.0, fmax=2100.0, sr=SR, frame_length=4096, hop_length=1024)
    v = voiced & np.isfinite(f0)
    if v.mean() < 0.3:
        return None, float(v.mean())
    return float(np.median(f0[v])), float(v.mean())


def fingerprint(y):
    import librosa

    m = librosa.feature.melspectrogram(y=y[:SR], sr=SR, n_fft=2048, hop_length=1024, n_mels=48)
    f = np.log(m + 1e-5).ravel()
    f = f - f.mean()
    return f / (np.linalg.norm(f) + 1e-9)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, action="append", required=True, help="Sample library root (repeatable)")
    ap.add_argument("--list", action="store_true", help="Only print an inventory by category")
    ap.add_argument("--category", help="Category name (see --list) or group: " + ", ".join(GROUPS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--seconds", type=float, default=1.0, help="Clip length")
    ap.add_argument("--max-source-seconds", type=float, default=6.0, help="Longer files count as non-one-shots")
    ap.add_argument("--exclude", action="append", default=[], help="Skip paths containing this text (repeatable)")
    ap.add_argument("--pitch", action="store_true", help="Detect pitch (slower) and store it in the manifest")
    ap.add_argument("--dedupe", type=float, default=0.995, help="Drop clips this similar to an earlier one (0 = off)")
    args = ap.parse_args()

    if args.list:
        list_inventory(args.source, args.max_source_seconds)
        return
    if not args.category or not args.out:
        ap.error("--category and --out are required unless --list")

    from tqdm import tqdm
    import soundfile as sf

    wanted = set(GROUPS.get(args.category, [args.category]))
    skipped = collections.Counter()
    candidates = []
    for root, path in scan(args.source):
        rel = path.relative_to(root)
        if classify(str(rel)) not in wanted:
            continue
        if any(x.lower() in str(rel).lower() for x in args.exclude):
            skipped["excluded"] += 1
            continue
        reason = is_loop(rel)
        if reason:
            skipped[reason] += 1
            continue
        d = duration_of(path)
        if d is None or d < 0.05:
            skipped["unreadable/too short"] += 1
            continue
        if d > args.max_source_seconds:
            skipped[f"longer than {args.max_source_seconds:g}s"] += 1
            continue
        candidates.append((root, path, d))

    args.out.mkdir(parents=True, exist_ok=True)
    manifest, prints = [], []
    for root, path, d in tqdm(candidates, desc=f"Processing {args.category}"):
        y, err = process(path, args.seconds)
        if err:
            skipped[err] += 1
            continue
        if args.dedupe > 0:
            fp = fingerprint(y)
            if prints and float(np.max(np.stack(prints) @ fp)) > args.dedupe:
                skipped["near-duplicate"] += 1
                continue
            prints.append(fp)
        rel = Path(root.name) / path.relative_to(root)
        out_path = (args.out / rel).with_suffix(".wav")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), y, SR, subtype="PCM_24")
        entry = {"file": str(rel.with_suffix(".wav")).replace("\\", "/"), "source": str(path),
                 "source_seconds": round(d, 3)}
        if args.pitch:
            hz, voiced = detect_pitch(y)
            entry["voiced"] = round(voiced, 2)
            if hz:
                midi = 69 + 12 * np.log2(hz / 440.0)
                entry.update(pitch_hz=round(hz, 2), midi=int(round(midi)), cents=int(round((midi - round(midi)) * 100)))
        manifest.append(entry)

    (args.out / "manifest.json").write_text(json.dumps({
        "category": args.category, "seconds": args.seconds, "sample_rate": SR,
        "sources": [str(s) for s in args.source], "skipped": dict(skipped), "clips": manifest}, indent=1))

    print(f"\n{len(manifest)} clips written to {args.out}")
    print("skipped:", dict(skipped))
    packs = collections.Counter(Path(m["file"]).parts[1] if len(Path(m["file"]).parts) > 2 else "?" for m in manifest)
    print("by pack:", dict(packs.most_common(8)))
    if args.pitch:
        import librosa

        pitched = [m for m in manifest if "midi" in m]
        print(f"pitched: {len(pitched)}/{len(manifest)}")
        if pitched:
            notes = collections.Counter(librosa.midi_to_note(m["midi"]) for m in pitched)
            print("notes:", dict(sorted(notes.items(), key=lambda kv: librosa.note_to_midi(kv[0]))))


if __name__ == "__main__":
    main()

"""Compare exported VST model folders on the same sounds, using identical metrics.

Each model folder holds latent_decoder.onnx + sound_map.json (as the VST loads it).
For every sound in --data, the model's stored latent for that sound is decoded and
compared with the original audio. Also measures CPU decode latency the way the VST
runs it (ONNX Runtime, one intra-op thread) and renders spectrogram comparisons.

  python scripts/compare_models.py --data <one-shot folder> \
      --model v1=D:/DSTORAGE/AudioModels/KickModel --model v2=D:/.../KickModel_v2 \
      --out runs/compare_kicks [--val-labels runs/kicks_v2/val_labels.json]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentaudio.v2.data import find_audio_files, load_one_shot  # noqa: E402


def decode_all(session, latents, batch=1):
    out = []
    for i in range(0, len(latents), batch):
        out.append(session.run(["audio_output"], {"latent_input": latents[i:i + batch].astype(np.float32)})[0])
    return np.concatenate(out)


def match_order(decoded, originals, sample_rate):
    """The v1 map stores points as 'Sample_i'. Check that point i really is file i by
    testing whether the diagonal of a small distance matrix is the minimum."""
    import torch
    from latentaudio.v2.losses import spectral_metrics

    k = min(24, len(decoded))
    hits = 0
    for i in range(k):
        y = torch.from_numpy(originals[i:i + 1]).repeat(k, 1)
        d = spectral_metrics(torch.from_numpy(decoded[:k]), y, sample_rate)["lsd_db"]
        hits += int(np.argmin(d) == i)
    return hits / k


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", action="append", required=True, help="name=folder (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--val-labels", type=Path, help="JSON list of labels held out from v2 training")
    ap.add_argument("--examples", type=int, default=4, help="spectrogram comparisons to render")
    args = ap.parse_args()

    import onnxruntime as ort
    import torch
    from latentaudio.v2.losses import spectral_metrics

    args.out.mkdir(parents=True, exist_ok=True)
    files = find_audio_files(args.data)
    labels = [str(f.relative_to(args.data).with_suffix("")).replace("\\", "/") for f in files]
    val_labels = set(json.loads(args.val_labels.read_text())) if args.val_labels else set()

    report = {}
    decoded_by_model = {}
    for spec in args.model:
        name, folder = spec.split("=", 1)
        folder = Path(folder)
        sound_map = json.loads((folder / "sound_map.json").read_text())
        points = sound_map["points"]
        sr = sound_map.get("sample_rate", 44100)
        latents = np.array([p["latent"] for p in points], dtype=np.float32)

        # Map points to files: by label if the map stores file labels, else by index.
        by_label = {p["label"]: i for i, p in enumerate(points)}
        if all(l in by_label for l in labels):
            pairs = [(by_label[l], j) for j, l in enumerate(labels)]
            how = "label"
        else:
            # v1 maps store 'Sample_i' in the order v1's loader read files:
            # rglob per extension (directory-walk order), not sorted order.
            v1_order = []
            for pattern in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
                v1_order.extend(args.data.rglob(pattern))
            position = {f: j for j, f in enumerate(files)}
            pairs = [(i, position[f]) for i, f in enumerate(v1_order[: len(points)]) if f in position]
            how = "index (v1 rglob order)"

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        t0 = time.perf_counter()
        sess = ort.InferenceSession(str(folder / "latent_decoder.onnx"), opts, providers=["CPUExecutionProvider"])
        load_s = time.perf_counter() - t0

        decode_all(sess, latents[:2])  # warm-up
        times = []
        for i in range(10):
            t0 = time.perf_counter()
            decode_all(sess, latents[i % len(latents):i % len(latents) + 1])
            times.append((time.perf_counter() - t0) * 1000)

        point_idx = [p for p, _ in pairs]
        decoded = decode_all(sess, latents[point_idx])
        length = decoded.shape[1]
        originals = np.stack([load_one_shot(files[j], sr, length) for _, j in pairs])
        order_score = match_order(decoded, originals, sr) if how.startswith("index") else 1.0

        # Compare timbre/shape independent of overall level: match each output's RMS to
        # its original (v1 was trained at a different global level). Raw level error is
        # reported separately as loudness_db (meaningful for v2 only).
        out_level_db = 20 * np.log10(np.sqrt((decoded ** 2).mean(1)) + 1e-9)
        raw_loud = spectral_metrics(torch.from_numpy(decoded), torch.from_numpy(originals), sr)["loudness_db"]
        gain = np.sqrt((originals ** 2).mean(1, keepdims=True)) / np.maximum(np.sqrt((decoded ** 2).mean(1, keepdims=True)), 1e-8)
        matched = (decoded * gain).astype(np.float32)
        m = spectral_metrics(torch.from_numpy(matched), torch.from_numpy(originals), sr)
        m["loudness_db"] = raw_loud
        is_val = np.array([labels[j] in val_labels for _, j in pairs])

        def summary(mask):
            if not mask.any():
                return None
            return {k: round(float(np.mean(np.array(v)[mask])), 3) for k, v in m.items()}

        # Morph smoothness: decode 5 evenly spaced latents between nearest-neighbour
        # pairs; a smooth space changes evenly (ratio of the largest step to the mean
        # step near 1), a bumpy one jumps or glitches in between.
        rng = np.random.default_rng(0)
        idx = rng.choice(len(latents), min(40, len(latents)), replace=False)
        ratios, mids = [], []
        for i in idx:
            d = np.linalg.norm(latents - latents[i], axis=1)
            d[i] = np.inf
            j = int(np.argmin(d))
            path = np.stack([latents[i] * (1 - t) + latents[j] * t for t in np.linspace(0, 1, 5)])
            audio_path = torch.from_numpy(decode_all(sess, path, batch=1))
            steps = [spectral_metrics(audio_path[k + 1:k + 2], audio_path[k:k + 1], sr)["mel_db"][0] for k in range(4)]
            ratios.append(max(steps) / max(np.mean(steps), 1e-6))
            mids.append(float(np.abs(audio_path[2]).max()))
        smooth = {"morph_step_ratio": round(float(np.mean(ratios)), 2),
                  "midpoint_peak_vs_ends": round(float(np.mean(mids)), 2)}

        report[name] = {
            "folder": str(folder), "matched_by": how, "order_check": round(order_score, 2),
            "sounds": len(pairs), "onnx_mb": round((folder / "latent_decoder.onnx").stat().st_size / 1e6, 1),
            "load_s": round(load_s, 2), "decode_ms_median": round(float(np.median(times)), 1), **smooth,
            "output_level_dbfs_mean": round(float(out_level_db.mean()), 1),
            "all": summary(np.ones(len(pairs), bool)), "train_only": summary(~is_val) if val_labels else None,
            "held_out": summary(is_val) if val_labels else None,
        }
        decoded_by_model[name] = (matched, originals, [labels[j] for _, j in pairs], sr)
        print(json.dumps({name: report[name]}, indent=1))

    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    render_examples(decoded_by_model, args.out, args.examples, val_labels)
    print(f"\nWrote {args.out / 'report.json'} and spectrogram PNGs")


def render_examples(decoded_by_model, out_dir, n, val_labels):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import librosa

    names = list(decoded_by_model)
    _, originals, labels, sr = decoded_by_model[names[0]]
    rng = np.random.default_rng(1)
    held = [i for i, l in enumerate(labels) if l in val_labels]
    picks = (held[: n // 2] if held else []) + list(rng.choice(len(labels), n - min(len(held), n // 2), replace=False))
    for i in picks:
        fig, axes = plt.subplots(1, 1 + len(names), figsize=(4 * (1 + len(names)), 3.2), facecolor="black")
        series = [("original", originals[i])] + [(nm, decoded_by_model[nm][0][i]) for nm in names]
        for ax, (title, y) in zip(axes, series):
            # Mel scale so sub-bass (808 fundamentals ~40 Hz) and its harmonics are visible.
            S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=128,
                                                                   n_mels=160, fmin=20, fmax=sr / 2), ref=1.0, top_db=90)
            ax.imshow(S, origin="lower", aspect="auto", cmap="magma", vmin=-70, vmax=20,
                      extent=[0, len(y) / sr, 0, S.shape[0]])
            ticks = [40, 100, 300, 1000, 3000, 10000]
            mel_pts = librosa.hz_to_mel(np.array(ticks))
            lo, hi = librosa.hz_to_mel(20), librosa.hz_to_mel(sr / 2)
            ax.set_yticks((mel_pts - lo) / (hi - lo) * S.shape[0])
            ax.set_yticklabels([f"{t//1000}k" if t >= 1000 else str(t) for t in ticks])
            ax.set_title(title, color="white", fontsize=10)
            ax.set_xlim(0, 1.0)
            ax.tick_params(colors="gray", labelsize=7)
            ax.set_xlabel("s", color="gray", fontsize=7)
        axes[0].set_ylabel("Hz (mel scale)", color="gray", fontsize=7)
        tag = " (held out)" if labels[i] in val_labels else ""
        fig.suptitle(labels[i] + tag, color="white", fontsize=9)
        plt.tight_layout()
        fig.savefig(out_dir / f"spec_{i:04d}.png", dpi=80, facecolor="black")
        plt.close(fig)


if __name__ == "__main__":
    main()

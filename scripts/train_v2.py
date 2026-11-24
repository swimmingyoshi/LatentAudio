"""Train a LatentAudio v2 model on a folder of one-shots, optionally export it for the VST.

Examples:
  python scripts/train_v2.py --data D:/DSTORAGE/Datasets/kicks --out runs/kicks_v2
  python scripts/train_v2.py --data ... --out ... --preset transient --epochs 800 \
      --export D:/DSTORAGE/AudioModels/KickModel_v2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from latentaudio.v2.data import load_folder_cached, split_indices  # noqa: E402
from latentaudio.v2.model import V2Config  # noqa: E402
from latentaudio.v2.trainer import TrainConfig, Trainer, load_v2  # noqa: E402

PRESETS = {
    # n_fft/hop: time resolution of the decoder; dim/layers: capacity & speed
    "base": dict(n_fft=1024, hop=256, dec_dim=384, dec_layers=8),
    "fast": dict(n_fft=1024, hop=256, dec_dim=256, dec_layers=6),
    "transient": dict(n_fft=512, hop=128, dec_dim=256, dec_layers=8),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Folder of audio one-shots (searched recursively)")
    ap.add_argument("--out", type=Path, required=True, help="Run directory (checkpoints, history, cache)")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="base")
    ap.add_argument("--latent-dim", type=int, default=128)
    ap.add_argument("--seconds", type=float, default=1.0, help="Sound length (use ~2 for sustained instruments)")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--beta", type=float, default=2e-3, help="KL weight (higher = smoother map, less detail)")
    ap.add_argument("--w-env", type=float, default=1.0, help="Energy-envelope loss weight (keeps long tails)")
    ap.add_argument("--pitch", action="store_true",
                    help="Pitch-conditioned model (melodic instruments); needs the dataset's manifest.json from build_dataset --pitch")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--resume", action="store_true", help="Continue the run in --out (same settings; --epochs may be raised)")
    ap.add_argument("--export", type=Path, help="Write a VST model folder here when done (uses ALL sounds)")
    args = ap.parse_args()

    if args.resume and not (args.out / "state.pt").exists():
        ap.error(f"--resume: no state.pt in {args.out}")

    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / "dataset_cache.npz"
    audio, labels = load_folder_cached(args.data, int(args.seconds * 44100), cache)

    midi = None
    if args.pitch:
        manifest = json.loads((args.data / "manifest.json").read_text())
        by_label = {c["file"].rsplit(".", 1)[0]: c["midi"] + c.get("cents", 0) / 100.0
                    for c in manifest["clips"] if "midi" in c}
        keep = [i for i, l in enumerate(labels) if l in by_label]
        print(f"Pitch-conditioned: {len(keep)}/{len(labels)} sounds have a detected pitch")
        audio, labels = audio[keep], [labels[i] for i in keep]
        midi = np.array([by_label[l] for l in labels], dtype=np.float32)

    train_idx, val_idx = split_indices(len(audio), args.val_fraction)
    (args.out / "val_labels.json").write_text(json.dumps([labels[i] for i in val_idx], indent=1))
    mcfg = V2Config(latent_dim=args.latent_dim, length=audio.shape[1], pitch_conditioned=args.pitch, **PRESETS[args.preset])
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, beta_kl=args.beta, w_env=args.w_env,
                       val_fraction=args.val_fraction, eval_every=args.eval_every)
    (args.out / "config.json").write_text(json.dumps({"model": mcfg.to_dict(), "train": tcfg.to_dict(),
                                                      "data": str(args.data), "preset": args.preset}, indent=1))

    trainer = Trainer(mcfg, tcfg, args.out)
    result = trainer.fit(audio[train_idx], audio[val_idx] if len(val_idx) else None,
                         pitch_train=None if midi is None else midi[train_idx],
                         pitch_val=None if midi is None or not len(val_idx) else midi[val_idx],
                         resume=args.resume)
    print(f"Done in {result['minutes']:.1f} min, best val recon {result['best_val_recon']:.4f}")

    if args.export:
        from latentaudio.v2.export import export_vst_folder

        model = load_v2(args.out / "best.pt")
        info = export_vst_folder(model, audio, labels, args.export, midi=midi)
        print(f"Exported VST folder to {args.export}: {info}")


if __name__ == "__main__":
    main()

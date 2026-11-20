# SPDX-License-Identifier: AGPL-3.0-or-later
"""Training loop for LatentAudio v2.

Design choices (vs the v1 loop):
  * the whole dataset lives on the GPU; batches are index gathers (no DataLoader
    worker processes, which on Windows each re-import torch and copy the data);
  * bf16 autocast (Ampere+) with float32 STFT losses - no GradScaler needed;
  * spectral losses on every step; scalars are only synced to the CPU at log time;
  * EMA of the weights is what gets evaluated, exported and saved as "best";
  * a held-out validation split so "better" is measured, not assumed.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from .losses import EnvelopeLoss, MelLoss, MultiResolutionSTFTLoss, kl_divergence, spectral_metrics
from .model import LatentAudioV2, V2Config


@dataclass
class TrainConfig:
    epochs: int = 600
    batch_size: int = 16
    lr: float = 4e-4
    min_lr: float = 2e-5
    warmup_steps: int = 300
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    beta_kl: float = 2e-3
    kl_warmup_steps: int = 2000
    free_bits: float = 0.02
    w_sc: float = 1.0
    w_log_mag: float = 1.0
    w_mel: float = 1.0
    w_env: float = 1.0
    ema_decay: float = 0.999
    val_fraction: float = 0.1
    eval_every: int = 20          # epochs
    seed: int = 0
    use_bf16: bool = True
    gain_augment_db: float = 3.0  # random +-dB gain per example
    pitch_shift_semitones: float = 2.0  # pitched models: random resample +-N st per batch
    pitch_shift_prob: float = 0.5

    def to_dict(self) -> dict:
        return asdict(self)


class Trainer:
    def __init__(self, model_cfg: V2Config, train_cfg: TrainConfig, out_dir: Path,
                 device: Optional[str] = None, log: Callable[[str], None] = print,
                 model: Optional[LatentAudioV2] = None):
        self.mcfg, self.tcfg = model_cfg, train_cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.log = log
        torch.manual_seed(train_cfg.seed)

        self.model = (model or LatentAudioV2(model_cfg)).to(self.device)
        self.ema = copy.deepcopy(self.model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

        self.mrstft = MultiResolutionSTFTLoss().to(self.device)
        self.mel = MelLoss(model_cfg.sample_rate).to(self.device)
        self.env = EnvelopeLoss().to(self.device)
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    def _lr_at(self, step: int, total: int) -> float:
        c = self.tcfg
        if step < c.warmup_steps:
            return c.lr * (step + 1) / c.warmup_steps
        progress = (step - c.warmup_steps) / max(1, total - c.warmup_steps)
        return c.min_lr + 0.5 * (c.lr - c.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))

    def _losses(self, model: LatentAudioV2, x: torch.Tensor, beta: float,
                midi: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        c = self.tcfg
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=c.use_bf16 and self.device.type == "cuda"):
            mu, logvar = model.encoder(x)
            z = model.reparameterize(mu.float(), logvar.float())
            y_hat = model.decoder(model.decoder_input(z, midi))
        stft = self.mrstft(y_hat, x)
        mel = self.mel(y_hat, x)
        env = self.env(y_hat, x) if c.w_env > 0 else y_hat.new_zeros(())
        kl, kl_raw = kl_divergence(mu.float(), logvar.float(), c.free_bits)
        recon = c.w_sc * stft["sc"] + c.w_log_mag * stft["log_mag"] + c.w_mel * mel + c.w_env * env
        return {"loss": recon + beta * kl, "recon": recon, "sc": stft["sc"], "log_mag": stft["log_mag"],
                "mel": mel, "env": env, "kl": kl_raw}

    @torch.no_grad()
    def _update_ema(self, decay: float) -> None:
        if not hasattr(self, "_ema_pairs"):
            pairs = [(e, p) for e, p in zip(self.ema.parameters(), self.model.parameters()) if p.requires_grad]
            self._ema_pairs = ([e for e, _ in pairs], [p for _, p in pairs])
        torch._foreach_lerp_(self._ema_pairs[0], self._ema_pairs[1], 1.0 - decay)

    @staticmethod
    def _pitch_shift(x: torch.Tensor, semitones: float) -> torch.Tensor:
        """Resample a batch so it plays `semitones` higher (and correspondingly shorter)."""
        ratio = 2.0 ** (semitones / 12.0)
        n = x.shape[1]
        pos = torch.arange(n, device=x.device, dtype=torch.float32) * ratio
        i0 = pos.floor().long().clamp(max=n - 1)
        i1 = (i0 + 1).clamp(max=n - 1)
        frac = (pos - pos.floor()).unsqueeze(0)
        y = x[:, i0] * (1 - frac) + x[:, i1] * frac
        return y * (pos < n - 1).float().unsqueeze(0)

    @torch.no_grad()
    def evaluate(self, data: torch.Tensor, model: Optional[LatentAudioV2] = None,
                 midi: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Reconstruction through the mean latent (what the VST plays)."""
        model = model or self.ema
        model.eval()
        metrics: Dict[str, List[float]] = {}
        losses = []
        for i in range(0, len(data), 32):
            x = data[i:i + 32]
            mu, _ = model.encoder(x)
            y_hat = model.decoder(model.decoder_input(mu, None if midi is None else midi[i:i + 32]))
            for k, v in spectral_metrics(y_hat, x, self.mcfg.sample_rate).items():
                metrics.setdefault(k, []).extend(v)
            stft = self.mrstft(y_hat, x)
            losses.append(float(stft["sc"] + stft["log_mag"] + self.mel(y_hat, x) + self.env(y_hat, x)) * len(x))
        out = {k: float(np.mean(v)) for k, v in metrics.items()}
        out["recon"] = sum(losses) / len(data)
        return out

    # ------------------------------------------------------------------
    def fit(self, train: np.ndarray, val: Optional[np.ndarray] = None,
            on_epoch: Optional[Callable[[int, Dict], None]] = None,
            should_stop: Optional[Callable[[], bool]] = None,
            pitch_train: Optional[np.ndarray] = None, pitch_val: Optional[np.ndarray] = None) -> Dict:
        """pitch_* : MIDI pitch per sound (float), required for pitch-conditioned models."""
        c = self.tcfg
        X = torch.from_numpy(train).to(self.device)
        V = torch.from_numpy(val).to(self.device) if val is not None and len(val) else None
        pitched = self.mcfg.pitch_conditioned
        if pitched and pitch_train is None:
            raise ValueError("pitch-conditioned model needs pitch_train")
        PX = torch.tensor(pitch_train, dtype=torch.float32, device=self.device) if pitched else None
        PV = (torch.tensor(pitch_val, dtype=torch.float32, device=self.device)
              if pitched and V is not None else None)
        n = len(X)
        bs = min(c.batch_size, n)
        steps_per_epoch = max(1, n // bs)
        total_steps = c.epochs * steps_per_epoch

        decay, no_decay = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (no_decay if p.ndim < 2 or name.endswith(".pos") else decay).append(p)
        opt = torch.optim.AdamW([{"params": decay, "weight_decay": c.weight_decay},
                                 {"params": no_decay, "weight_decay": 0.0}], lr=c.lr, betas=(0.8, 0.99))

        self.log(f"v2 model: {self.model.num_parameters() / 1e6:.2f}M params "
                 f"(decoder {self.model.num_parameters(self.model.decoder) / 1e6:.2f}M), "
                 f"{n} train / {0 if V is None else len(V)} val, {steps_per_epoch} steps/epoch, device={self.device}")

        best = math.inf
        step = 0
        t0 = time.time()
        gen = torch.Generator(device=self.device)
        gen.manual_seed(c.seed)
        cpu_rng = np.random.default_rng(c.seed)  # host-side draws: no GPU sync
        for epoch in range(1, c.epochs + 1):
            self.model.train()
            perm = torch.randperm(n, device=self.device, generator=gen)
            acc: Dict[str, torch.Tensor] = {}
            for b in range(steps_per_epoch):
                idx = perm[b * bs:(b + 1) * bs]
                x = X[idx]
                midi = PX[idx] if pitched else None
                if pitched and c.pitch_shift_semitones > 0 and cpu_rng.random() < c.pitch_shift_prob:
                    st = float(cpu_rng.uniform(-1, 1)) * c.pitch_shift_semitones
                    x = self._pitch_shift(x, st)
                    midi = midi + st
                if c.gain_augment_db > 0:
                    gain_db = (torch.rand(len(x), 1, device=self.device, generator=gen) * 2 - 1) * c.gain_augment_db
                    x = x * torch.pow(10.0, gain_db / 20)

                lr = self._lr_at(step, total_steps)
                for g in opt.param_groups:
                    g["lr"] = lr
                beta = c.beta_kl * min(1.0, step / max(1, c.kl_warmup_steps))

                losses = self._losses(self.model, x, beta, midi)
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)
                opt.step()
                # EMA warm-up: average over the steps seen so far until the full decay applies.
                self._update_ema(min(c.ema_decay, (1 + step) / (10 + step)))
                step += 1
                for k, v in losses.items():
                    acc[k] = acc.get(k, 0.0) + v.detach()

            record = {"epoch": epoch, "step": step, "lr": lr, "beta": beta,
                      **{k: float(v) / steps_per_epoch for k, v in acc.items()},  # one sync per epoch
                      "elapsed_s": time.time() - t0}
            if not math.isfinite(record["loss"]):
                raise FloatingPointError(f"Loss became non-finite at epoch {epoch}")

            if epoch % c.eval_every == 0 or epoch == c.epochs:
                ev = self.evaluate(V if V is not None else X[: min(n, 64)], midi=PV if V is not None else (PX[: min(n, 64)] if pitched else None))
                record.update({f"val_{k}": v for k, v in ev.items()})
                if ev["recon"] < best:
                    best = ev["recon"]
                    self.save(self.out_dir / "best.pt", extra={"epoch": epoch, "val": ev})
                self.log(f"epoch {epoch:4d}  loss {record['loss']:.4f}  recon {record['recon']:.4f}  "
                         f"kl {record['kl']:.1f}  | val lsd {ev['lsd_db']:.2f} dB  mel {ev['mel_db']:.2f} dB  "
                         f"loud {ev['loudness_db']:.2f} dB  | {record['elapsed_s'] / 60:.1f} min")
            self.history.append(record)
            if on_epoch:
                on_epoch(epoch, record)
            if should_stop and should_stop():
                self.log(f"Stopped at epoch {epoch}")
                break

        self.save(self.out_dir / "last.pt", extra={"epoch": c.epochs})
        (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=1))
        return {"best_val_recon": best, "minutes": (time.time() - t0) / 60}

    def save(self, path: Path, extra: Optional[Dict] = None) -> None:
        torch.save({"model_type": "v2", "model_config": self.mcfg.to_dict(), "train_config": self.tcfg.to_dict(),
                    "model_state": self.ema.state_dict(), **(extra or {})}, str(path))


def load_v2(path: Path, device: str = "cpu") -> LatentAudioV2:
    ckpt = torch.load(str(path), map_location=device, weights_only=False)
    model = LatentAudioV2(V2Config.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    return model.to(device).eval()

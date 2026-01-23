# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LatentAudio - Direct Neural Audio Generation and Exploration
# Copyright (C) 2024 LatentAudio Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# generator.py
"""Optimized audio generator with faster training loop."""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import os
from scipy import signal
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

from ..types import (
    GeneratorConfig,
    TrainingConfig,
    GenerationConfig,
    TrainingMetrics,
    LatentVector,
    AudioArray,
    LatentPreset,
)
from .vae import UnconditionalVAE
from .simple_vae import SimpleFastVAE
from .training import TrainingLogger, create_optimizer_and_scheduler, compute_vae_loss
from ..explorer.interpolator import LatentInterpolator
from ..explorer.walker import LatentWalker
from ..explorer.preset import PresetManager
from ..explorer.attributes import AttributeExplorer
from ..explorer.projector import LatentProjector
from ..config import (
    AUDIO_LIBS_AVAILABLE,
    LOG_AUDIO_INTERVAL,
    LOW_PASS_ORDER,
    DEFAULT_POSTPROCESS_LOW_PASS,
)

from ..logging import log_training_start, log_model_save, log_model_load
from loguru import logger
import librosa
import soundfile as sf


def check_tensor_health(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """Check if tensor contains NaN or Inf values."""
    if torch.isnan(tensor).any():
        logger.error(f"{name} contains NaN!")
        return False
    if torch.isinf(tensor).any():
        logger.error(f"{name} contains Inf!")
        return False
    return True


def plot_waveform_image(audio: np.ndarray, sr: int, title: str) -> np.ndarray:
    """Create waveform plot image for TensorBoard logging."""
    if audio.ndim > 1:
        audio = audio.squeeze()

    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    fig, ax = plt.subplots(figsize=(10, 3), facecolor="black")
    ax.set_facecolor("black")
    ax.plot(time_axis, audio, color="#00FF00", linewidth=1.5, alpha=0.8)
    ax.set_title(title, color="white", fontsize=12, pad=10)
    ax.set_xlabel("Time (s)", color="white", fontsize=10)
    ax.set_ylabel("Amplitude", color="white", fontsize=10)
    ax.set_xlim(0, duration)
    ax.set_ylim(-1.1, 1.1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, alpha=0.2, color="white")
    plt.tight_layout()

    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    plt.close(fig)

    return image


class AdvancedAudioGenerator:
    """Optimized audio generator with faster training."""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        if config is None:
            config = GeneratorConfig()

        self.config = config
        self.waveform_length = int(config.sample_rate * config.duration)

        self.model: Optional[nn.Module] = None
        self.device = torch.device(
            config.device if config.device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.training_logger: Optional[TrainingLogger] = None
        self._cancel_training = False

        # Explorer components
        self.interpolator = LatentInterpolator()
        self.walker = LatentWalker(latent_dim=config.latent_dim)
        self.preset_manager = PresetManager()
        self.attribute_explorer = AttributeExplorer(latent_dim=config.latent_dim)
        self.projector = LatentProjector(latent_dim=config.latent_dim)

        # Cache for Sound Map
        self.cached_latents: Optional[np.ndarray] = None
        self.cached_labels: Optional[List[str]] = None

        # Training state

        self._total_epochs_trained = 0
        self._best_loss = float("inf")
        self._training_history = {}
        self._optimizer_state = None
        self._scheduler_state = None

        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError("Audio libraries required")

        logger.info(
            f"AdvancedAudioGenerator initialized: {config.sample_rate}Hz, {config.duration}s, device={self.device}"
        )

    def build_model(
        self,
        latent_dim: Optional[int] = None,
        skip_dropout_prob: float = 0.5,
        use_simple: bool = True,
    ) -> nn.Module:
        """Build VAE model with extra conservative initialization."""
        if latent_dim is None:
            latent_dim = self.config.latent_dim

        if use_simple:
            logger.info("Building SimpleFastVAE (optimized for speed)...")
            self.model = SimpleFastVAE(
                latent_dim=latent_dim, waveform_length=self.waveform_length
            ).to(self.device)
        else:
            logger.info("Building UnconditionalVAE (slower, better quality)...")
            self.model = UnconditionalVAE(
                latent_dim=latent_dim,
                waveform_length=self.waveform_length,
                skip_dropout_prob=skip_dropout_prob,
            ).to(self.device)

        logger.info(f"VAE model built: {latent_dim}D latent space")
        return self.model

    def offload_to_ram(self) -> None:
        """Move model weights to CPU RAM to free VRAM."""
        if self.model:
            self.model.to("cpu")
            torch.cuda.empty_cache()
            logger.info("Model offloaded to CPU RAM")

    def load_to_vram(self) -> None:
        """Move model weights back to VRAM (GPU)."""
        if self.model:
            self.model.to(self.device)
            logger.info(f"Model loaded to {self.device}")

    def load_audio_file(self, filepath: Union[str, Path]) -> AudioArray:
        """Load and preprocess audio file with clipping protection."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")
        audio, sr = librosa.load(str(filepath), sr=self.config.sample_rate, mono=True)

        # Soft-clip to prevent harsh edges
        audio = np.tanh(audio * 0.95) / 0.95

        if len(audio) > self.waveform_length:
            audio = audio[: self.waveform_length]
        else:
            audio = np.pad(audio, (0, self.waveform_length - len(audio)))
        return audio

    def load_dataset(self, folder: Union[str, Path], recursive: bool = True) -> List[AudioArray]:
        """Load all audio files from a folder."""
        folder = Path(folder)
        samples = []
        patterns = ["*.wav", "*.mp3", "*.flac", "*.ogg"]
        files = []
        for p in patterns:
            if recursive:
                files.extend(list(folder.rglob(p)))
            else:
                files.extend(list(folder.glob(p)))
        for f in tqdm(files, desc="Loading audio"):
            try:
                samples.append(self.load_audio_file(f))
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        logger.info(f"Loaded {len(samples)} audio samples from {folder}")
        return samples

    def log_mel_spectrogram(self, tag: str, audio: np.ndarray, epoch: int) -> None:
        """Create and log Mel-spectrogram to TensorBoard."""
        if self.training_logger is None:
            return

        try:
            if audio.ndim > 1:
                audio = audio.squeeze()

            # Compute Mel-spectrogram
            S = librosa.feature.melspectrogram(
                y=audio, sr=self.config.sample_rate, n_mels=128, fmax=self.config.sample_rate // 2
            )
            S_dB = librosa.power_to_db(S, ref=np.max)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 4), facecolor="black")
            ax.set_facecolor("black")
            img = librosa.display.specshow(
                S_dB,
                x_axis="time",
                y_axis="mel",
                sr=self.config.sample_rate,
                fmax=self.config.sample_rate // 2,
                ax=ax,
                cmap="magma",
            )
            ax.set_title(f"{tag} (Epoch {epoch})", color="white")
            plt.tight_layout()

            # Convert to image
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            plt.close(fig)

            # Log to TensorBoard
            self.training_logger.log_image(f"Spectrograms/{tag}", image, epoch)

        except Exception as e:
            logger.warning(f"Failed to log Mel-spectrogram: {e}")

    def train(
        self,
        samples: List[AudioArray],
        config: Optional[TrainingConfig] = None,
        log_dir: Optional[Path] = None,
        resume_from: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Training loop - NO WARMUP."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.load_to_vram()
        if config is None:
            config = TrainingConfig()
        if len(samples) == 0:
            raise ValueError("No training samples provided")

        log_training_start(config)
        self.training_logger = TrainingLogger(log_dir)

        # CRITICAL: Log ALL config settings to verify what's being used
        logger.info("=" * 60)
        logger.info("TRAINING CONFIG VERIFICATION")
        logger.info("=" * 60)
        config_dict = config.to_dict()
        for key, value in sorted(config_dict.items()):
            if key != "callback":
                logger.info(f"  {key}: {value}")
        logger.info("=" * 60)

        self.training_logger.log_hyperparameters(config)
        self.training_logger.log_model_summary(self.model)

        # LOG ORIGINAL AUDIO AND SPECTROGRAM ONCE FOR COMPARISON
        fixed_sample_np = np.array(samples[0:1])
        fixed_sample = torch.tensor(fixed_sample_np, dtype=torch.float32).to(self.device)
        self.training_logger.log_audio(
            "Comparison/Original", fixed_sample, self.config.sample_rate, 0
        )
        self.log_mel_spectrogram("Comparison/Original", fixed_sample_np, 0)

        if hasattr(self, "_training_started_callback") and self._training_started_callback:
            self._training_started_callback()

        optimizer, scheduler = create_optimizer_and_scheduler(self.model, config)

        # RESTORE STATE IF RESUMING
        start_epoch = 0
        if resume_from:
            logger.info("Restoring optimizer and scheduler state for resumption...")
            if resume_from.get("optimizer_state"):
                try:
                    optimizer.load_state_dict(resume_from["optimizer_state"])
                except Exception as e:
                    logger.warning(f"Could not restore optimizer state: {e}")

            if resume_from.get("scheduler_state"):
                try:
                    scheduler.load_state_dict(resume_from["scheduler_state"])
                except Exception as e:
                    logger.warning(f"Could not restore scheduler state: {e}")

            start_epoch = resume_from.get("total_epochs_trained", 0)
            self._best_loss = resume_from.get("best_loss", float("inf"))
            logger.info(f"Resuming from absolute epoch {start_epoch + 1}")

        scaler = torch.cuda.amp.GradScaler(
            enabled=config.use_amp and self.device.type == "cuda",
            init_scale=2.0**10,
            growth_factor=1.5,
            backoff_factor=0.5,
            growth_interval=200,
        )

        from .training import calculate_beta, calculate_skip_dropout, calculate_skip_swap
        from .losses import MultiResolutionSTFTLoss

        mrstft = MultiResolutionSTFTLoss(mode=config.stft_mode).to(self.device)

        X = torch.tensor(np.array(samples), dtype=torch.float32)

        # Validate and normalize training data
        logger.info("Validating training data...")
        if torch.isnan(X).any():
            logger.error("NaN detected in training data! Removing corrupted samples...")
            valid_mask = ~torch.isnan(X).any(dim=1)
            X = X[valid_mask]
            logger.warning(
                f"Removed {(~valid_mask).sum()} corrupted samples. {len(X)} samples remaining."
            )

        if torch.isinf(X).any():
            logger.error("Inf detected in training data! Clamping...")
            X = torch.clamp(X, -10.0, 10.0)

        # CRITICAL FIX: Force strict normalization to [-1, 1]
        # This matches the decoder's Tanh output range, preventing distortion
        data_max = X.abs().max()
        if data_max > 0:
            X = X / data_max  # Normalize to exactly [-1, 1]

        logger.info(f"Data range: [{X.min():.3f}, {X.max():.3f}], std: {X.std():.3f}")

        dataset = torch.utils.data.TensorDataset(X)

        if self.device.type == "cuda":
            num_workers = 8
            persistent_workers = True
            prefetch_factor = 8
            pin_memory = True
        else:
            num_workers = min(12, os.cpu_count() - 2 or 1)
            persistent_workers = True
            prefetch_factor = 4
            pin_memory = False

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
            drop_last=True,
        )

        self.model.train()
        fixed_sample = X[0:1].to(self.device)
        best_loss = self._best_loss

        epoch = 0
        avg_loss = 0.0
        avg_kl = 0.0

        logger.info(f"Starting training (AMP: {config.use_amp})")
        logger.info(f"STFT skip interval: {config.stft_skip_interval}")
        logger.info(f"Workers: {num_workers}, Prefetch: {prefetch_factor}")
        logger.info("NO WARMUP - Full beta from epoch 1!")

        # Absolute epoch initialization for safety
        epoch = start_epoch

        try:
            # NO WARMUP - removed all warmstart code
            for session_epoch in range(1, config.epochs + 1):
                if self._cancel_training:
                    break

                # Absolute epoch for schedules and logging
                epoch = start_epoch + session_epoch

                # Use beta immediately from epoch 1
                current_beta = calculate_beta(epoch, config)
                current_skip_dropout = calculate_skip_dropout(epoch)
                current_skip_swap_prob = calculate_skip_swap(epoch)

                epoch_loss, epoch_recon, epoch_kl, epoch_waveform, epoch_spectral, epoch_skip = (
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                )
                n_batches = 0

                optimizer.zero_grad(set_to_none=True)

                total_batches = len(loader)
                progress_bar = tqdm(
                    enumerate(loader),
                    total=total_batches,
                    desc=f"Epoch {epoch:4d}",
                    leave=False,
                    ncols=120,
                )

                for i, (batch_x,) in progress_bar:
                    if self._cancel_training:
                        break
                    batch_x = batch_x.to(self.device, non_blocking=True)

                    compute_stft = (i % config.stft_skip_interval) == 0
                    use_amp_this_batch = config.use_amp and self.device.type == "cuda"

                    # Decide whether to use generated skips for this batch to force training
                    use_gen_skips = torch.rand(1).item() < current_skip_swap_prob

                    # CRITICAL: Skip AMP if previous gradients were unstable or loss is very high
                    if hasattr(self, "_last_grad_overflow") and self._last_grad_overflow:
                        use_amp_this_batch = False
                        logger.debug(
                            f"Disabling AMP for batch {i} due to previous gradient overflow"
                        )

                    with torch.amp.autocast("cuda", enabled=use_amp_this_batch):
                        batch_x_clamped = torch.clamp(batch_x, -5.0, 5.0)

                        # New VAE forward returns skips and gen_skips
                        reconstructed, mu, logvar, real_skips, gen_skips = self.model(
                            batch_x_clamped,
                            skip_prob=current_skip_dropout,
                            use_generated_skips=use_gen_skips,
                        )

                        # Clamp outputs
                        reconstructed = torch.clamp(reconstructed, -5.0, 5.0)
                        mu = torch.clamp(mu, -5.0, 5.0)
                        logvar = torch.clamp(logvar, -5.0, 5.0)

                        # Early NaN detection
                        if (
                            torch.isnan(reconstructed).any()
                            or torch.isnan(mu).any()
                            or torch.isnan(logvar).any()
                        ):
                            logger.error(f"NaN detected in model outputs at batch {i}! Skipping.")
                            optimizer.zero_grad(set_to_none=True)
                            continue

                        # Check for extreme values
                        if (
                            reconstructed.abs().max() > 100
                            or mu.abs().max() > 50
                            or logvar.max() > 10
                        ):
                            logger.warning(
                                f"Extreme values detected! "
                                f"recon_max={reconstructed.abs().max():.1f}, "
                                f"mu_max={mu.abs().max():.1f}, "
                                f"logvar_max={logvar.max():.1f}. "
                                f"Clamping and continuing."
                            )
                            reconstructed = torch.clamp(reconstructed, -10.0, 10.0)
                            mu = torch.clamp(mu, -10.0, 10.0)
                            logvar = torch.clamp(logvar, -10.0, 10.0)

                        z_sample = self.model.reparameterize(mu, logvar)
                        loss_dict = compute_vae_loss(
                            reconstructed,
                            batch_x,
                            mu,
                            logvar,
                            current_beta,
                            config,
                            mrstft,
                            model=self.model,
                            z=z_sample,
                            compute_stft=compute_stft,
                            real_skips=real_skips,
                            gen_skips=gen_skips,
                            epoch=epoch,
                        )
                        total_loss = loss_dict["total_loss"] / config.gradient_accumulation_steps

                    # CRITICAL FIX: Check for gradient overflow AFTER backward but BEFORE accumulation
                    scaler.scale(total_loss).backward()

                    # Check for overflow immediately after backward
                    self._last_grad_overflow = False
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and (
                            torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                        ):
                            logger.error(
                                f"Epoch {epoch}, Batch {i}: NaN/Inf gradient detected in {name}"
                            )
                            self._last_grad_overflow = True
                            break

                    if self._last_grad_overflow:
                        logger.warning(
                            f"Epoch {epoch}, Batch {i}: NaN/Inf gradient (norm=inf). Skipping step."
                        )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        scaler._scale_queue.append(scaler._scale)  # Maintain scaler state
                        continue

                    # Gradient accumulation with NaN protection
                    if (i + 1) % config.gradient_accumulation_steps == 0:
                        if torch.isnan(total_loss) or torch.isinf(total_loss):
                            logger.warning(f"NaN/Inf loss detected! Skipping step.")
                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            continue

                        scaler.unscale_(optimizer)

                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), config.grad_clip
                        )

                        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                            logger.warning(
                                f"Epoch {epoch}, Batch {i}: NaN/Inf gradient (norm={grad_norm}). Skipping."
                            )

                            if epoch <= 10:
                                for name, param in self.model.named_parameters():
                                    if param.grad is not None and (
                                        torch.isnan(param.grad).any()
                                        or torch.isinf(param.grad).any()
                                    ):
                                        logger.error(f"  NaN/Inf in: {name}")

                            optimizer.zero_grad(set_to_none=True)
                            scaler.update()
                            continue

                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        self._last_grad_overflow = False

                    epoch_loss += total_loss.item() * config.gradient_accumulation_steps
                    epoch_recon += loss_dict["recon_loss"].item()
                    epoch_kl += loss_dict["kl_loss"].item()
                    epoch_waveform += loss_dict["waveform_loss"].item()
                    epoch_spectral += loss_dict["spectral_loss"].item()
                    epoch_skip += loss_dict["skip_loss"].item()
                    n_batches += 1

                    running_loss = epoch_loss / n_batches
                    running_kl = epoch_kl / n_batches
                    running_skip = epoch_skip / n_batches
                    progress_bar.set_postfix(
                        {
                            "loss": f"{running_loss:.4f}",
                            "kl": f"{running_kl:.5f}",
                            "skip": f"{running_skip:.5f}",
                            "Î²": f"{current_beta:.5f}",
                        }
                    )

                avg_loss = epoch_loss / n_batches
                avg_kl = epoch_kl / n_batches

                # Check model health periodically
                if epoch % 50 == 0:
                    if self.model.check_for_nan():
                        logger.error(
                            f"NaN detected in model parameters at epoch {epoch}! Resetting..."
                        )
                        self.model.reset_nan_parameters()

                scheduler.step()
                current_lr = optimizer.param_groups[0]["lr"]

                # Update history and best loss tracking
                if "loss" not in self._training_history:
                    self._training_history["loss"] = []
                self._training_history["loss"].append(avg_loss)

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._best_loss = best_loss

                # Periodically update optimizer/scheduler state for safety
                if epoch % 10 == 0:
                    self._optimizer_state = optimizer.state_dict()
                    self._scheduler_state = scheduler.state_dict()

                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch {epoch:4d} | "
                        f"Loss: {avg_loss:8.6f} | "
                        f"KL: {avg_kl:6.2f} | "
                        f"Beta: {current_beta:.4f} | "
                        f"LR: {current_lr:.6f}"
                    )

                metrics = TrainingMetrics(
                    epoch=epoch,
                    total_loss=avg_loss,
                    reconstruction_loss=epoch_recon / n_batches,
                    kl_loss=epoch_kl / n_batches,
                    learning_rate=current_lr,
                    epoch_time=0.0,
                    beta_kl=current_beta,
                    waveform_loss=epoch_waveform / n_batches,
                    spectral_loss=epoch_spectral / n_batches,
                    skip_loss=epoch_skip / n_batches,
                    skip_dropout=current_skip_dropout,
                )

                self.training_logger.log_metrics(epoch, metrics)

                if config.callback:
                    config.callback(epoch, avg_loss, metrics.to_dict())

                if epoch % (LOG_AUDIO_INTERVAL) == 0:
                    self.model.eval()
                    with torch.no_grad():
                        # 1. Reconstruction with Skips (Fixed Skips - "The Cheated Version")
                        recon_fixed, _, _, _, _ = self.model(fixed_sample)
                        self.training_logger.log_audio(
                            "Reconstruction/Fixed_Skips",
                            recon_fixed,
                            self.config.sample_rate,
                            epoch,
                        )

                        recon_np = recon_fixed.detach().cpu().numpy().squeeze()
                        recon_img = plot_waveform_image(
                            recon_np, self.config.sample_rate, f"Fixed Skip Recon (Epoch {epoch})"
                        )
                        self.training_logger.log_image(
                            "Plots/Reconstruction_Fixed", recon_img, epoch
                        )
                        self.log_mel_spectrogram("Reconstruction_Fixed", recon_np, epoch)

                        # 2. Reconstruction WITHOUT Skips (Pure Latent - "The Honest Version")
                        mu, _ = self.model.encode(fixed_sample)
                        # Simple call to decode(mu) uses generated skips (inference mode)
                        recon_pure = self.model.decode(mu)
                        self.training_logger.log_audio(
                            "Reconstruction/Pure_Latent", recon_pure, self.config.sample_rate, epoch
                        )

                        pure_np = recon_pure.detach().cpu().numpy().squeeze()
                        pure_img = plot_waveform_image(
                            pure_np, self.config.sample_rate, f"Pure Latent Recon (Epoch {epoch})"
                        )
                        self.training_logger.log_image("Plots/Reconstruction_Pure", pure_img, epoch)
                        self.log_mel_spectrogram("Reconstruction_Pure", pure_np, epoch)

                        # 3. Random Generation (Pure Latent)
                        random_z = self.model.sample_latent(1, str(self.device))
                        generated = self.model.decode(random_z)
                        self.training_logger.log_audio(
                            "Generation/Random", generated, self.config.sample_rate, epoch
                        )

                        gen_np = generated.detach().cpu().numpy().squeeze()
                        gen_img = plot_waveform_image(
                            gen_np, self.config.sample_rate, f"Random Generation (Epoch {epoch})"
                        )
                        self.training_logger.log_image("Plots/Generation", gen_img, epoch)
                        self.log_mel_spectrogram("Generation", gen_np, epoch)

                    self.model.train()
        finally:
            # Final state update - ensures we save progress even if interrupted
            self._total_epochs_trained = epoch
            self._optimizer_state = optimizer.state_dict()
            self._scheduler_state = scheduler.state_dict()

            if self.training_logger:
                self.training_logger.flush()
                self.training_logger.close()
                self.training_logger = None

        return {"total_loss": avg_loss, "kl_loss": avg_kl}

    def generate_from_latent(
        self, latent_vector: LatentVector, config: Optional[GenerationConfig] = None
    ) -> AudioArray:
        """Generate audio from a latent vector."""
        if self.model is None:
            raise RuntimeError("Model not built.")
        self.load_to_vram()
        if config is None:
            config = GenerationConfig()
        self.model.eval()
        z = latent_vector
        if z.ndim == 1:
            z = z.reshape(1, -1)
        z_tensor = torch.tensor(z, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            audio = self.model.decode(z_tensor).cpu().numpy().squeeze()
        if config.apply_postprocessing:
            audio = self._postprocess(audio)
        return audio

    def generate_random(
        self, n_samples: int = 1, config: Optional[GenerationConfig] = None
    ) -> List[AudioArray]:
        """Generate random samples."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        if config is None:
            config = GenerationConfig()
        results = []
        for i in range(n_samples):
            z = self.model.sample_latent(1, str(self.device)).cpu().numpy().squeeze()
            audio = self.generate_from_latent(z, config)
            results.append(audio)
        return results

    def encode_audio(self, audio: AudioArray) -> LatentVector:
        """Encode audio to latent vector."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        self.model.eval()
        x = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.model.encode(x)
            return mu.cpu().numpy().squeeze()

    def _postprocess(self, audio: AudioArray) -> AudioArray:
        """Apply postprocessing with signal-adaptive gating and de-clicking."""
        # 1. Standard Low-Pass Filter
        sos = signal.butter(
            LOW_PASS_ORDER,
            DEFAULT_POSTPROCESS_LOW_PASS,
            "lp",
            fs=self.config.sample_rate,
            output="sos",
        )
        audio = signal.sosfilt(sos, audio)

        # 2. Smear-Killer: Signal-Adaptive Gate (-60dB)
        # Only fades out if the signal naturally dropped to "silence" before the end
        abs_audio = np.abs(audio)
        threshold = 10 ** (-60 / 20)  # 0.001 (-60dB)

        hot_indices = np.where(abs_audio > threshold)[0]
        if len(hot_indices) > 0:
            last_hot = hot_indices[-1]
            # If the signal went silent >20ms before the end, clean the remaining smear
            if last_hot < len(audio) - (self.config.sample_rate // 50):
                fade_start = last_hot + 10  # Tiny buffer
                fade_len = 512  # Smooth ~11ms fade
                fade_end = min(fade_start + fade_len, len(audio))

                if fade_start < len(audio):
                    fade_curve = np.linspace(1.0, 0.0, fade_end - fade_start)
                    audio[fade_start:fade_end] *= fade_curve
                    audio[fade_end:] = 0.0
        else:
            # Signal is entirely below threshold, zero it
            audio[:] = 0.0

        # 3. Micro-Declick (0.1ms / 4-sample taper)
        # Only applied if the very last sample isn't already zero
        if len(audio) > 4 and audio[-1] != 0:
            audio[-4:] *= np.linspace(0.75, 0.0, 4)

        max_amp = np.max(np.abs(audio))
        if max_amp > 1.0:
            audio = audio / max_amp
        return audio

    def interpolate(
        self, z1: LatentVector, z2: LatentVector, n_steps: int = 10, method: str = "linear"
    ) -> List[LatentVector]:
        return self.interpolator.interpolate(z1, z2, n_steps, method)

    def random_walk(
        self,
        start_vector: Optional[LatentVector] = None,
        n_steps: int = 8,
        step_size: float = 0.4,
        temperature: float = 1.0,
        momentum: float = 0.5,
        origin_pull: float = 0.1,
    ) -> List[LatentVector]:
        return self.walker.random_walk(
            start_vector, n_steps, step_size, temperature, momentum, origin_pull
        )

    def save_audio(self, audio: AudioArray, filepath: Union[str, Path]) -> None:
        """Save audio to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(filepath), audio, self.config.sample_rate)

    def save_model(self, filepath: Union[str, Path], **kwargs) -> None:
        """Save model checkpoint with full training state."""
        if self.model is None:
            raise RuntimeError("No model to save")

        from ..types import ModelCheckpoint

        model_type = "simple" if isinstance(self.model, SimpleFastVAE) else "complex"

        checkpoint = ModelCheckpoint(
            model_state=self.model.state_dict(),
            model_type=model_type,
            config=self.config,
            optimizer_state=self._optimizer_state,
            scheduler_state=self._scheduler_state,
            total_epochs_trained=self._total_epochs_trained,
            best_loss=self._best_loss,
            training_history=self._training_history,
            presets={name: p for name, p in self.preset_manager.presets.items()},
            directions=self.attribute_explorer.discovered_directions,
            cached_latents=self.cached_latents,
            cached_labels=self.cached_labels,
        )

        torch.save(checkpoint.to_dict(), str(filepath))
        log_model_save(Path(filepath))

    def load_model(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load model checkpoint and restore training state."""
        checkpoint_dict = torch.load(str(filepath), map_location=self.device, weights_only=False)

        # Update config from checkpoint
        self.config.latent_dim = checkpoint_dict.get("latent_dim", self.config.latent_dim)
        self.waveform_length = checkpoint_dict.get("waveform_length", self.waveform_length)

        model_type = checkpoint_dict.get("model_type", "simple")

        if model_type == "simple" or "encoder.conv" in str(checkpoint_dict["model_state"].keys()):
            self.model = SimpleFastVAE(
                latent_dim=self.config.latent_dim, waveform_length=self.waveform_length
            ).to(self.device)
        else:
            self.model = UnconditionalVAE(
                latent_dim=self.config.latent_dim, waveform_length=self.waveform_length
            ).to(self.device)

        self.model.load_state_dict(checkpoint_dict["model_state"])

        # Restore training state
        self._total_epochs_trained = checkpoint_dict.get("total_epochs_trained", 0)
        self.config.sample_rate = checkpoint_dict.get("sample_rate", self.config.sample_rate)
        self.config.duration = checkpoint_dict.get("duration", self.config.duration)
        self.waveform_length = int(self.config.sample_rate * self.config.duration)
        self._best_loss = checkpoint_dict.get("best_loss", float("inf"))
        self._optimizer_state = checkpoint_dict.get("optimizer_state")
        self._scheduler_state = checkpoint_dict.get("scheduler_state")
        self._training_history = checkpoint_dict.get("training_history", {})

        # Restore Explorer Data
        explorer_data = checkpoint_dict.get("explorer_data", {})

        # Restore latents/labels for map
        self.cached_latents = (
            np.array(explorer_data["cached_latents"])
            if explorer_data.get("cached_latents") is not None
            else None
        )
        self.cached_labels = explorer_data.get("cached_labels")

        # Restore presets if available
        if "presets" in explorer_data:
            for name, p_data in explorer_data["presets"].items():
                try:
                    from ..types import LatentPreset

                    self.preset_manager.save_preset(LatentPreset.from_dict(p_data))
                except:
                    pass

        # Restore directions
        if "directions" in explorer_data:
            for name, d_data in explorer_data["directions"].items():
                try:
                    from ..types import DiscoveredDirection

                    self.attribute_explorer.discovered_directions[name] = (
                        DiscoveredDirection.from_dict(d_data)
                    )
                except:
                    pass

        self.model.eval()

        log_model_load(Path(filepath))

        # If we successfully loaded a model, we can definitely resume training it,
        # even if it's an "old" model with no total_epochs_trained key.
        can_resume = self.model is not None

        return {
            "loaded": True,
            "can_resume": can_resume,
            "total_epochs_trained": self._total_epochs_trained,
            "best_loss": self._best_loss,
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "training_history": self._training_history,
        }

    def start_tensorboard(self, port: int = 6006, host: str = "localhost") -> None:
        """Launch TensorBoard pointing to the absolute root logs directory."""
        if self.training_logger:
            self.training_logger.start_tensorboard(port, host)
        else:
            # Use the absolute root even in explorer mode
            from ..config import DEFAULT_LOGS_DIR

            absolute_logs_root = str(DEFAULT_LOGS_DIR.resolve().absolute())

            # Create a temporary logger just to launch the process with the correct ROOT
            temp_logger = TrainingLogger(log_dir=DEFAULT_LOGS_DIR, experiment_name="explorer")
            temp_logger.start_tensorboard(port, host)
            logger.info(f"TensorBoard launched at ROOT: {absolute_logs_root}")

    def get_training_state(self) -> Dict[str, Any]:
        return {
            "can_resume": self._total_epochs_trained > 0,
            "total_epochs_trained": self._total_epochs_trained,
            "best_loss": self._best_loss,
            "optimizer_state": self._optimizer_state,
            "scheduler_state": self._scheduler_state,
            "training_history": self._training_history,
        }

    def save_preset(self, name: str, vector: LatentVector, description: str = "") -> None:
        self.preset_manager.save_preset(
            LatentPreset(name=name, vector=vector, description=description)
        )

    def get_preset(self, name: str) -> Optional[LatentVector]:
        preset = self.preset_manager.get_preset(name)
        return preset.vector if preset else None

    def list_presets(self) -> List[str]:
        return self.preset_manager.list_presets()

    def delete_preset(self, name: str) -> bool:
        return self.preset_manager.delete_preset(name)

    def get_preset_info(self, name: str) -> Optional[LatentPreset]:
        return self.preset_manager.get_preset(name)

    # --- Attribute Discovery ---
    def discover_direction(
        self, name: str, positive_samples: List[LatentVector], negative_samples: List[LatentVector]
    ):
        return self.attribute_explorer.discover_direction(name, positive_samples, negative_samples)

    def apply_attribute(
        self, base_vector: LatentVector, direction_name: str, strength: float
    ) -> LatentVector:
        return self.attribute_explorer.apply_attribute(base_vector, direction_name, strength)

    # --- Projection ---
    def fit_projector(self, latent_vectors: np.ndarray):
        self.projector.fit(latent_vectors)

    def project_to_2d(self, latent_vectors: np.ndarray) -> np.ndarray:
        return self.projector.transform(latent_vectors)

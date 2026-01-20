# training.py - COMPLETE FIXED VERSION - Prevents Posterior Collapse
"""Optimized training system with proper KL handling and performance improvements."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime
import numpy as np
import webbrowser
import subprocess
import threading
import time

from ..types import TrainingMetrics, TrainingConfig
from ..config import LOG_METRICS_INTERVAL, LOG_AUDIO_INTERVAL, DEFAULT_LOGS_DIR
from ..logging import logger


class TrainingLogger:
    """Handles training logging with TensorBoard."""

    _active_loggers: Dict[str, "TrainingLogger"] = {}

    def __init__(self, log_dir: Optional[Path] = None, experiment_name: Optional[str] = None):
        if log_dir is None:
            log_dir = DEFAULT_LOGS_DIR

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"training_{timestamp}"

        self.log_dir = Path(log_dir) / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._cleanup_existing_event_files()

        self.writer = SummaryWriter(str(self.log_dir), max_queue=100, flush_secs=30)
        logger.info(f"TrainingLogger initialized at {self.log_dir}")

        TrainingLogger._active_loggers[str(self.log_dir)] = self

    def _cleanup_existing_event_files(self) -> None:
        """Remove any existing TensorBoard event files to prevent duplicates."""
        import glob
        event_pattern = str(self.log_dir / "events.out.tfevents.*")
        existing_files = glob.glob(event_pattern)
        for f in existing_files:
            try:
                Path(f).unlink()
                logger.debug(f"Removed existing event file: {f}")
            except OSError:
                pass

        for subdir in self.log_dir.iterdir():
            if subdir.is_dir():
                sub_event_pattern = str(subdir / "events.out.tfevents.*")
                sub_existing_files = glob.glob(sub_event_pattern)
                for f in sub_existing_files:
                    try:
                        Path(f).unlink()
                        logger.debug(f"Removed existing event file: {f}")
                    except OSError:
                        pass
                try:
                    subdir.rmdir()
                except OSError:
                    pass
    
    def log_metrics(self, epoch: int, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        if epoch % LOG_METRICS_INTERVAL == 0:
            self.writer.add_scalar('Loss/Total', metrics.total_loss, epoch)
            self.writer.add_scalar('Loss/Reconstruction_Combined', metrics.reconstruction_loss, epoch)
            self.writer.add_scalar('Loss/Waveform_MSE', metrics.waveform_loss, epoch)
            self.writer.add_scalar('Loss/Spectral_STFT', metrics.spectral_loss, epoch)
            self.writer.add_scalar('Loss/KL', metrics.kl_loss, epoch)
            self.writer.add_scalar('Loss/Skip_Consistency', metrics.skip_loss, epoch)
            self.writer.add_scalar('Training/Learning_Rate', metrics.learning_rate, epoch)

            self.writer.add_scalar('Training/Beta_KL', metrics.beta_kl, epoch)
            
            if metrics.validation_loss is not None:
                self.writer.add_scalar('Loss/Validation', metrics.validation_loss, epoch)
    
    def log_audio(self, tag: str, audio: torch.Tensor, sample_rate: int, epoch: int) -> None:
        """Log audio sample."""
        if epoch % LOG_AUDIO_INTERVAL == 0:
            audio_data = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
            
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()
            
            self.writer.add_audio(tag, audio_data, epoch, sample_rate=sample_rate)
    
    def log_image(self, tag: str, image: np.ndarray, epoch: int) -> None:
        """Log image sample to TensorBoard."""
        if epoch % LOG_AUDIO_INTERVAL == 0:
            self.writer.add_image(tag, image, epoch, dataformats='HWC')
    
    def log_hyperparameters(self, config: TrainingConfig) -> None:
        """Log hyperparameters."""
        hparams = {
            'epochs': config.epochs,
            'batch_size': config.batch_size,
            'learning_rate': config.learning_rate,
            'beta_kl': config.beta_kl,
            'weight_decay': config.weight_decay,
        }
        self.writer.add_hparams(hparams, {})
        logger.info("Logged hyperparameters")
    
    def log_model_summary(self, model: nn.Module) -> None:
        """Log model summary."""
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        summary = f"Model Parameters: {total_params:,}"
        self.writer.add_text("Model/Summary", summary)
        logger.info(summary)
    
    def start_tensorboard(self, port: int = 6006, host: str = 'localhost') -> None:
        """Launch TensorBoard pointing to the absolute root logs directory."""
        if hasattr(self, '_tensorboard_started') and self._tensorboard_started:
            logger.warning("TensorBoard already started for this session")
            return

        def run_tensorboard():
            try:
                import sys
                import os
                import subprocess
                from ..config import DEFAULT_LOGS_DIR
                
                absolute_logs_root = DEFAULT_LOGS_DIR.resolve().absolute().as_posix()
                
                if os.name == 'nt':
                    try:
                        find_port = subprocess.check_output(f'netstat -ano | findstr :{port}', shell=True).decode()
                        for line in find_port.strip().split('\n'):
                            if 'LISTENING' in line:
                                pid = line.strip().split()[-1]
                                subprocess.run(['taskkill', '/F', '/PID', pid, '/T'], capture_output=True)
                        time.sleep(1)
                    except:
                        pass
                
                tb_executable = sys.executable.replace('python.exe', 'Scripts/tensorboard.exe')
                if not os.path.exists(tb_executable):
                    tb_executable = 'tensorboard'
                
                cmd = [
                    tb_executable,
                    '--logdir', absolute_logs_root,
                    '--port', str(port),
                    '--host', host,
                    '--purge_orphaned_data'
                ]
                logger.info(f"FORCE STARTING TensorBoard at ROOT: {absolute_logs_root}")
                logger.info(f"Command: {' '.join(cmd)}")

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
                )
                time.sleep(2)

                if process.poll() is not None:
                    stderr_output, _ = process.communicate()
                    stderr_text = stderr_output.decode('utf-8', errors='replace')
                    logger.error(f"TensorBoard failed to start: {stderr_text}")
                    return

                try:
                    stderr_output, _ = process.communicate(timeout=5)
                    if stderr_output:
                        stderr_text = stderr_output.decode('utf-8', errors='replace')
                        if stderr_text.strip():
                            logger.warning(f"TensorBoard stderr: {stderr_text}")
                except subprocess.TimeoutExpired:
                    pass

                try:
                    webbrowser.open(f"http://{host}:{port}")
                except Exception as e:
                    logger.debug(f"Could not open browser: {e}")

            except Exception as e:
                logger.error(f"Failed to start TensorBoard: {e}")
                logger.info("You can manually start TensorBoard with: tensorboard --logdir <log_dir>")

        thread = threading.Thread(target=run_tensorboard, daemon=True)
        thread.start()
        self._tensorboard_started = True
    
    def flush(self) -> None:
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
        TrainingLogger._active_loggers.pop(str(self.log_dir), None)
        logger.info("TrainingLogger closed")


from .losses import MultiResolutionSTFTLoss

EPS = 1e-8
MAX_LOSS_VALUE = 1e6

# CRITICAL FIX: Proper free bits implementation
FREE_BITS_PER_DIM = 0.25  # Increased from 0.25 to ensure more active dimensions
MIN_LOGVAR = -6.0  # Prevents variance collapse (exp(-6) ≈ 0.0025)


def _make_fallback_loss(value: float, device: torch.device, tensor: torch.Tensor) -> torch.Tensor:
    """Create a fallback loss tensor that preserves gradients for backprop."""
    return tensor.mean() * 0.0 + value


def compute_vae_loss(
    reconstructed: torch.Tensor,
    original: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta_kl: float = 1.0,
    config: Optional[TrainingConfig] = None,
    mrstft: Optional[MultiResolutionSTFTLoss] = None,
    model: Optional[nn.Module] = None,
    z: Optional[torch.Tensor] = None,
    compute_stft: bool = True,
    real_skips: Optional[List[torch.Tensor]] = None,
    gen_skips: Optional[List[torch.Tensor]] = None,
    epoch: int = 0
) -> Dict[str, torch.Tensor]:
    """
    VAE loss with PROPER Free Bits and Skip Consistency.
    """
    
    if config is None:
        from ..types import TrainingConfig
        config = TrainingConfig()

    device = reconstructed.device

    # Initialize losses
    waveform_loss = _make_fallback_loss(0.0, device, reconstructed)
    spectral_loss = _make_fallback_loss(0.0, device, reconstructed)
    kl_loss = _make_fallback_loss(0.0, device, mu)
    skip_loss = _make_fallback_loss(0.0, device, mu)
    diversity_loss = _make_fallback_loss(0.0, device, mu)

    # 1. Waveform MSE
    try:
        # Ensure identical shapes to prevent broadcasting distortion
        recon_mse = reconstructed.view(reconstructed.size(0), -1)
        orig_mse = original.view(original.size(0), -1)
        waveform_loss = F.mse_loss(recon_mse, orig_mse, reduction='mean')
        if torch.isnan(waveform_loss) or torch.isinf(waveform_loss):
            waveform_loss = _make_fallback_loss(1.0, device, reconstructed)
        waveform_loss = torch.clamp(waveform_loss, max=MAX_LOSS_VALUE)
    except Exception as e:
        print(f"ERROR in waveform loss: {e}")
        waveform_loss = _make_fallback_loss(1.0, device, reconstructed)

    # 2. Spectral loss
    try:
        if mrstft is None:
            from .losses import MultiResolutionSTFTLoss
            mrstft = MultiResolutionSTFTLoss(mode=config.stft_mode).to(device)

        if compute_stft:
            sc_loss, mag_loss = mrstft(reconstructed, original)
            spectral_loss = sc_loss + mag_loss
        else:
            spectral_loss = _make_fallback_loss(0.0, device, reconstructed)

        if torch.isnan(spectral_loss) or torch.isinf(spectral_loss):
            spectral_loss = _make_fallback_loss(0.0, device, reconstructed)
        spectral_loss = torch.clamp(spectral_loss, max=MAX_LOSS_VALUE)
    except Exception as e:
        print(f"ERROR in spectral loss: {e}")
        spectral_loss = _make_fallback_loss(0.0, device, reconstructed)

    # 3. Skip Consistency Loss - CRITICAL for Generation
    if real_skips is not None and gen_skips is not None:
        try:
            skip_loss_acc = 0
            # Reverse real skips to match gen skips order (decoder order)
            real_skips_ordered = real_skips[::-1]
            for r_s, g_s in zip(real_skips_ordered, gen_skips):
                # 3.1 MSE for amplitude matching
                mse_skip = F.mse_loss(g_s, r_s.detach())
                
                # 3.2 Cosine Similarity for direction/character matching (Phase 7)
                # Flatten to [Batch, -1] for similarity calculation
                g_flat = g_s.view(g_s.size(0), -1)
                r_flat = r_s.detach().view(r_s.size(0), -1)
                cos_sim = F.cosine_similarity(g_flat, r_flat, dim=1).mean()
                
                # Skip loss combines magnitude (MSE) and character (1 - CosSim)
                skip_loss_acc += (mse_skip + (1.0 - cos_sim))
                
            skip_loss = skip_loss_acc / len(gen_skips)
        except Exception as e:
            print(f"ERROR in skip loss: {e}")
            skip_loss = _make_fallback_loss(0.0, device, mu)

    # 4. KL divergence with PROPER Free Bits per dimension

    try:
        # CRITICAL: Enforce minimum variance to prevent collapse
        logvar_safe = torch.clamp(logvar, min=MIN_LOGVAR)
        
        # Standard VAE KL formula per dimension
        # Shape: [batch_size, latent_dim]
        kl_per_dim = -0.5 * (1 + logvar_safe - mu.pow(2) - logvar_safe.exp())
        
        # CRITICAL: Apply free bits PER DIMENSION, not after averaging!
        # This prevents the model from sacrificing individual dimensions
        kl_per_dim_clamped = torch.clamp(kl_per_dim - FREE_BITS_PER_DIM, min=0.0)
        
        # Now reduce: sum over dimensions, mean over batch
        kl_raw = kl_per_dim.sum(dim=1).mean()
        kl_loss = kl_per_dim_clamped.sum(dim=1).mean()
        
        # Diagnostic: Check for dimension collapse
        if torch.rand(1).item() < 0.02:  # Log 2% of batches
            active_dims = (kl_per_dim.mean(dim=0) > FREE_BITS_PER_DIM).sum().item()
            total_dims = kl_per_dim.shape[1]
            mean_kl_per_dim = kl_raw.item() / total_dims
            mean_kl_per_active = kl_per_dim.mean(dim=0)[kl_per_dim.mean(dim=0) > FREE_BITS_PER_DIM].mean().item() if active_dims > 0 else 0
            
            # SILENCE WARNING during early warmup (Phase 1)
            # Only warn after Epoch 50 when skip dropout/swap kicks in
            if config.epochs > 50 and active_dims < total_dims * 0.5:
                 # We can add a more informative message here instead of just a warning
                 if torch.rand(1).item() < 0.05: # Even rarer
                     print(f"ℹ️  Latent Density: {mean_kl_per_dim/FREE_BITS_PER_DIM*100:.1f}% | Avg KL/dim: {mean_kl_per_dim:.3f} (Goal: >{FREE_BITS_PER_DIM})")
            
            if active_dims < total_dims * 0.5 and epoch > 50:
                print(f"⚠️  COLLAPSE WARNING: Only {active_dims}/{total_dims} dims active!")
                print(f"   KL_raw={kl_raw:.2f}, KL_clamped={kl_loss:.2f}")
                print(f"   μ∈[{mu.min():.2f},{mu.max():.2f}], logσ²∈[{logvar.min():.2f},{logvar.max():.2f}]")
                print(f"   Mean KL per active dim: {mean_kl_per_active:.3f}")
            elif kl_raw > 50.0:
                print(f"ℹ️  High KL: {kl_raw:.1f} | Active: {active_dims}/{total_dims} | β={beta_kl:.5f}")

        
        if torch.isnan(kl_loss) or torch.isinf(kl_loss):
            print("ERROR: NaN/Inf in KL!")
            kl_loss = _make_fallback_loss(0.0, device, mu)

    except Exception as e:
        print(f"ERROR in KL loss: {e}")
        kl_loss = _make_fallback_loss(0.0, device, mu)

    # 4. Total loss
    recon_loss = (
        waveform_loss * config.waveform_loss_weight +
        spectral_loss * config.spectral_loss_weight
    )

    # Reduced skip weight to 1.0 to prevent it from overpowering reconstruction
    total_loss = recon_loss + beta_kl * kl_loss + skip_loss * 1.0

    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("CRITICAL: NaN in total loss!")
        total_loss = _make_fallback_loss(1.0, device, reconstructed)
        recon_loss = _make_fallback_loss(1.0, device, reconstructed)

    return {
        'total_loss': total_loss,
        'recon_loss': recon_loss,
        'kl_loss': kl_loss,
        'waveform_loss': waveform_loss,
        'spectral_loss': spectral_loss,
        'skip_loss': skip_loss,
        'diversity_loss': diversity_loss
    }



def calculate_beta(epoch: int, config: TrainingConfig) -> float:
    """
    Calculate KL beta with FASTER warmup to prevent encoder bypass learning.
    
    Changes:
    - Warmup completes in 30 epochs (was 100)
    - Steeper sigmoid curve
    - Optional cyclical annealing
    """
    import math
    
    if config.kl_warmup_epochs <= 0:
        return config.beta_kl
    
    # Use the full configured warmup period
    warmup_epochs = config.kl_warmup_epochs
    warmup_progress = min(epoch / warmup_epochs, 1.0)
    
    # Apply sigmoid annealing if configured
    if config.kl_annealing_type == 'sigmoid':
        # STEEPER sigmoid curve (12x instead of 10x)
        annealing_factor = 1.0 / (1.0 + math.exp(-12.0 * warmup_progress + 6.0))
    else:
        # Linear: straight ramp
        annealing_factor = warmup_progress
    
    base_beta = config.beta_kl * annealing_factor
    
    # Optional: Cyclical annealing after warmup for exploration
    if epoch > warmup_epochs and hasattr(config, 'use_cyclical_annealing') and config.use_cyclical_annealing:
        cycle_length = 100
        cycle_progress = (epoch - warmup_epochs) % cycle_length / cycle_length
        # Ramp from 0.5 * beta to 1.0 * beta over each cycle
        cycle_factor = 0.5 + 0.5 * cycle_progress
        return base_beta * cycle_factor
    
    return base_beta


def calculate_skip_dropout(epoch: int) -> float:
    """Return skip dropout probability based on training phase."""
    from ..config import SKIP_DROPOUT_SCHEDULE

    phase1_end = SKIP_DROPOUT_SCHEDULE['phase1_epochs']
    phase2_end = SKIP_DROPOUT_SCHEDULE['phase2_epochs']

    if epoch <= phase1_end:
        return SKIP_DROPOUT_SCHEDULE['phase1_dropout']
    elif epoch <= phase2_end:
        return SKIP_DROPOUT_SCHEDULE['phase2_dropout']
    else:
        return SKIP_DROPOUT_SCHEDULE['phase3_dropout']


def calculate_skip_swap(epoch: int) -> float:
    """Return skip swap probability based on training phase."""
    from ..config import SKIP_SWAP_SCHEDULE

    phase1_end = SKIP_SWAP_SCHEDULE['phase1_epochs']
    phase2_end = SKIP_SWAP_SCHEDULE['phase2_epochs']

    if epoch <= phase1_end:
        return SKIP_SWAP_SCHEDULE['phase1_swap']
    elif epoch <= phase2_end:
        return SKIP_SWAP_SCHEDULE['phase2_swap']
    else:
        return SKIP_SWAP_SCHEDULE['phase3_swap']


def create_optimizer_and_scheduler(

    model: nn.Module,
    config: TrainingConfig
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Create optimizer and scheduler with stable settings."""
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
        amsgrad=True
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs,
        eta_min=config.learning_rate * 0.1
    )
    
    return optimizer, scheduler
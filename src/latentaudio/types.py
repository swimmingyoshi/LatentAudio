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
# types.py - Type definitions and dataclasses for LatentAudio
"""Type definitions and dataclasses for LatentAudio."""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Protocol, Callable, Self
from pathlib import Path
import numpy as np
from .config import (
    LATENT_DIM,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_DURATION,
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BETA_KL,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SCHEDULER_FACTOR,
    DEFAULT_GRAD_CLIP,
    STFT_MODE,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
    LOG_BATCH_INTERVAL,
    STFT_LOSS_SKIP_INTERVAL,
)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# ============================================================================
# TYPE ALIASES
# ============================================================================
AudioArray = np.ndarray  # Shape: (n_samples,) or (batch_size, n_samples)
LatentVector = np.ndarray  # Shape: (latent_dim,) or (batch_size, latent_dim)
PresetDict = Dict[str, Any]  # Serialized preset data


# ============================================================================
# PROTOCOLS
# ============================================================================
class TrainingCallback(Protocol):
    """Protocol for training progress callbacks."""

    def __call__(self, epoch: int, loss: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """Called during training with progress information."""
        ...


class GenerationCallback(Protocol):
    """Protocol for generation progress callbacks."""

    def __call__(self, step: int, total_steps: int, audio: Optional[AudioArray] = None) -> None:
        """Called during generation with progress information."""
        ...


# ============================================================================
# NESTED CONFIGURATION (PHASE 4: CLEANER API)
# ============================================================================
@dataclass(frozen=True)
class AudioConfig:
    """Audio processing configuration."""

    sample_rate: int = 44100
    duration: float = 0.5

    def __post_init__(self):
        """Validate configuration after initialization."""
        from .validator import validate_sample_rate, validate_duration

        sr_result = validate_sample_rate(self.sample_rate)
        if not sr_result.is_valid:
            raise ValueError(f"Invalid sample rate: {sr_result.errors}")

        dur_result = validate_duration(self.duration)
        if not dur_result.is_valid:
            raise ValueError(f"Invalid duration: {dur_result.errors}")

    @property
    def waveform_length(self) -> int:
        """Calculate waveform length from sample rate and duration."""
        return int(self.sample_rate * self.duration)


@dataclass(frozen=True)
class ModelConfig:
    """Neural network model configuration."""

    latent_dim: int = 128

    def __post_init__(self):
        """Validate configuration after initialization."""
        from .validator import validate_latent_dim

        dim_result = validate_latent_dim(self.latent_dim)
        if not dim_result.is_valid:
            raise ValueError(f"Invalid latent dimension: {dim_result.errors}")


@dataclass(frozen=True)
class DeviceConfig:
    """Device configuration for computation."""

    device: Optional[str] = None

    def __post_init__(self):
        if self.device is None:
            if TORCH_AVAILABLE and torch is not None:
                self.__dict__["device"] = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.__dict__["device"] = "cpu"

    @property
    def torch_device(self) -> torch.device:
        """Get PyTorch device object."""
        if TORCH_AVAILABLE and torch is not None:
            return torch.device(self.device)
        else:
            raise RuntimeError("PyTorch not available")


# ============================================================================
# CONFIGURATION DATACLASSES
# ============================================================================
@dataclass
class GeneratorConfig:
    """Configuration for audio generator initialization."""

    # Legacy flat API (backwards compatible)
    sample_rate: int = DEFAULT_SAMPLE_RATE
    duration: float = DEFAULT_DURATION
    latent_dim: int = LATENT_DIM
    device: Optional[str] = None  # None = auto-detect (cuda/cpu)

    # New nested API (optional)
    audio: Optional[AudioConfig] = None
    model: Optional[ModelConfig] = None
    device_obj: Optional[DeviceConfig] = None

    def __post_init__(self):
        # Handle new nested API
        if self.audio is not None:
            self.sample_rate = self.audio.sample_rate
            self.duration = self.audio.duration
        if self.model is not None:
            self.latent_dim = self.model.latent_dim
        if self.device_obj is not None:
            self.device = self.device_obj.device

        # Auto-detect device if not specified
        if self.device is None:
            if TORCH_AVAILABLE and torch is not None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"

    @property
    def waveform_length(self) -> int:
        """Calculate waveform length."""
        return int(self.sample_rate * self.duration)

    @property
    def audio_config(self) -> AudioConfig:
        """Get audio configuration."""
        return AudioConfig(sample_rate=self.sample_rate, duration=self.duration)

    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration."""
        return ModelConfig(latent_dim=self.latent_dim)

    @property
    def device_config(self) -> DeviceConfig:
        """Get device configuration."""
        return DeviceConfig(device=self.device)

    @classmethod
    def from_nested(
        cls, audio: AudioConfig, model: ModelConfig, device: DeviceConfig
    ) -> "GeneratorConfig":
        """Create from nested configuration objects."""
        return cls(
            sample_rate=audio.sample_rate,
            duration=audio.duration,
            latent_dim=model.latent_dim,
            device=device.device,
            audio=audio,
            model=model,
            device_obj=device,
        )


@dataclass
@dataclass
class TrainingConfig:
    """Configuration for model training - ANTI-COLLAPSE VERSION."""

    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    beta_kl: float = 0.005  # INCREASED from 0.002 for stronger latent space
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    scheduler_patience: int = DEFAULT_SCHEDULER_PATIENCE
    scheduler_factor: float = DEFAULT_SCHEDULER_FACTOR
    grad_clip: float = DEFAULT_GRAD_CLIP
    latent_dim: int = LATENT_DIM

    # CRITICAL: Annealing and Loss improvements for anti-collapse
    kl_warmup_epochs: int = 30  # REDUCED from 100 - faster warmup prevents bypass learning
    kl_annealing_type: str = "linear"  # 'linear' or 'sigmoid'
    use_cyclical_annealing: bool = False  # Optional: set to True for exploration
    spectral_loss_weight: float = 5.0
    waveform_loss_weight: float = 5.0  # Balanced Phase 7 Weights
    stft_mode: str = STFT_MODE

    # VRAM / Performance
    use_amp: bool = False  # Disabled by default due to stability issues
    gradient_accumulation_steps: int = DEFAULT_GRADIENT_ACCUMULATION_STEPS
    batch_log_interval: int = LOG_BATCH_INTERVAL
    stft_skip_interval: int = STFT_LOSS_SKIP_INTERVAL

    callback: Optional[object] = None  # TrainingCallback type

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callback)."""
        data = asdict(self)
        data.pop("callback", None)  # Callbacks can't be serialized
        return data


@dataclass
class GenerationConfig:
    """Configuration for audio generation."""

    temperature: float = 1.0
    apply_postprocessing: bool = True
    normalize: bool = True
    callback: Optional[GenerationCallback] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callback)."""
        data = asdict(self)
        data.pop("callback", None)  # Callbacks can't be serialized
        return data


@dataclass
class InterpolationConfig:
    """Configuration for latent space interpolation."""

    method: str = "spherical"  # 'linear' or 'spherical'
    n_steps: int = 10
    callback: Optional[GenerationCallback] = None

    def __post_init__(self):
        if self.method not in ["linear", "spherical"]:
            raise ValueError(f"Invalid interpolation method: {self.method}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callback)."""
        data = asdict(self)
        data.pop("callback", None)
        return data


@dataclass
class WalkConfig:
    """Configuration for random walks in latent space."""

    n_steps: int = 8
    step_size: float = 0.4
    momentum: float = 0.5
    origin_pull: float = 0.1
    start_vector: Optional[LatentVector] = None
    callback: Optional[GenerationCallback] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding callback and array)."""
        data = asdict(self)
        data.pop("callback", None)
        data.pop("start_vector", None)
        return data


# ============================================================================
# TRAINING METRICS
# ============================================================================
@dataclass
class TrainingMetrics:
    """Metrics collected during training."""

    epoch: int
    total_loss: float
    reconstruction_loss: float
    kl_loss: float
    learning_rate: float
    epoch_time: float
    beta_kl: float = 0.0  # Added for tracking annealing
    waveform_loss: float = 0.0  # Added for detailed tracking
    spectral_loss: float = 0.0  # Added for detailed tracking
    skip_loss: float = 0.0  # Added for skip consistency tracking
    skip_dropout: float = 0.0  # Added for skip generation training

    validation_loss: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "epoch": self.epoch,
            "total_loss": self.total_loss,
            "reconstruction_loss": self.reconstruction_loss,
            "kl_loss": self.kl_loss,
            "learning_rate": self.learning_rate,
            "epoch_time": self.epoch_time,
            "validation_loss": self.validation_loss,
        }


# ============================================================================
# VALIDATION RESULTS
# ============================================================================
@dataclass
class ValidationResult:
    """Result of input validation."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add a validation error."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a validation warning."""
        self.warnings.append(message)


# ============================================================================
# LATENT SPACE TYPES
# ============================================================================
@dataclass
class LatentPreset:
    """A named latent space preset."""

    name: str
    latent_vector: LatentVector
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "latent_vector": (
                self.latent_vector.tolist()
                if hasattr(self.latent_vector, "tolist")
                else self.latent_vector
            ),
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LatentPreset":
        """Create from dictionary (e.g., from JSON)."""
        import numpy as np

        return cls(
            name=data["name"],
            latent_vector=np.array(data["latent_vector"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
        )


@dataclass
class DiscoveredDirection:
    """A discovered direction in latent space."""

    name: str
    direction_vector: LatentVector
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "direction_vector": (
                self.direction_vector.tolist()
                if hasattr(self.direction_vector, "tolist")
                else self.direction_vector
            ),
            "description": self.description,
            "tags": self.tags,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiscoveredDirection":
        """Create from dictionary (e.g., from JSON)."""
        import numpy as np

        return cls(
            name=data["name"],
            direction_vector=np.array(data["direction_vector"]),
            description=data.get("description", ""),
            tags=data.get("tags", []),
            created_at=data.get("created_at", ""),
        )


# ============================================================================
# MODEL CHECKPOINT
# ============================================================================
@dataclass
class ModelCheckpoint:
    """Complete model checkpoint for saving/loading."""

    model_state: Dict[str, Any]
    model_type: str
    config: GeneratorConfig
    training_config: Optional[TrainingConfig] = None
    presets: Dict[str, LatentPreset] = field(default_factory=dict)
    directions: Dict[str, DiscoveredDirection] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Cached Map Data (Phase 8)
    cached_latents: Optional[np.ndarray] = None
    cached_labels: Optional[List[str]] = None

    # Training state for resuming (Phase 6)

    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    current_epoch: int = 0
    total_epochs_trained: int = 0
    best_loss: float = float("inf")
    training_history: Dict[str, List[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize checkpoint to dictionary."""
        return {
            "model_state": self.model_state,
            "model_type": self.model_type,
            "sample_rate": self.config.sample_rate,
            "duration": self.config.duration,
            "waveform_length": int(self.config.sample_rate * self.config.duration),
            "latent_dim": self.config.latent_dim,
            "explorer_data": {
                "presets": {name: p.to_dict() for name, p in self.presets.items()},
                "directions": {name: d.to_dict() for name, d in self.directions.items()},
                "cached_latents": (
                    self.cached_latents.tolist() if self.cached_latents is not None else None
                ),
                "cached_labels": self.cached_labels,
            },
            "metadata": self.metadata,
            # Training state for resuming
            "optimizer_state": self.optimizer_state,
            "scheduler_state": self.scheduler_state,
            "current_epoch": self.current_epoch,
            "total_epochs_trained": self.total_epochs_trained,
            "best_loss": self.best_loss,
            "training_history": self.training_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelCheckpoint":
        """Deserialize checkpoint from dictionary."""
        import numpy as np

        # Extract explorer data
        explorer_data = data.get("explorer_data", {})
        cached_latents_raw = explorer_data.get("cached_latents")
        cached_latents = np.array(cached_latents_raw) if cached_latents_raw is not None else None

        return cls(
            model_state=data["model_state"],
            model_type=data["model_type"],
            config=GeneratorConfig(
                sample_rate=data.get("sample_rate", 44100),
                duration=data.get("duration", 0.5),
                latent_dim=data.get("latent_dim", 64),
            ),
            optimizer_state=data.get("optimizer_state"),
            scheduler_state=data.get("scheduler_state"),
            current_epoch=data.get("current_epoch", 0),
            total_epochs_trained=data.get("total_epochs_trained", 0),
            best_loss=data.get("best_loss", float("inf")),
            training_history=data.get("training_history", {}),
            presets={},
            directions={},
            metadata=data.get("metadata", {}),
            cached_latents=cached_latents,
            cached_labels=explorer_data.get("cached_labels"),
        )


# ============================================================================
# UTILITY TYPES
# ============================================================================
ModelBackend = str  # 'pytorch' - extensible for future backends
DeviceType = str  # 'cpu', 'cuda', 'mps', etc.
PathLike = str | Path

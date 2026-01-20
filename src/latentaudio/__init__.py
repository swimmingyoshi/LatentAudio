# src/latentaudio/__init__.py
"""LatentAudio: Direct Neural Audio Generation and Exploration.

A neural audio synthesis system that provides direct exploration of the learned
manifold of sounds through latent space manipulation.
"""

__version__ = "1.0.0"
__author__ = "LatentAudio Team"

# Import main classes for convenience
from .core.generator import AdvancedAudioGenerator
# Explorer functionality is now integrated into AdvancedAudioGenerator
from .logging import setup_logging
from .config import *
from .types import *

__all__ = [
    # Main classes
    "AdvancedAudioGenerator",

    # Config
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_DURATION",
    "LATENT_DIM",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_LEARNING_RATE",

    # Types - Legacy API
    "GeneratorConfig",
    "TrainingConfig",
    "GenerationConfig",
    "LatentPreset",
    "AudioArray",
    "LatentVector",

    # New Clean API (Phase 4)
    "AudioConfig",
    "ModelConfig",
    "DeviceConfig",
]
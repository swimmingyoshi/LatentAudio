# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LatentAudio - Direct Neural Audio Generation and Exploration
# Copyright (C) 2026 swimmingyoshi
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

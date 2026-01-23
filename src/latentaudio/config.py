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
# config.py - UPDATED Configuration for preventing posterior collapse

from pathlib import Path

# ============================================================================
# AUDIO SETTINGS
# ============================================================================
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION = 1.0
WAVEFORM_LENGTH = int(DEFAULT_SAMPLE_RATE * DEFAULT_DURATION)

# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================
LATENT_DIM = 128

# Dense VAE layers (fallback)
ENCODER_LAYERS = [1024, 512, 256]
DECODER_LAYERS = [512, 1024, 2048]

# U-Net Encoder
ENCODER_CONV_CHANNELS = [64, 128, 256, 512]
ENCODER_CONV_KERNEL_SIZES = [5, 5, 5, 5]
ENCODER_CONV_STRIDES = [4, 4, 4, 4]
ENCODER_FC_LAYERS = [1024]

# U-Net Decoder
DECODER_FC_LAYERS = [1024]
DECODER_CONV_CHANNELS = [512, 256, 128, 64]
DECODER_CONV_KERNEL_SIZES = [5, 5, 5, 5]
DECODER_CONV_STRIDES = [4, 4, 4, 4]
DECODER_CONV_PADDING = [1, 1, 1, 1]

# Skip connections
SKIP_CHANNELS = [64, 128, 256, 512]

# Activation and regularization
LEAKY_RELU_SLOPE = 0.2
ENCODER_DROPOUT_RATE = 0.1
DECODER_DROPOUT_RATE = 0.1
SKIP_DROPOUT_PROB = 0.5

# ============================================================================
# TRAINING DEFAULTS - ANTI-COLLAPSE SETTINGS
# ============================================================================
DEFAULT_EPOCHS = 1000
DEFAULT_BATCH_SIZE = 8
DEFAULT_LEARNING_RATE = 0.0001
DEFAULT_BETA_KL = 0.001
DEFAULT_WEIGHT_DECAY = 0.001
DEFAULT_GRAD_CLIP = 1.0  # Increased for Phase 7 complex gradients


# ============================================================================
# VRAM OPTIMIZATION
# ============================================================================
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 2

# Scheduler settings
DEFAULT_SCHEDULER_PATIENCE = 150
DEFAULT_SCHEDULER_FACTOR = 0.5

# ============================================================================
# GENERATION SETTINGS
# ============================================================================
DEFAULT_TEMPERATURE = 1.0
DEFAULT_POSTPROCESS_LOW_PASS = 10000
DEFAULT_FADE_LENGTH = 0.001

# ============================================================================
# EXPLORATION SETTINGS
# ============================================================================
DEFAULT_INTERPOLATION_STEPS = 10
DEFAULT_WALK_STEPS = 8
DEFAULT_WALK_STEP_SIZE = 0.4

# ============================================================================
# FILE PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent

DEFAULT_MODELS_DIR = PROJECT_ROOT / "TestModels"
DEFAULT_INTERPOLATIONS_DIR = PROJECT_ROOT / "Interpolations"
DEFAULT_WALK_DIR = PROJECT_ROOT / "Walk"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs"

# ============================================================================
# UI SETTINGS
# ============================================================================
MAX_VISIBLE_LATENT_DIMS = 128
SLIDER_RANGE = (-300, 300)
WAVEFORM_HEIGHT = 150
PREVIEW_HEIGHT = 120

# ============================================================================
# AUDIO PROCESSING
# ============================================================================
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".aiff")
LOW_PASS_ORDER = 4

# ============================================================================
# TRAINING MONITORING
# ============================================================================
LOG_AUDIO_INTERVAL = 20
LOG_METRICS_INTERVAL = 10
LOG_BATCH_INTERVAL = 0
STFT_LOSS_SKIP_INTERVAL = 5

# ============================================================================
# SPECTRAL LOSS SETTINGS - FAST MODE
# ============================================================================
STFT_MODE = "fast"

# Quality mode (HD Resolution for Phase 7 + Transient Micro-STFT)
STFT_QUALITY_RESOLUTIONS = [128, 256, 512, 1024, 2048, 4096]
STFT_QUALITY_HOPS = [32, 64, 128, 256, 512, 1024]
STFT_QUALITY_WIN_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

# Fast mode (Legacy mapped to HD)
STFT_FAST_RESOLUTIONS = [128, 256, 512, 1024, 2048, 4096]
STFT_FAST_HOPS = [32, 64, 128, 256, 512, 1024]
STFT_FAST_WIN_LENGTHS = [128, 256, 512, 1024, 2048, 4096]

# ============================================================================
# SKIP GENERATION TRAINING
# ============================================================================
SKIP_DROPOUT_SCHEDULE = {
    "phase1_epochs": 10,  # Sync with swap
    "phase1_dropout": 0.0,
    "phase2_epochs": 100,
    "phase2_dropout": 0.05,
    "phase3_dropout": 0.1,
}

SKIP_SWAP_SCHEDULE = {
    "phase1_epochs": 10,  # Start forcing honesty earlier
    "phase1_swap": 0.25,  # Low probability initial test
    "phase2_epochs": 200,
    "phase2_swap": 0.4,
    "phase3_swap": 0.5,
}


# ============================================================================
# VALIDATION LIMITS
# ============================================================================
MAX_LATENT_DIM = 1024
MIN_LATENT_DIM = 16
MAX_DURATION = 2.0
MIN_DURATION = 0.1
MAX_SAMPLE_RATE = 192000
MIN_SAMPLE_RATE = 8000

# ============================================================================
# AUDIO LIBRARIES AVAILABILITY
# ============================================================================
try:
    import librosa
    import soundfile as sf

    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

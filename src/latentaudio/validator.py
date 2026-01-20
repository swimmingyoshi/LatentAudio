# validator.py - Input validation utilities for LatentAudio
"""Input validation utilities for LatentAudio."""

import numpy as np
from .types import ValidationResult, AudioArray, LatentVector
from .config import (
    MAX_LATENT_DIM, MIN_LATENT_DIM, MAX_DURATION, MIN_DURATION,
    MAX_SAMPLE_RATE, MIN_SAMPLE_RATE, LATENT_DIM, WAVEFORM_LENGTH,
    SUPPORTED_AUDIO_EXTENSIONS
)

def validate_latent_vector(
    vector: np.ndarray,
    expected_dim: int = LATENT_DIM,
    name: str = "latent vector"
) -> ValidationResult:
    """Validate a latent vector.

    Args:
        vector: The latent vector to validate
        expected_dim: Expected dimensionality
        name: Name for error messages

    Returns:
        ValidationResult with any errors or warnings
    """
    result = ValidationResult(is_valid=True)

    if not isinstance(vector, np.ndarray):
        result.add_error(f"{name} must be a numpy array, got {type(vector)}")
        return result

    if vector.dtype not in [np.float32, np.float64]:
        result.add_warning(f"{name} should be float32/float64, got {vector.dtype}")

    if vector.ndim not in [1, 2]:
        result.add_error(f"{name} must be 1D or 2D, got {vector.ndim}D")

    if vector.ndim == 1 and vector.shape[0] != expected_dim:
        result.add_error(f"1D {name} must have {expected_dim} dimensions, got {vector.shape[0]}")

    if vector.ndim == 2 and vector.shape[1] != expected_dim:
        result.add_error(f"2D {name} must have {expected_dim} dimensions in last axis, got {vector.shape[1]}")

    # Check for NaN/inf values
    if not np.isfinite(vector).all():
        result.add_error(f"{name} contains NaN or infinite values")

    return result

def validate_audio_array(
    audio: np.ndarray,
    expected_length: int = WAVEFORM_LENGTH,
    name: str = "audio array"
) -> ValidationResult:
    """Validate an audio array.

    Args:
        audio: The audio array to validate
        expected_length: Expected length in samples
        name: Name for error messages

    Returns:
        ValidationResult with any errors or warnings
    """
    result = ValidationResult(is_valid=True)

    if not isinstance(audio, np.ndarray):
        result.add_error(f"{name} must be a numpy array, got {type(audio)}")
        return result

    if audio.dtype not in [np.float32, np.float64, np.int16, np.int32]:
        result.add_warning(f"{name} should be float32/float64 or int16/int32, got {audio.dtype}")

    if audio.ndim not in [1, 2]:
        result.add_error(f"{name} must be 1D or 2D, got {audio.ndim}D")

    if audio.ndim == 1 and audio.shape[0] != expected_length:
        result.add_error(f"1D {name} must have {expected_length} samples, got {audio.shape[0]}")

    if audio.ndim == 2:
        if audio.shape[1] != expected_length:
            result.add_error(f"2D {name} must have {expected_length} samples per channel, got {audio.shape[1]}")

    # Check for NaN/inf values
    if not np.isfinite(audio).all():
        result.add_error(f"{name} contains NaN or infinite values")

    # Check amplitude range
    if audio.size > 0:
        max_val = np.max(np.abs(audio))
        if max_val > 10.0:  # Very loud, likely not normalized
            result.add_warning(f"{name} has very high amplitude ({max_val:.1f}), may need normalization")

    return result

def validate_sample_rate(sample_rate: int) -> ValidationResult:
    """Validate sample rate.

    Args:
        sample_rate: Sample rate in Hz

    Returns:
        ValidationResult with any errors
    """
    result = ValidationResult(is_valid=True)

    if not isinstance(sample_rate, int):
        result.add_error(f"Sample rate must be an integer, got {type(sample_rate)}")

    if sample_rate < MIN_SAMPLE_RATE or sample_rate > MAX_SAMPLE_RATE:
        result.add_error(f"Sample rate must be between {MIN_SAMPLE_RATE} and {MAX_SAMPLE_RATE} Hz, got {sample_rate}")

    return result

def validate_duration(duration: float) -> ValidationResult:
    """Validate duration.

    Args:
        duration: Duration in seconds

    Returns:
        ValidationResult with any errors
    """
    result = ValidationResult(is_valid=True)

    if not isinstance(duration, (int, float)):
        result.add_error(f"Duration must be a number, got {type(duration)}")

    if duration < MIN_DURATION or duration > MAX_DURATION:
        result.add_error(f"Duration must be between {MIN_DURATION} and {MAX_DURATION} seconds, got {duration}")

    return result

def validate_latent_dim(latent_dim: int) -> ValidationResult:
    """Validate latent dimension.

    Args:
        latent_dim: Number of latent dimensions

    Returns:
        ValidationResult with any errors
    """
    result = ValidationResult(is_valid=True)

    if not isinstance(latent_dim, int):
        result.add_error(f"Latent dimension must be an integer, got {type(latent_dim)}")

    if latent_dim < MIN_LATENT_DIM or latent_dim > MAX_LATENT_DIM:
        result.add_error(f"Latent dimension must be between {MIN_LATENT_DIM} and {MAX_LATENT_DIM}, got {latent_dim}")

    return result

def validate_file_extension(filepath: str) -> ValidationResult:
    """Validate audio file extension.

    Args:
        filepath: Path to audio file

    Returns:
        ValidationResult with any errors
    """
    result = ValidationResult(is_valid=True)

    import os
    _, ext = os.path.splitext(filepath.lower())

    if ext not in SUPPORTED_AUDIO_EXTENSIONS:
        result.add_error(f"Unsupported file extension '{ext}'. Supported: {SUPPORTED_AUDIO_EXTENSIONS}")

    return result
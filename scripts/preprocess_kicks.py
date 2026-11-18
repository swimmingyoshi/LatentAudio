#!/usr/bin/env python3
"""
Kick Drum Preprocessor for LatentAudio

Normalizes kick drum samples to exactly 1.0 seconds for consistent VAE training.
Handles padding short files and trimming long files.

Usage:
    python preprocess_kicks.py --input /path/to/kicks --output /path/to/processed
"""

import argparse
from pathlib import Path
import warnings
import numpy as np

# Try to import audio libraries
try:
    import soundfile as sf
    import librosa
    LIBS_AVAILABLE = True
except ImportError:
    print("Error: soundfile and librosa required. Install with: pip install soundfile librosa")
    LIBS_AVAILABLE = False

warnings.filterwarnings("ignore")

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')
TARGET_DURATION = 1.0
TARGET_SAMPLE_RATE = 44100
FADE_DURATION = 0.01  # 10ms fade in/out to avoid clicks


def load_audio(filepath: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(str(filepath))
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def normalize_length(audio: np.ndarray, sr: int, target_duration: float = TARGET_DURATION) -> np.ndarray:
    """Normalize audio to target duration by padding or trimming."""
    # Handle multi-channel audio
    if audio.ndim == 1:
        # Mono
        audio = audio[:, np.newaxis]  # Add channel dimension
        was_mono = True
    else:
        was_mono = False

    target_length = int(sr * target_duration)
    current_length = audio.shape[0]

    if current_length < target_length:
        # Pad with silence
        pad_length = target_length - current_length
        # Add fade out at end of original audio to avoid clicks
        if current_length > int(sr * FADE_DURATION):
            fade_samples = int(sr * FADE_DURATION)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            # Apply fade to all channels
            audio[-fade_samples:, :] *= fade_out[:, np.newaxis]

        # Pad with zeros
        audio = np.pad(audio, ((0, pad_length), (0, 0)), mode='constant')

    elif current_length > target_length:
        # Trim to target length
        audio = audio[:target_length, :]

    # Convert back to mono if originally mono
    if was_mono:
        audio = audio.squeeze(axis=1)

    return audio


def save_audio(audio: np.ndarray, filepath: Path, sr: int = TARGET_SAMPLE_RATE) -> None:
    """Save audio array to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(filepath), audio, sr)


def process_file(input_file: Path, output_file: Path) -> dict:
    """Process a single audio file."""
    try:
        audio, sr = load_audio(input_file, TARGET_SAMPLE_RATE)
        original_duration = len(audio) / sr

        # Normalize length
        processed_audio = normalize_length(audio, sr)
        processed_duration = len(processed_audio) / sr

        # Save processed file
        save_audio(processed_audio, output_file, sr)

        return {
            'filename': input_file.name,
            'original_duration': original_duration,
            'processed_duration': processed_duration,
            'action': 'padded' if original_duration < TARGET_DURATION else 'trimmed' if original_duration > TARGET_DURATION else 'unchanged',
            'error': None
        }
    except Exception as e:
        return {
            'filename': input_file.name,
            'error': str(e)
        }


def preprocess_kicks(input_dir: Path, output_dir: Path) -> dict:
    """Preprocess all kick samples in directory."""
    if not LIBS_AVAILABLE:
        return {'error': 'Audio libraries not available'}

    if not input_dir.exists():
        return {'error': f'Input directory not found: {input_dir}'}

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_files': 0,
        'processed_files': 0,
        'padded_files': 0,
        'trimmed_files': 0,
        'unchanged_files': 0,
        'errors': [],
        'file_details': []
    }

    for audio_file in sorted(input_dir.rglob('*')):
        if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
            stats['total_files'] += 1

            # Create output path maintaining relative structure
            relative_path = audio_file.relative_to(input_dir)
            output_file = output_dir / relative_path

            result = process_file(audio_file, output_file)

            if result.get('error'):
                stats['errors'].append(f"{result['filename']}: {result['error']}")
            else:
                stats['processed_files'] += 1
                stats['file_details'].append(result)

                if result['action'] == 'padded':
                    stats['padded_files'] += 1
                elif result['action'] == 'trimmed':
                    stats['trimmed_files'] += 1
                else:
                    stats['unchanged_files'] += 1

    return stats


def print_stats(stats: dict) -> None:
    """Print preprocessing statistics."""
    if stats.get('error'):
        print(f"Error: {stats['error']}")
        return

    print("\n" + "=" * 70)
    print("KICK DRUM PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files found:       {stats['total_files']}")
    print(f"Successfully processed:  {stats['processed_files']}")
    print(f"Files padded (<1s):      {stats['padded_files']}")
    print(f"Files trimmed (>1s):     {stats['trimmed_files']}")
    print(f"Files unchanged (=1s):   {stats['unchanged_files']}")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    print(f"\nAll processed files are now exactly {TARGET_DURATION} seconds.")
    print("Use the processed directory for training to avoid length-related artifacts.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess kick drum samples to normalize lengths to 1.0 seconds'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to directory containing kick samples'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output directory for processed files'
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Kick Drum Preprocessor")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target Duration: {TARGET_DURATION}s")
    print(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  Fade Duration: {FADE_DURATION}s")

    stats = preprocess_kicks(input_path, output_path)
    print_stats(stats)


if __name__ == '__main__':
    main()
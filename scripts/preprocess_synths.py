#!/usr/bin/env python3
"""
Synth Preprocessor for LatentAudio

Extracts the first 1.0 second from synth samples for consistent VAE training.
Unlike kick drums, synths are trimmed to capture the attack/initial portion only.

Usage:
    python preprocess_synths.py --input /path/to/synths --output /path/to/processed
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
FADE_DURATION = 0.01  # 10ms fade out to avoid clicks


def load_audio(filepath: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(str(filepath))
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def extract_first_second(audio: np.ndarray, sr: int, target_duration: float = TARGET_DURATION) -> np.ndarray:
    """Extract first second of audio with fade out."""
    # Handle multi-channel audio
    if audio.ndim == 1:
        # Mono
        audio = audio[:, np.newaxis]  # Add channel dimension
        was_mono = True
    else:
        was_mono = False

    target_length = int(sr * target_duration)
    current_length = audio.shape[0]

    # Extract first second (or pad if shorter)
    if current_length >= target_length:
        audio = audio[:target_length, :]
    else:
        # If shorter than 1 second, pad with silence
        pad_length = target_length - current_length
        audio = np.pad(audio, ((0, pad_length), (0, 0)), mode='constant')

    # Apply fade out at the end to avoid clicks
    fade_samples = int(sr * FADE_DURATION)
    if audio.shape[0] >= fade_samples:
        fade_out = np.linspace(1.0, 0.0, fade_samples)
        audio[-fade_samples:, :] *= fade_out[:, np.newaxis]

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

        # Extract first second
        processed_audio = extract_first_second(audio, sr)
        processed_duration = len(processed_audio) / sr

        # Save processed file
        save_audio(processed_audio, output_file, sr)

        return {
            'filename': input_file.name,
            'original_duration': original_duration,
            'processed_duration': processed_duration,
            'action': 'extracted' if original_duration >= TARGET_DURATION else 'padded',
            'error': None
        }
    except Exception as e:
        return {
            'filename': input_file.name,
            'error': str(e)
        }


def preprocess_synths(input_dir: Path, output_dir: Path) -> dict:
    """Preprocess all synth samples in directory."""
    if not LIBS_AVAILABLE:
        return {'error': 'Audio libraries not available'}

    if not input_dir.exists():
        return {'error': f'Input directory not found: {input_dir}'}

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_files': 0,
        'processed_files': 0,
        'extracted_files': 0,
        'padded_files': 0,
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

                if result['action'] == 'extracted':
                    stats['extracted_files'] += 1
                else:
                    stats['padded_files'] += 1

    return stats


def print_stats(stats: dict) -> None:
    """Print preprocessing statistics."""
    if stats.get('error'):
        print(f"Error: {stats['error']}")
        return

    print("\n" + "=" * 70)
    print("SYNTH PREPROCESSING COMPLETE")
    print("=" * 70)
    print(f"Total files found:       {stats['total_files']}")
    print(f"Successfully processed:  {stats['processed_files']}")
    print(f"Files extracted (≥1s):   {stats['extracted_files']}")
    print(f"Files padded (<1s):      {stats['padded_files']}")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    print(f"\nAll processed files are now exactly {TARGET_DURATION} seconds.")
    print("Each file contains the first second (attack/initial portion) of the original.")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess synth samples to extract first 1.0 second'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to directory containing synth samples'
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

    print(f"Synth Preprocessor")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Target Duration: {TARGET_DURATION}s (first second only)")
    print(f"  Sample Rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  Fade Duration: {FADE_DURATION}s")

    stats = preprocess_synths(input_path, output_path)
    print_stats(stats)


if __name__ == '__main__':
    main()
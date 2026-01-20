#!/usr/bin/env python3
"""
Audio Analysis Script for LatentAudio

Analyzes audio files for length consistency and silence detection.
Designed to check kick drum samples or other datasets before training.

Usage:
    python analyze_audio.py --input /path/to/audio/files
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
TARGET_DURATION = 1.0  # Expected duration in seconds
TARGET_SAMPLE_RATE = 44100
SILENCE_THRESHOLD_DB = -40.0  # Less strict than ESC-50's -60dB for drums
MIN_SILENCE_DURATION = 0.1  # Minimum duration to consider as "long silence" in seconds


def load_audio(filepath: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(str(filepath))
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def get_duration(audio: np.ndarray, sr: int) -> float:
    """Get audio duration in seconds."""
    return len(audio) / sr


def is_silent_segment(segment: np.ndarray, threshold_db: float = SILENCE_THRESHOLD_DB) -> bool:
    """Check if audio segment is silent based on RMS energy."""
    rms = np.sqrt(np.mean(segment ** 2))
    if rms <= 0:
        return True
    rms_db = 20 * np.log10(rms)
    return rms_db < threshold_db


def detect_long_silences(audio: np.ndarray, sr: int, threshold_db: float = SILENCE_THRESHOLD_DB,
                        min_duration: float = MIN_SILENCE_DURATION) -> list[tuple[float, float]]:
    """Detect long silent periods in audio.

    Returns list of (start_time, end_time) tuples for silences >= min_duration.
    """
    # Use a sliding window to detect silences
    window_size = int(sr * 0.01)  # 10ms windows for fine-grained detection
    hop_size = window_size // 2

    silences = []
    current_silence_start = None

    for i in range(0, len(audio) - window_size + 1, hop_size):
        window = audio[i:i + window_size]
        if is_silent_segment(window, threshold_db):
            if current_silence_start is None:
                current_silence_start = i / sr
        else:
            if current_silence_start is not None:
                silence_end = (i + window_size) / sr
                silence_duration = silence_end - current_silence_start
                if silence_duration >= min_duration:
                    silences.append((current_silence_start, silence_end))
                current_silence_start = None

    # Handle silence at the end
    if current_silence_start is not None:
        silence_end = len(audio) / sr
        silence_duration = silence_end - current_silence_start
        if silence_duration >= min_duration:
            silences.append((current_silence_start, silence_end))

    return silences


def analyze_audio_file(filepath: Path) -> dict:
    """Analyze a single audio file."""
    try:
        audio, sr = load_audio(filepath, TARGET_SAMPLE_RATE)
        duration = get_duration(audio, sr)
        silences = detect_long_silences(audio, sr)

        # Calculate total silence duration
        total_silence_duration = sum(end - start for start, end in silences)

        return {
            'filename': filepath.name,
            'duration': duration,
            'sample_rate': sr,
            'is_correct_length': abs(duration - TARGET_DURATION) < 0.01,
            'silences': silences,
            'total_silence_duration': total_silence_duration,
            'silence_percentage': (total_silence_duration / duration) * 100 if duration > 0 else 0,
            'error': None
        }
    except Exception as e:
        return {
            'filename': filepath.name,
            'error': str(e)
        }


def analyze_directory(input_dir: Path) -> dict:
    """Analyze all audio files in directory."""
    if not LIBS_AVAILABLE:
        return {'error': 'Audio libraries not available'}

    if not input_dir.exists():
        return {'error': f'Input directory not found: {input_dir}'}

    stats = {
        'total_files': 0,
        'valid_files': 0,
        'durations': [],
        'correct_length_files': 0,
        'files_with_long_silences': 0,
        'errors': [],
        'file_details': []
    }

    for audio_file in sorted(input_dir.rglob('*')):
        if audio_file.suffix.lower() in AUDIO_EXTENSIONS:
            stats['total_files'] += 1
            result = analyze_audio_file(audio_file)

            if result.get('error'):
                stats['errors'].append(f"{result['filename']}: {result['error']}")
            else:
                stats['valid_files'] += 1
                stats['durations'].append(result['duration'])
                stats['file_details'].append(result)

                if result['is_correct_length']:
                    stats['correct_length_files'] += 1

                if result['silences']:
                    stats['files_with_long_silences'] += 1

    return stats


def print_analysis(stats: dict) -> None:
    """Print analysis results."""
    if stats.get('error'):
        print(f"Error: {stats['error']}")
        return

    print("\n" + "=" * 70)
    print("AUDIO ANALYSIS REPORT")
    print("=" * 70)
    print(f"Total files found:     {stats['total_files']}")
    print(f"Valid audio files:     {stats['valid_files']}")

    if stats['valid_files'] == 0:
        print("No valid audio files to analyze.")
        return

    durations = stats['durations']
    print("\nDuration Statistics:")
    print(f"  Min duration: {min(durations):.3f}s")
    print(f"  Max duration: {max(durations):.3f}s")
    print(f"  Avg duration: {np.mean(durations):.3f}s")
    print(f"  Files at target {TARGET_DURATION}s: {stats['correct_length_files']}/{stats['valid_files']} ({stats['correct_length_files']/stats['valid_files']*100:.1f}%)")

    print("\nSilence Analysis:")
    print(f"  Files with long silences: {stats['files_with_long_silences']}/{stats['valid_files']} ({stats['files_with_long_silences']/stats['valid_files']*100:.1f}%)")

    # Show details for files with issues
    print("\nFiles with length issues (not 1.0s):")
    length_issues = [f for f in stats['file_details'] if not f['is_correct_length']]
    if length_issues:
        for file_info in length_issues[:10]:  # Show first 10
            print(f"  - {file_info['filename']}: {file_info['duration']:.3f}s")
        if len(length_issues) > 10:
            print(f"  ... and {len(length_issues) - 10} more")
    else:
        print("  None - all files are correct length!")

    print("\nFiles with long silences:")
    silence_issues = [f for f in stats['file_details'] if f['silences']]
    if silence_issues:
        for file_info in silence_issues[:10]:  # Show first 10
            silence_count = len(file_info['silences'])
            silence_pct = file_info['silence_percentage']
            print(f"  - {file_info['filename']}: {silence_count} silences ({silence_pct:.1f}%)")
        if len(silence_issues) > 10:
            print(f"  ... and {len(silence_issues) - 10} more")
    else:
        print("  None - no long silences detected!")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"  - {error}")
        if len(stats['errors']) > 5:
            print(f"  ... and {len(stats['errors']) - 5} more")

    print("\nRecommendations:")
    if stats['correct_length_files'] < stats['valid_files']:
        print("- Preprocess files to ensure all are exactly 1.0 seconds")
        print("  - Pad short files with fade-in/out to avoid clicks")
        print("  - Trim long files at zero crossings")
    if stats['files_with_long_silences'] > 0:
        print("- Review files with long silences - may indicate recording issues")
        print("  - Consider trimming silent portions")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze audio files for length consistency and silence detection'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to directory containing audio files'
    )

    args = parser.parse_args()
    input_path = Path(args.input)

    print(f"Audio Analysis Tool")
    print(f"  Input: {input_path}")
    print(f"  Target Duration: {TARGET_DURATION}s")
    print(f"  Silence Threshold: {SILENCE_THRESHOLD_DB}dB")
    print(f"  Min Silence Duration: {MIN_SILENCE_DURATION}s")

    stats = analyze_directory(input_path)
    print_analysis(stats)


if __name__ == '__main__':
    main()
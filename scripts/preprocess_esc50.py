#!/usr/bin/env python3
"""
ESC-50 Dataset Preprocessor

Converts ESC-50 audio clips (5s) into 0.5s segments compatible with LatentAudio VAE training.
ESC-50: https://github.com/karolpiczak/ESC-50

Usage:
    python preprocess_esc50.py --input /path/to/ESC-50 --output /path/to/preprocessed
"""

import argparse
import pathlib
from pathlib import Path
import shutil
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')
TARGET_DURATION = 0.5
TARGET_SAMPLE_RATE = 44100
SEGMENTS_PER_CLIP = 10

CATEGORIES = {
    0: 'dog',
    1: 'rooster',
    2: 'pig',
    3: 'cow',
    4: 'frog',
    5: 'cat',
    6: 'hen',
    7: 'insects',
    8: 'sheep',
    9: 'crow',
    10: 'rain',
    11: 'sea_waves',
    12: 'crackling_fire',
    13: 'crickets',
    14: 'chirping_birds',
    15: 'water_drops',
    16: 'wind',
    17: 'pouring_water',
    18: 'toilet_flush',
    19: 'thunderstorm',
    20: 'crying_baby',
    21: 'sneezing',
    22: 'clapping',
    23: 'breathing',
    24: 'coughing',
    25: 'footsteps',
    26: 'laughing',
    27: 'brushing_teeth',
    28: 'snoring',
    29: 'drinking_sipping',
    30: 'door_wood_knock',
    31: 'mouse_click',
    32: 'keyboard_typing',
    33: 'door_wood_creaks',
    34: 'can_opening',
    35: 'washing_machine',
    36: 'vacuum_cleaner',
    37: 'clock_alarm',
    38: 'clock_tick',
    39: 'glass_breaking',
    40: 'helicopter',
    41: 'chainsaw',
    42: 'siren',
    43: 'car_horn',
    44: 'engine',
    45: 'train',
    46: 'church_bells',
    47: 'airplane',
    48: 'fireworks',
    49: 'hand_saw',
}


def parse_esc50_filename(filename: str) -> dict:
    """Parse ESC-50 filename to extract metadata.

    Filename format: {fold}-{id}-{recording_letter}-{target}.wav (ESC-50-master)
    Example: 1-100032-A-0.wav

    The recording letter (A, B, etc.) is just an identifier for multiple
    recordings from the same source, NOT the category. The target number
    (0-49) determines the actual category.
    """
    name = Path(filename).stem
    parts = name.split('-')

    if len(parts) == 4:
        return {
            'fold': int(parts[0]),
            'id': parts[1],
            'recording_letter': parts[2],
            'target': int(parts[3]),
        }
    else:
        raise ValueError(f"Invalid ESC-50 filename format: {filename}")


def load_audio(filepath: Path, target_sr: int = TARGET_SAMPLE_RATE) -> tuple[np.ndarray, int]:
    """Load audio file and resample to target sample rate."""
    audio, sr = sf.read(str(filepath))
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return audio, sr


def split_into_segments(
    audio: np.ndarray,
    sr: int,
    segment_duration: float = TARGET_DURATION,
    segments_per_clip: int = SEGMENTS_PER_CLIP
) -> list[np.ndarray]:
    """Split audio into equal segments."""
    segment_length = int(sr * segment_duration)
    total_length = segment_length * segments_per_clip

    if len(audio) < total_length:
        pad_length = total_length - len(audio)
        audio = np.pad(audio, (0, pad_length), mode='constant')
    elif len(audio) > total_length:
        audio = audio[:total_length]

    segments = []
    for i in range(segments_per_clip):
        start = i * segment_length
        end = start + segment_length
        segment = audio[start:end]
        segments.append(segment)

    return segments


def is_silent(segment: np.ndarray, threshold_db: float = -60.0) -> bool:
    """Check if audio segment is silent based on RMS energy.

    Args:
        segment: Audio segment array
        threshold_db: Silence threshold in dB (default -60dB)

    Returns:
        True if segment is below threshold (silent)
    """
    # Calculate RMS energy
    rms = np.sqrt(np.mean(segment ** 2))

    # Convert to dB (avoid log(0))
    if rms <= 0:
        return True

    rms_db = 20 * np.log10(rms)

    return rms_db < threshold_db


def save_segment(segment: np.ndarray, filepath: Path, sr: int = TARGET_SAMPLE_RATE) -> None:
    """Save audio segment to WAV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(filepath), segment, sr)


def preprocess_esc50(
    input_dir: Path,
    output_dir: Path,
    fold: int | None = None,
    target_sr: int = TARGET_SAMPLE_RATE,
    silence_threshold_db: float = -60.0
) -> dict:
    """Preprocess ESC-50 dataset.

    Args:
        input_dir: Path to ESC-50 audio directory
        output_dir: Path to output directory for preprocessed files
        fold: Optional fold number (1-5) to process only specific fold
        target_sr: Target sample rate
        silence_threshold_db: Silence threshold in dB (default -60dB)

    Returns:
        Dictionary with preprocessing statistics
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    stats = {
        'total_clips': 0,
        'total_segments': 0,
        'silent_segments_skipped': 0,
        'categories_processed': set(),
        'folds_processed': set(),
        'errors': []
    }

    for audio_file in sorted(input_dir.glob('*.wav')):
        try:
            metadata = parse_esc50_filename(audio_file.name)

            if fold is not None and metadata['fold'] != fold:
                continue

            stats['total_clips'] += 1
            stats['folds_processed'].add(metadata['fold'])
            stats['categories_processed'].add(metadata['target'])

            audio, sr = load_audio(audio_file, target_sr)

            segments = split_into_segments(audio, sr)

            category_name = CATEGORIES.get(metadata['target'], f"cat_{metadata['target']}")
            fold_num = metadata['fold']
            clip_id = metadata['id']
            recording_letter = metadata['recording_letter']

            for seg_idx, segment in enumerate(segments):
                # Skip silent segments
                if is_silent(segment, silence_threshold_db):
                    stats['silent_segments_skipped'] += 1
                    continue

                segment_filename = f"{category_name}_fold{fold_num}_clip{clip_id}_{recording_letter}_seg{seg_idx:02d}.wav"
                segment_path = output_dir / category_name / segment_filename

                save_segment(segment, segment_path, sr)
                stats['total_segments'] += 1

        except Exception as e:
            stats['errors'].append(f"{audio_file.name}: {e}")

    stats['categories_processed'] = sorted(stats['categories_processed'])
    stats['folds_processed'] = sorted(stats['folds_processed'])

    return stats


def print_stats(stats: dict) -> None:
    """Print preprocessing statistics."""
    print("\n" + "=" * 60)
    print("ESC-50 Preprocessing Complete")
    print("=" * 60)
    print(f"  Total clips processed:    {stats['total_clips']}")
    print(f"  Total segments created:   {stats['total_segments']}")
    print(f"  Silent segments skipped:  {stats['silent_segments_skipped']}")
    print(f"  Segments per clip:        {SEGMENTS_PER_CLIP}")
    print(f"  Categories covered:       {len(stats['categories_processed'])}")
    print(f"  Folds processed:          {stats['folds_processed']}")

    if stats['errors']:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"    - {error}")
        if len(stats['errors']) > 5:
            print(f"    ... and {len(stats['errors']) - 5} more")

    print("\n  Preprocessing ratio: 1 clip -> {0} segments".format(SEGMENTS_PER_CLIP))
    print(f"  Expected audio duration: {stats['total_segments'] * TARGET_DURATION:.1f}s")
    print("=" * 60)


def verify_output(output_dir: Path, expected_segments: int) -> bool:
    """Verify output directory contains expected number of segments."""
    if not output_dir.exists():
        print(f"Output directory not found: {output_dir}")
        return False

    total_files = sum(1 for _ in output_dir.rglob('*.wav'))
    if total_files >= expected_segments * 0.9:
        print(f"\n✓ Output verified: {total_files} audio segments found")
        return True
    else:
        print(f"\n✗ Output mismatch: expected ~{expected_segments}, found {total_files}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess ESC-50 dataset for LatentAudio VAE training'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to ESC-50 audio directory'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Path to output directory for preprocessed files'
    )
    parser.add_argument(
        '--fold', '-f',
        type=int,
        default=None,
        choices=range(1, 6),
        help='Process only specific fold (1-5)'
    )
    parser.add_argument(
        '--sr', '-s',
        type=int,
        default=TARGET_SAMPLE_RATE,
        help=f'Target sample rate (default: {TARGET_SAMPLE_RATE})'
    )
    parser.add_argument(
        '--silence-threshold',
        type=float,
        default=-60.0,
        help='Silence threshold in dB (default: -60.0). Segments below this threshold are skipped.'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify output after preprocessing'
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"ESC-50 Preprocessor")
    print(f"  Input:  {input_path}")
    print(f"  Output: {output_path}")
    if args.fold:
        print(f"  Fold:   {args.fold} only")
    print(f"  Sample Rate: {args.sr} Hz")
    print(f"  Segment Duration: {TARGET_DURATION}s")
    print(f"  Segments per Clip: {SEGMENTS_PER_CLIP}")
    print(f"  Silence Threshold: {args.silence_threshold}dB")
    print()

    stats = preprocess_esc50(input_path, output_path, args.fold, args.sr, args.silence_threshold)
    print_stats(stats)

    if args.verify:
        verify_output(output_path, stats['total_segments'])


if __name__ == '__main__':
    main()

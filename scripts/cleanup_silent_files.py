#!/usr/bin/env python3
"""
Silent File Cleanup Script

Removes silent audio files from preprocessed ESC-50 dataset.
Useful for cleaning up already-preprocessed data that contains silent segments.

Usage:
    python cleanup_silent_files.py --input /path/to/preprocessed/data --threshold -60.0
"""

import argparse
import pathlib
from pathlib import Path
import shutil

import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')


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


def cleanup_silent_files(input_dir: Path, threshold_db: float = -60.0, dry_run: bool = False) -> dict:
    """Clean up silent files from preprocessed dataset.

    Args:
        input_dir: Directory containing preprocessed audio files
        threshold_db: Silence threshold in dB
        dry_run: If True, only report what would be deleted

    Returns:
        Dictionary with cleanup statistics
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    stats = {
        'files_checked': 0,
        'silent_files_found': 0,
        'silent_files_removed': 0,
        'total_files_processed': 0,
        'errors': []
    }

    # Find all audio files recursively
    audio_files = []
    for ext in SUPPORTED_EXTENSIONS:
        audio_files.extend(input_dir.rglob(f'*{ext}'))

    print(f"Found {len(audio_files)} audio files to check")

    for audio_file in sorted(audio_files):
        try:
            stats['files_checked'] += 1

            # Load audio
            audio, _ = sf.read(str(audio_file))

            # Check if silent
            if is_silent(audio, threshold_db):
                stats['silent_files_found'] += 1

                if dry_run:
                    print(f"Would remove: {audio_file}")
                else:
                    try:
                        audio_file.unlink()  # Remove the file
                        stats['silent_files_removed'] += 1
                        print(f"Removed: {audio_file}")
                    except Exception as e:
                        stats['errors'].append(f"Failed to remove {audio_file}: {e}")
            else:
                stats['total_files_processed'] += 1

        except Exception as e:
            stats['errors'].append(f"Error processing {audio_file}: {e}")

    return stats


def print_cleanup_stats(stats: dict, dry_run: bool = False) -> None:
    """Print cleanup statistics."""
    print("\n" + "=" * 60)
    if dry_run:
        print("DRY RUN - Silent File Cleanup Report")
    else:
        print("Silent File Cleanup Complete")
    print("=" * 60)
    print(f"  Files checked:           {stats['files_checked']}")
    print(f"  Silent files found:      {stats['silent_files_found']}")
    if not dry_run:
        print(f"  Silent files removed:    {stats['silent_files_removed']}")
        print(f"  Files kept:              {stats['total_files_processed']}")
    else:
        print(f"  Files that would be kept: {stats['total_files_processed']}")

    if stats['errors']:
        print(f"\n  Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:5]:
            print(f"    - {error}")
        if len(stats['errors']) > 5:
            print(f"    ... and {len(stats['errors']) - 5} more")

    success_rate = (stats['files_checked'] - len(stats['errors'])) / max(stats['files_checked'], 1) * 100
    print(f"\n  Success rate: {success_rate:.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Clean up silent audio files from preprocessed ESC-50 dataset'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Path to preprocessed audio directory'
    )
    parser.add_argument(
        '--threshold', '-t',
        type=float,
        default=-60.0,
        help='Silence threshold in dB (default: -60.0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )

    args = parser.parse_args()

    input_path = Path(args.input)

    print("Silent File Cleanup")
    print(f"  Input:  {input_path}")
    print(f"  Threshold: {args.threshold}dB")
    print(f"  Dry run: {args.dry_run}")
    print()

    stats = cleanup_silent_files(input_path, args.threshold, args.dry_run)
    print_cleanup_stats(stats, args.dry_run)


if __name__ == '__main__':
    main()
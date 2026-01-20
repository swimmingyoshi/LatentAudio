#!/usr/bin/env python3
"""
Diagnostic script to check training data for NaN/Inf and normalization issues.
Run this before training to catch data problems early.
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import sys

def check_audio_file(filepath, sample_rate=44100):
    """Check a single audio file for issues."""
    issues = []
    
    try:
        # Load audio
        audio, sr = librosa.load(str(filepath), sr=sample_rate, mono=True)
        
        # Check for NaN
        if np.isnan(audio).any():
            issues.append(f"❌ CRITICAL: Contains NaN values")
        
        # Check for Inf
        if np.isinf(audio).any():
            issues.append(f"❌ CRITICAL: Contains Inf values")
        
        # Check range
        audio_min = audio.min()
        audio_max = audio.max()
        audio_range = audio_max - audio_min
        
        if audio_range == 0:
            issues.append(f"⚠️  WARNING: Silent file (all zeros)")
        elif abs(audio_max) > 10.0 or abs(audio_min) > 10.0:
            issues.append(f"⚠️  WARNING: Unnormalized (range: [{audio_min:.2f}, {audio_max:.2f}])")
        
        # Check for clipping
        clipped = (np.abs(audio) > 0.99).sum()
        if clipped > len(audio) * 0.01:  # More than 1% clipped
            issues.append(f"⚠️  WARNING: {clipped} samples clipped ({100*clipped/len(audio):.1f}%)")
        
        # Check DC offset
        dc_offset = np.mean(audio)
        if abs(dc_offset) > 0.1:
            issues.append(f"⚠️  WARNING: DC offset detected ({dc_offset:.3f})")
        
        # Return stats
        stats = {
            'filepath': filepath.name,
            'samples': len(audio),
            'duration': len(audio) / sr,
            'min': audio_min,
            'max': audio_max,
            'mean': dc_offset,
            'std': np.std(audio),
            'issues': issues
        }
        
        return stats, len(issues) > 0
        
    except Exception as e:
        return {
            'filepath': filepath.name,
            'issues': [f"❌ CRITICAL: Failed to load - {str(e)}"]
        }, True


def diagnose_dataset(data_dir, sample_rate=44100):
    """Diagnose entire dataset."""
    data_dir = Path(data_dir)
    
    # Find all audio files
    patterns = ['*.wav', '*.mp3', '*.flac', '*.ogg']
    files = []
    for pattern in patterns:
        files.extend(list(data_dir.rglob(pattern)))
    
    if len(files) == 0:
        print(f"❌ ERROR: No audio files found in {data_dir}")
        return
    
    print(f"Found {len(files)} audio files")
    print(f"Checking files at {sample_rate}Hz sample rate...")
    print("=" * 80)
    
    # Check each file
    problem_files = []
    all_stats = []
    
    for filepath in tqdm(files, desc="Checking files"):
        stats, has_issues = check_audio_file(filepath, sample_rate)
        all_stats.append(stats)
        
        if has_issues:
            problem_files.append(stats)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(files)}")
    print(f"Problem files: {len(problem_files)}")
    print(f"Clean files: {len(files) - len(problem_files)}")
    
    if len(problem_files) > 0:
        print(f"\n⚠️  {len(problem_files)} FILES WITH ISSUES:")
        print("=" * 80)
        
        for stats in problem_files:
            print(f"\n{stats['filepath']}:")
            for issue in stats['issues']:
                print(f"  {issue}")
    
    # Overall statistics
    if len(all_stats) > 0:
        print("\n" + "=" * 80)
        print("DATASET STATISTICS")
        print("=" * 80)
        
        all_mins = [s['min'] for s in all_stats if 'min' in s]
        all_maxs = [s['max'] for s in all_stats if 'max' in s]
        all_means = [s['mean'] for s in all_stats if 'mean' in s]
        all_stds = [s['std'] for s in all_stats if 'std' in s]
        
        if all_mins:
            print(f"Overall min: {min(all_mins):.4f}")
            print(f"Overall max: {max(all_maxs):.4f}")
            print(f"Average mean: {np.mean(all_means):.4f}")
            print(f"Average std: {np.mean(all_stds):.4f}")
            
            # Check if normalization is needed
            if max(abs(min(all_mins)), abs(max(all_maxs))) > 2.0:
                print("\n⚠️  RECOMMENDATION: Data should be normalized to [-1, 1] range")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if len(problem_files) == 0:
        print("✅ Dataset looks good! Ready for training.")
    else:
        critical_issues = sum(1 for s in problem_files if any('CRITICAL' in i for i in s['issues']))
        if critical_issues > 0:
            print(f"❌ CRITICAL: {critical_issues} files must be fixed or removed before training")
            print("   Files with NaN/Inf will cause immediate training failure!")
        
        warnings = len(problem_files) - critical_issues
        if warnings > 0:
            print(f"⚠️  WARNING: {warnings} files have minor issues")
            print("   Training may work but quality could be affected")
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_data.py <path_to_training_data>")
        print("Example: python diagnose_data.py ./training_audio")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    sample_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 44100
    
    diagnose_dataset(data_dir, sample_rate)
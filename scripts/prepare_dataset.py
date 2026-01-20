#!/usr/bin/env python3
"""
Advanced Dataset Preparator for LatentAudio.
Implements Phase-Locked Transient Alignment, Peak Normalization, and Polarity Correction.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add src to path for internal imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import librosa
    import soundfile as sf
    LIBS_AVAILABLE = True
except ImportError:
    LIBS_AVAILABLE = False

warnings.filterwarnings("ignore")

# ============================================================================
# DSP FUNCTIONS
# ============================================================================

def align_transient(audio: np.ndarray, threshold_db: float = -60.0) -> np.ndarray:
    """Crops leading silence to align the transient to Sample 0."""
    # Convert threshold to amplitude
    threshold = 10**(threshold_db / 20)
    
    # Find first sample above threshold
    abs_audio = np.abs(audio)
    indices = np.where(abs_audio > threshold)[0]
    
    if len(indices) > 0:
        start_idx = indices[0]
        # Return audio from that point onwards
        return audio[start_idx:]
    return audio

def fix_polarity(audio: np.ndarray, search_window: int = 512) -> np.ndarray:
    """Ensures the first major transient is positive (pushing out)."""
    # Look at the first N samples
    window = audio[:min(len(audio), search_window)]
    if len(window) == 0:
        return audio
        
    # Find the absolute peak in the window
    peak_idx = np.argmax(np.abs(window))
    
    # If the peak is negative, flip the whole signal
    if window[peak_idx] < 0:
        return audio * -1.0
    return audio

def peak_normalize(audio: np.ndarray, target_db: float = -0.1) -> np.ndarray:
    """Scales audio to target peak amplitude."""
    max_amp = np.max(np.abs(audio))
    if max_amp > 0:
        target_amp = 10**(target_db / 20)
        return audio * (target_amp / max_amp)
    return audio

def standardize_length(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Pads or trims audio to exact target length."""
    if len(audio) > target_samples:
        return audio[:target_samples]
    elif len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)))
    return audio

# ============================================================================
# PROCESSING ENGINE
# ============================================================================

def process_single_file(args: tuple) -> Dict:
    """Worker function for parallel processing."""
    file_path, out_path, target_sr, duration, align, polar, norm = args
    
    try:
        # 1. Load
        audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        
        # 2. Transient Alignment
        if align:
            audio = align_transient(audio)
            
        # 3. Polarity Correction
        if polar:
            audio = fix_polarity(audio)
            
        # 4. Normalization
        if norm:
            audio = peak_normalize(audio)
            
        # 5. Length Control
        target_samples = int(target_sr * duration)
        audio = standardize_length(audio, target_samples)
        
        # 6. Save
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), audio, target_sr)
        
        return {"status": "ok", "file": file_path.name}
    except Exception as e:
        return {"status": "error", "file": file_path.name, "msg": str(e)}

def main():
    if not LIBS_AVAILABLE:
        print("Error: 'librosa' and 'soundfile' are required.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Advanced Dataset Preparator")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input folder")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output folder")
    parser.add_argument("--sr", type=int, default=44100, help="Target sample rate")
    parser.add_argument("--duration", type=float, default=1.0, help="Target duration (s)")
    parser.add_argument("--no-align", action="store_true", help="Disable transient alignment")
    parser.add_argument("--no-polar", action="store_true", help="Disable polarity correction")
    parser.add_argument("--no-norm", action="store_true", help="Disable peak normalization")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} not found.")
        sys.exit(1)

    # Gather files
    extensions = ('.wav', '.mp3', '.flac', '.ogg', '.aiff')
    all_files = [f for f in input_dir.rglob("*") if f.suffix.lower() in extensions]
    
    print(f"\n🚀 Preparing {len(all_files)} samples...")
    print(f"   Settings: {args.sr}Hz, {args.duration}s, Align={not args.no_align}, Polar={not args.no_polar}, Norm={not args.no_norm}")

    # Prepare worker arguments
    worker_tasks = []
    for f in all_files:
        rel_path = f.relative_to(input_dir)
        out_f = output_dir / rel_path
        worker_tasks.append((
            f, out_f, args.sr, args.duration, 
            not args.no_align, not args.no_polar, not args.no_norm
        ))

    # Parallel Execute
    results = []
    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(process_single_file, worker_tasks), total=len(worker_tasks), desc="Processing"):
            results.append(res)

    # Report
    errors = [r for r in results if r["status"] == "error"]
    print(f"\n✅ Done! Processed {len(results) - len(errors)} files.")
    if errors:
        print(f"⚠️ Encountered {len(errors)} errors. Check log for details.")
        for e in errors[:5]:
            print(f"   - {e['file']}: {e['msg']}")

if __name__ == "__main__":
    main()

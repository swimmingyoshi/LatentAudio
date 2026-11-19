# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dataset loading for v2: one-shots -> fixed-length, loudness-normalised float32."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np

AUDIO_EXTENSIONS = (".wav", ".flac", ".aif", ".aiff", ".mp3", ".ogg")


def find_audio_files(folder: Path) -> List[Path]:
    files = [p for p in Path(folder).rglob("*") if p.suffix.lower() in AUDIO_EXTENSIONS]
    return sorted(files, key=lambda p: str(p).lower())


def load_one_shot(path: Path, sample_rate: int, length: int, peak: float = 0.89,
                  trim_db: float = 60.0) -> np.ndarray | None:
    """Mono, resampled, leading silence trimmed, cropped/padded to `length`,
    peak-normalised. Returns None for silent/unreadable files."""
    import librosa

    try:
        # Only read what we can use (plus slack for trimming leading silence).
        audio, _ = librosa.load(str(path), sr=sample_rate, mono=True, duration=length / sample_rate + 2.0)
    except Exception:
        return None
    if audio.size == 0 or not np.isfinite(audio).all():
        return None

    # Align the onset: drop leading samples more than trim_db below the peak.
    level = np.max(np.abs(audio))
    if level < 1e-5:
        return None
    above = np.nonzero(np.abs(audio) > level * 10 ** (-trim_db / 20))[0]
    audio = audio[above[0]:]

    if audio.size >= length:
        audio = audio[:length].copy()
        fade = min(441, length)  # 10 ms fade-out where we cut the tail
        audio[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
    else:
        audio = np.pad(audio, (0, length - audio.size))

    audio *= peak / max(np.max(np.abs(audio)), 1e-6)
    return audio.astype(np.float32)


def load_folder(folder: Path, sample_rate: int = 44100, length: int = 44100,
                peak: float = 0.89) -> Tuple[np.ndarray, List[str]]:
    """Returns ([N, length] float32, labels). Labels are paths relative to `folder`."""
    from tqdm import tqdm

    folder = Path(folder)
    audio, labels = [], []
    for path in tqdm(find_audio_files(folder), desc=f"Loading {folder.name}"):
        x = load_one_shot(path, sample_rate, length, peak)
        if x is not None:
            audio.append(x)
            labels.append(str(path.relative_to(folder).with_suffix("")).replace("\\", "/"))
    if not audio:
        raise ValueError(f"No usable audio found in {folder}")
    return np.stack(audio), labels


def split_indices(n: int, val_fraction: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = int(round(n * val_fraction)) if n >= 20 else 0
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])

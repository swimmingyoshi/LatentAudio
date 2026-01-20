# widgets/__init__.py - UI widget components
"""UI widget components for LatentAudio."""

from .visualizer import WaveformVisualizer
from .latent_sliders import LatentVectorWidget
from .controls import GenerationControls, PlaybackControls, StatusWidget
from .sound_map import SoundMapWidget

__all__ = [
    "WaveformVisualizer",
    "LatentVectorWidget",
    "GenerationControls",
    "PlaybackControls",
    "StatusWidget",
    "SoundMapWidget",
]

# tabs/__init__.py - UI tab components
"""UI tab components for LatentAudio."""

from .presets import PresetManager
from .morph import MorphTab
from .walk import WalkTab
from .reconstruction import ReconstructionTab
from .variations import VariationsTab
from .attributes import AttributesTab

__all__ = [
    "PresetManager",
    "MorphTab",
    "WalkTab",
    "ReconstructionTab",
    "VariationsTab",
    "AttributesTab",
]

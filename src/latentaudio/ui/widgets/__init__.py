# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LatentAudio - Direct Neural Audio Generation and Exploration
# Copyright (C) 2024 LatentAudio Team
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
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

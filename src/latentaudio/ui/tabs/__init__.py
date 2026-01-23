# SPDX-License-Identifier: AGPL-3.0-or-later
#
# LatentAudio - Direct Neural Audio Generation and Exploration
# Copyright (C) 2026 swimmingyoshi
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

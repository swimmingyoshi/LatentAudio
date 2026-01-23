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
# attributes.py - Latent arithmetic and attribute discovery
"""Utilities for discovering and applying directions in latent space."""

import numpy as np
from typing import List, Optional, Dict, Tuple
from ..types import LatentVector, DiscoveredDirection
from ..logging import logger


class AttributeExplorer:
    """
    Handles discovery of semantic directions in latent space using arithmetic.

    This allows 'finding' directions that correspond to qualities like
    'brightness', 'noisiness', or specific instrument characteristics.
    """

    def __init__(self, latent_dim: int = 128):
        self.latent_dim = latent_dim
        self.discovered_directions: Dict[str, DiscoveredDirection] = {}

    def discover_direction(
        self, name: str, positive_samples: List[LatentVector], negative_samples: List[LatentVector]
    ) -> DiscoveredDirection:
        """
        Discover a direction by comparing two sets of samples.

        Logic: direction = mean(positive) - mean(negative)
        """
        if not positive_samples or not negative_samples:
            raise ValueError("Must provide both positive and negative samples")

        pos_mean = np.mean(positive_samples, axis=0)
        neg_mean = np.mean(negative_samples, axis=0)

        direction = pos_mean - neg_mean

        # Normalize the direction
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm

        discovered = DiscoveredDirection(
            name=name,
            direction_vector=direction,
            description=f"Direction discovered from {len(positive_samples)} pos and {len(negative_samples)} neg samples",
        )

        self.discovered_directions[name] = discovered
        logger.info(
            f"Discovered latent direction: '{name}' (magnitude preserved via normalization)"
        )
        return discovered

    def apply_attribute(
        self, base_vector: LatentVector, direction_name: str, strength: float
    ) -> LatentVector:
        """Apply a discovered direction to a base vector with a given strength."""
        if direction_name not in self.discovered_directions:
            raise ValueError(f"Direction '{direction_name}' not found")

        direction = self.discovered_directions[direction_name].direction_vector
        return base_vector + (direction * strength)

    def get_attribute_intensity(self, vector: LatentVector, direction_name: str) -> float:
        """Calculate how much of a specific attribute is present in a vector (projection)."""
        if direction_name not in self.discovered_directions:
            raise ValueError(f"Direction '{direction_name}' not found")

        direction = self.discovered_directions[direction_name].direction_vector
        # Project vector onto normalized direction
        return np.dot(vector, direction)

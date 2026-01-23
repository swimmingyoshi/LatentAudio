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
# interpolator.py - Latent space interpolation utilities
"""Latent space interpolation and morphing utilities."""

import numpy as np
from typing import List, Optional
from ..types import InterpolationConfig, LatentVector
from ..logging import logger


class LatentInterpolator:
    """
    Handles interpolation between points in latent space.

    Supports multiple interpolation methods including linear and spherical
    linear interpolation (SLERP) for better quality morphing.
    """

    def __init__(self):
        """Initialize the interpolator."""
        logger.debug("LatentInterpolator initialized")

    def interpolate(
        self, z1: LatentVector, z2: LatentVector, n_steps: int = 10, method: str = "linear"
    ) -> List[LatentVector]:
        """
        Interpolate between two latent vectors.

        Args:
            z1: Starting latent vector
            z2: Ending latent vector
            n_steps: Number of interpolation steps (including start and end)
            method: Interpolation method ('linear' or 'spherical')

        Returns:
            List of interpolated latent vectors
        """
        if method == "linear":
            return self._linear_interpolate(z1, z2, n_steps)
        elif method == "spherical":
            return self._spherical_interpolate(z1, z2, n_steps)
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

    def _linear_interpolate(
        self, z1: LatentVector, z2: LatentVector, n_steps: int
    ) -> List[LatentVector]:
        """Linear interpolation between two vectors."""
        alphas = np.linspace(0, 1, n_steps)
        return [z1 * (1 - a) + z2 * a for a in alphas]

    def _spherical_interpolate(
        self, z1: LatentVector, z2: LatentVector, n_steps: int
    ) -> List[LatentVector]:
        """
        Spherical linear interpolation (SLERP).

        Better for high-dimensional latent spaces as it preserves
        angular relationships.
        """
        # Normalize vectors
        z1_norm = z1 / (np.linalg.norm(z1) + 1e-8)
        z2_norm = z2 / (np.linalg.norm(z2) + 1e-8)

        # Compute angle between vectors
        dot = np.sum(z1_norm * z2_norm)
        dot = np.clip(dot, -1, 1)  # Handle numerical precision issues
        theta = np.arccos(dot)

        # If vectors are very close, use linear interpolation
        if abs(theta) < 1e-4:
            return self._linear_interpolate(z1, z2, n_steps)

        # Generate interpolation angles
        alphas = np.linspace(0, 1, n_steps)

        # SLERP formula: sin((1-t)θ)/sin(θ) * z1 + sin(tθ)/sin(θ) * z2
        sin_theta = np.sin(theta)
        return [(np.sin((1 - a) * theta) * z1 + np.sin(a * theta) * z2) / sin_theta for a in alphas]

    def morph_sequence(
        self, points: List[LatentVector], steps_per_segment: int = 10, method: str = "linear"
    ) -> List[LatentVector]:
        """
        Create a morphing sequence through multiple points.

        Args:
            points: List of latent vectors to morph through
            steps_per_segment: Steps between each pair of points
            method: Interpolation method

        Returns:
            Continuous morphing sequence
        """
        if len(points) < 2:
            raise ValueError("Need at least 2 points for morphing")

        sequence = []

        for i in range(len(points) - 1):
            segment = self.interpolate(
                points[i], points[i + 1], n_steps=steps_per_segment, method=method
            )

            # Avoid duplicating points between segments
            if i > 0:
                segment = segment[1:]

            sequence.extend(segment)

        logger.debug(f"Created morph sequence: {len(sequence)} steps through {len(points)} points")
        return sequence

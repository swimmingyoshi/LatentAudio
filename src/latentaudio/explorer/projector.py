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
# projector.py - Latent space projection (UMAP/t-SNE)
"""Utilities for projecting high-dimensional latent space to 2D/3D."""

import numpy as np
from typing import List, Optional, Union
from ..logging import logger

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.manifold import TSNE


class LatentProjector:
    """
    Handles projection of latent vectors into lower-dimensional space (2D).

    Used for creating interactive sound maps.
    """

    def __init__(self, latent_dim: int = 128, method: str = "auto"):
        """
        Initialize the projector.

        Args:
            latent_dim: Input dimensionality
            method: 'umap', 'tsne', or 'auto' (prefers umap)
        """
        self.latent_dim = latent_dim

        if method == "auto":
            self.method = "umap" if UMAP_AVAILABLE else "tsne"
        else:
            self.method = method

        self.reducer = None
        self.is_fitted = False
        logger.debug(f"LatentProjector initialized using {self.method}")

    def fit(self, latent_vectors: np.ndarray):
        """
        Fit the projection model to a set of latent vectors.

        Args:
            latent_vectors: Array of shape (n_samples, latent_dim)
        """
        if len(latent_vectors) < 2:
            logger.warning("Need at least 2 samples to fit projector")
            return

        logger.info(f"Fitting {self.method} projector with {len(latent_vectors)} samples...")

        if self.method == "umap" and UMAP_AVAILABLE:
            # UMAP parameters tuned for latent space structure
            self.reducer = umap.UMAP(
                n_neighbors=min(15, len(latent_vectors) - 1),
                min_dist=0.1,
                metric="euclidean",
                random_state=42,
            )
        else:
            # t-SNE fallback
            self.reducer = TSNE(
                n_components=2,
                perplexity=min(30, len(latent_vectors) - 1),
                random_state=42,
                init="pca",
                learning_rate="auto",
            )

        self.reducer.fit(latent_vectors)
        self.is_fitted = True
        logger.info(f"{self.method} projector fitted successfully")

    def transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """
        Project latent vectors into 2D space.

        Args:
            latent_vectors: Array of shape (n_samples, latent_dim)

        Returns:
            Array of shape (n_samples, 2)
        """
        if not self.is_fitted:
            # If not fitted, we might just do PCA as a quick fallback if we have enough points
            # but usually we want to fit on the dataset first.
            raise RuntimeError("Projector must be fitted before transform")

        if self.method == "umap" and UMAP_AVAILABLE:
            return self.reducer.transform(latent_vectors)
        else:
            # t-SNE doesn't have a stable .transform() for new data in sklearn easily
            # (unless using OpenTSNE or similar).
            # For t-SNE we usually fit_transform the whole batch.
            if hasattr(self.reducer, "fit_transform"):
                return self.reducer.fit_transform(latent_vectors)
            return np.zeros((len(latent_vectors), 2))

    def fit_transform(self, latent_vectors: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(latent_vectors)
        if self.method == "tsne":
            # For t-SNE, the result is stored during fit in some implementations
            # but usually we just return it here.
            return self.reducer.embedding_
        return self.transform(latent_vectors)

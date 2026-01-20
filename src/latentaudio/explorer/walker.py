# walker.py - Random walk generation in latent space
"""Random walk generation and exploration utilities."""

import numpy as np
from typing import List, Optional, Dict, Any
from ..types import WalkConfig, LatentVector
from ..logging import logger


class LatentWalker:
    """
    Generates exploratory walks through latent space.

    Useful for discovering new sounds and understanding the structure
    of the learned latent manifold.
    """

    def __init__(self, latent_dim: int = 128):
        """
        Initialize the walker.

        Args:
            latent_dim: Dimensionality of latent space
        """
        self.latent_dim = latent_dim
        logger.debug(f"LatentWalker initialized for {latent_dim}D space")

    def random_walk(
        self,
        start_vector: Optional[LatentVector] = None,
        n_steps: int = 8,
        step_size: float = 0.4,
        temperature: float = 1.0,
        momentum: float = 0.5,
        origin_pull: float = 0.1
    ) -> List[LatentVector]:
        """
        Generate a random walk through latent space with momentum.

        Args:
            start_vector: Starting point. If None, starts at origin.
            n_steps: Number of steps to take
            step_size: Size of each step
            temperature: Controls randomness (higher = more exploration)
            momentum: Persistence of direction (0.0 to 1.0)
            origin_pull: Subtle pull toward the center to stay on the manifold

        Returns:
            List of latent vectors representing the walk path
        """
        if start_vector is None:
            current = np.zeros(self.latent_dim)
        else:
            current = start_vector.copy()

        walk = [current.copy()]
        velocity = np.zeros(self.latent_dim)

        for step in range(n_steps):
            # 1. Generate new random direction
            random_dir = np.random.randn(self.latent_dim)
            random_dir = random_dir / (np.linalg.norm(random_dir) + 1e-8)

            # 2. Add origin pull (don't wander into dead space)
            # Pull is proportional to distance from center
            pull_dir = -current / (np.linalg.norm(current) + 1e-8)
            dist_from_origin = np.linalg.norm(current)
            
            # 3. Combine directions: Momentum + Random + Pull
            # We use (1-momentum) as the noise weight
            combined_dir = (
                momentum * velocity + 
                (1.0 - momentum) * random_dir + 
                origin_pull * (dist_from_origin / 10.0) * pull_dir
            )
            
            # Re-normalize direction
            velocity = combined_dir / (np.linalg.norm(combined_dir) + 1e-8)

            # 4. Take step scaled by step size and temperature
            step_vector = velocity * step_size * temperature
            current = current + step_vector
            walk.append(current.copy())

        logger.debug(f"Generated momentum walk: {len(walk)} points, displacement: {np.linalg.norm(walk[-1] - walk[0]):.2f}")
        return walk

    def guided_walk(
        self,
        start_vector: LatentVector,
        target_vector: LatentVector,
        n_steps: int = 10,
        exploration_ratio: float = 0.3
    ) -> List[LatentVector]:
        """
        Generate a walk that moves toward a target but with exploration.

        Args:
            start_vector: Starting point
            target_vector: Target to move toward
            n_steps: Number of steps
            exploration_ratio: Balance between moving to target (0.0) and exploring (1.0)

        Returns:
            Walk path that balances directed movement with exploration
        """
        current = start_vector.copy()
        walk = [current.copy()]

        # Direction to target
        target_direction = target_vector - start_vector
        target_direction = target_direction / (np.linalg.norm(target_direction) + 1e-8)

        for step in range(n_steps):
            # Combine target direction with random exploration
            random_direction = np.random.randn(self.latent_dim)
            random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)

            # Blend directions based on exploration ratio
            direction = (
                (1 - exploration_ratio) * target_direction +
                exploration_ratio * random_direction
            )
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Take step
            step_size = np.linalg.norm(target_vector - start_vector) / (n_steps + 1)
            current = current + direction * step_size
            walk.append(current.copy())

        logger.debug(f"Generated guided walk: {len(walk)} points toward target")
        return walk

    def boundary_walk(
        self,
        center_vector: LatentVector,
        radius: float = 1.0,
        n_points: int = 8
    ) -> List[LatentVector]:
        """
        Generate points on the boundary of a hypersphere around a center point.

        Useful for exploring the edges of learned distributions.

        Args:
            center_vector: Center point of the hypersphere
            radius: Radius of the hypersphere
            n_points: Number of points to generate on boundary

        Returns:
            Points on the hypersphere boundary
        """
        points = []

        for i in range(n_points):
            # Generate random direction
            direction = np.random.randn(self.latent_dim)
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Scale to radius and offset from center
            point = center_vector + direction * radius
            points.append(point)

        logger.debug(f"Generated boundary walk: {n_points} points at radius {radius}")
        return points

    def grid_walk(
        self,
        center_vector: LatentVector,
        dimensions: List[int],
        step_sizes: List[float]
    ) -> List[LatentVector]:
        """
        Generate a grid of points around a center vector.

        Useful for systematic exploration of specific dimensions.

        Args:
            center_vector: Center point
            dimensions: Which dimensions to vary (indices)
            step_sizes: Step sizes for each dimension

        Returns:
            Grid of latent vectors
        """
        if len(dimensions) != len(step_sizes):
            raise ValueError("dimensions and step_sizes must have same length")

        # Start with center point
        points = [center_vector.copy()]

        # Generate variations for each specified dimension
        for dim_idx, dim in enumerate(dimensions):
            step_size = step_sizes[dim_idx]
            new_points = []

            for point in points:
                # Add point with dimension decreased
                decreased = point.copy()
                decreased[dim] -= step_size
                new_points.append(decreased)

                # Add point with dimension increased
                increased = point.copy()
                increased[dim] += step_size
                new_points.append(increased)

            points.extend(new_points)

        logger.debug(f"Generated grid walk: {len(points)} points across {len(dimensions)} dimensions")
        return points
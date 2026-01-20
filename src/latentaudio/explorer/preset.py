# preset.py - Preset management with JSON storage
"""Preset management with JSON-based persistence."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from ..types import LatentPreset, LatentVector
from ..config import DEFAULT_MODELS_DIR
from ..logging import logger


class PresetManager:
    """
    Manages latent space presets with JSON file storage.

    Presets are saved as JSON files alongside model checkpoints,
    allowing for easy sharing and version control.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        """
        Initialize preset manager.

        Args:
            storage_dir: Directory for preset storage. Uses default if None.
        """
        self.storage_dir = storage_dir or DEFAULT_MODELS_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.presets_file = self.storage_dir / "presets.json"

        # In-memory preset storage
        self.presets: Dict[str, LatentPreset] = {}

        # Load existing presets
        self._load_presets()

        logger.debug(f"PresetManager initialized, {len(self.presets)} presets loaded")

    def _load_presets(self) -> None:
        """Load presets from JSON file."""
        if not self.presets_file.exists():
            logger.debug("No presets file found, starting fresh")
            return

        try:
            with open(self.presets_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            presets = {}
            for name, preset_data in data.items():
                try:
                    preset = LatentPreset.from_dict(preset_data)
                    presets[name] = preset
                except Exception as e:
                    logger.warning(f"Failed to load preset '{name}': {e}")

            self.presets = presets
            logger.info(f"Loaded {len(self.presets)} presets from {self.presets_file}")

        except Exception as e:
            logger.error(f"Failed to load presets file: {e}")
            self.presets = {}

    def _save_presets(self) -> None:
        """Save presets to JSON file."""
        try:
            data = {name: preset.to_dict() for name, preset in self.presets.items()}

            with open(self.presets_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.presets)} presets to {self.presets_file}")

        except Exception as e:
            logger.error(f"Failed to save presets: {e}")

    def save_preset(
        self,
        name: str,
        latent_vector: LatentVector,
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> None:
        """
        Save a latent vector as a preset.

        Args:
            name: Unique name for the preset
            latent_vector: The latent vector to save
            description: Optional description
            tags: Optional list of tags
        """
        if name in self.presets:
            logger.warning(f"Overwriting existing preset '{name}'")

        preset = LatentPreset(
            name=name,
            latent_vector=latent_vector.copy(),
            description=description,
            tags=tags or [],
            created_at=datetime.now().isoformat()
        )

        self.presets[name] = preset
        self._save_presets()

        logger.info(f"Saved preset '{name}' with tags: {tags}")

    def get_preset(self, name: str) -> Optional[LatentVector]:
        """
        Retrieve a preset by name.

        Args:
            name: Name of the preset

        Returns:
            Latent vector if found, None otherwise
        """
        preset = self.presets.get(name)
        if preset is None:
            logger.debug(f"Preset '{name}' not found")
            return None

        logger.debug(f"Retrieved preset '{name}'")
        return preset.latent_vector.copy()

    def get_preset_info(self, name: str) -> Optional[LatentPreset]:
        """
        Get full preset information.

        Args:
            name: Name of the preset

        Returns:
            Full preset object if found, None otherwise
        """
        preset = self.presets.get(name)
        if preset is None:
            return None

        # Return a copy to prevent external modification
        return LatentPreset(
            name=preset.name,
            latent_vector=preset.latent_vector.copy(),
            description=preset.description,
            tags=preset.tags.copy(),
            created_at=preset.created_at
        )

    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset.

        Args:
            name: Name of the preset to delete

        Returns:
            True if deleted, False if not found
        """
        if name not in self.presets:
            logger.debug(f"Preset '{name}' not found for deletion")
            return False

        del self.presets[name]
        self._save_presets()

        logger.info(f"Deleted preset '{name}'")
        return True

    def list_presets(self) -> List[str]:
        """
        Get list of all preset names.

        Returns:
            Sorted list of preset names
        """
        return sorted(self.presets.keys())

    def find_presets_by_tag(self, tag: str) -> List[str]:
        """
        Find presets with a specific tag.

        Args:
            tag: Tag to search for

        Returns:
            List of preset names with the tag
        """
        matching = [
            name for name, preset in self.presets.items()
            if tag in preset.tags
        ]
        logger.debug(f"Found {len(matching)} presets with tag '{tag}'")
        return matching

    def find_similar_presets(
        self,
        latent_vector: LatentVector,
        n: int = 5
    ) -> List[tuple[str, float]]:
        """
        Find presets most similar to a given latent vector.

        Args:
            latent_vector: Query vector
            n: Number of results to return

        Returns:
            List of (preset_name, distance) tuples, sorted by distance
        """
        import numpy as np

        distances = []
        for name, preset in self.presets.items():
            distance = np.linalg.norm(latent_vector - preset.latent_vector)
            distances.append((name, distance))

        distances.sort(key=lambda x: x[1])
        result = distances[:n]

        logger.debug(f"Found {len(result)} similar presets for query vector")
        return result

    def export_presets(self, filepath: Path) -> None:
        """
        Export presets to a JSON file.

        Args:
            filepath: Destination file path
        """
        data = {name: preset.to_dict() for name, preset in self.presets.items()}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Exported {len(self.presets)} presets to {filepath}")

    def import_presets(self, filepath: Path, overwrite: bool = False) -> int:
        """
        Import presets from a JSON file.

        Args:
            filepath: Source file path
            overwrite: Whether to overwrite existing presets with same names

        Returns:
            Number of presets imported
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        imported_count = 0
        for name, preset_data in data.items():
            if name in self.presets and not overwrite:
                logger.warning(f"Skipping existing preset '{name}' (use overwrite=True)")
                continue

            try:
                preset = LatentPreset.from_dict(preset_data)
                self.presets[name] = preset
                imported_count += 1
            except Exception as e:
                logger.warning(f"Failed to import preset '{name}': {e}")

        if imported_count > 0:
            self._save_presets()
            logger.info(f"Imported {imported_count} presets from {filepath}")

        return imported_count

    def clear_all_presets(self) -> None:
        """Clear all presets."""
        self.presets.clear()
        self._save_presets()
        logger.info("Cleared all presets")
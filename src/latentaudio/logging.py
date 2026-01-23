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
# logging.py - Logging configuration for LatentAudio
"""Logging configuration and utilities for LatentAudio."""

import sys
from pathlib import Path
from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    format_string: str | None = None,
    serialize: bool = False,
) -> None:
    """Configure logging for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for log output
        format_string: Custom format string, or None for default
        serialize: Whether to serialize logs as JSON
    """
    # Remove default handler
    logger.remove()

    # Default format
    if format_string is None:
        if serialize:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            )
        else:
            format_string = (
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}:{function}:{line}</cyan> | "
                "<level>{message}</level>"
            )

    # Add console handler
    logger.add(
        sys.stdout,
        level=level,
        format=format_string,
        serialize=serialize,
        colorize=True,
    )

    # Add file handler if specified
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file,
            level=level,
            format=format_string,
            serialize=serialize,
            rotation="10 MB",
            retention="1 week",
            encoding="utf-8",
        )

    # Set up uncaught exception handler
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        # Also print to stderr for immediate visibility
        import traceback

        print("UNCAUGHT EXCEPTION DETAILS:", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stderr)

    sys.excepthook = handle_exception

    logger.info("Logging initialized", level=level, log_file=str(log_file) if log_file else None)


# Convenience functions for different log levels
def log_training_start(config) -> None:
    """Log training initialization."""
    logger.info("Starting model training", **config.__dict__)


def log_training_epoch(epoch: int, metrics) -> None:
    """Log training epoch metrics."""
    logger.info(f"Training epoch {epoch}", **metrics.__dict__)


def log_generation_start(config) -> None:
    """Log generation initialization."""
    logger.info("Starting audio generation", **config.__dict__)


def log_model_save(path: Path) -> None:
    """Log model saving."""
    logger.info("Model saved", path=str(path))


def log_model_load(path: Path) -> None:
    """Log model loading."""
    logger.info("Model loaded", path=str(path))


# Initialize with default settings
setup_logging()

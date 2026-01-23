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
# cli.py - Command-line interface for LatentAudio
"""Command-line interface for LatentAudio operations."""

import os
import sys
from pathlib import Path
from typing import Optional, List
import json

import click
import numpy as np
from tqdm import tqdm

from .core.generator import AdvancedAudioGenerator
from .types import GeneratorConfig, TrainingConfig, AudioConfig, ModelConfig, DeviceConfig
from .logging import setup_logging, logger
from .config import DEFAULT_MODELS_DIR


def setup_cli_logging(verbose: bool = False):
    """Setup logging for CLI operations."""
    level = "DEBUG" if verbose else "INFO"
    setup_logging(level=level, log_file=None)


def load_config_from_file(filepath: Path) -> dict:
    """Load configuration from JSON file."""
    if not filepath.exists():
        raise click.BadParameter(f"Configuration file not found: {filepath}")

    with open(filepath, "r") as f:
        return json.load(f)


def create_generator_from_config(config_data: dict) -> AdvancedAudioGenerator:
    """Create generator from configuration dictionary."""
    # Handle both old and new config formats
    if "sample_rate" in config_data:
        # Legacy format
        config = GeneratorConfig(
            sample_rate=config_data["sample_rate"],
            duration=config_data.get("duration", 0.5),
            latent_dim=config_data.get("latent_dim", 128),
            device=config_data.get("device"),
        )
    else:
        # New nested format
        audio_data = config_data.get("audio", {})
        model_data = config_data.get("model", {})
        device_data = config_data.get("device", {})

        config = GeneratorConfig(
            audio=AudioConfig(**audio_data),
            model=ModelConfig(**model_data),
            device=DeviceConfig(**device_data),
        )

    return AdvancedAudioGenerator(config)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option(
    "--config", "-c", type=click.Path(exists=True, path_type=Path), help="Configuration file (JSON)"
)
@click.pass_context
def cli(ctx, verbose, config):
    """LatentAudio: Direct Neural Audio Generation and Exploration.

    A neural audio synthesis system that provides direct exploration
    of the learned manifold of sounds through latent space manipulation.
    """
    setup_cli_logging(verbose)

    # Store config for subcommands
    ctx.ensure_object(dict)
    if config:
        ctx.obj["config"] = load_config_from_file(config)
    else:
        ctx.obj["config"] = None

    logger.info("LatentAudio CLI initialized")


@cli.command()
@click.argument("data_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=DEFAULT_MODELS_DIR / "trained_model.pth",
    help="Output model path",
)
@click.option("--epochs", "-e", type=int, default=300, help="Number of training epochs")
@click.option("--batch-size", "-b", type=int, default=16, help="Batch size for training")
@click.option("--lr", type=float, default=0.0003, help="Learning rate")
@click.option("--latent-dim", "-l", type=int, default=128, help="Latent dimension size")
@click.option("--sample-rate", "-r", type=int, default=44100, help="Audio sample rate")
@click.option("--duration", "-d", type=float, default=0.5, help="Audio duration in seconds")
@click.option("--device", type=str, default="auto", help="Device to use (cpu, cuda, auto)")
@click.option("--tensorboard", is_flag=True, help="Launch TensorBoard after training")
@click.pass_context
def train(
    ctx,
    data_dir,
    output,
    epochs,
    batch_size,
    lr,
    latent_dim,
    sample_rate,
    duration,
    device,
    tensorboard,
):
    """Train a new VAE model on audio data.

    DATA_DIR should contain WAV/MP3/FLAC audio files to train on.
    """
    logger.info(f"Starting training with data from: {data_dir}")

    # Create generator
    if ctx.obj["config"]:
        generator = create_generator_from_config(ctx.obj["config"])
    else:
        config = GeneratorConfig(
            sample_rate=sample_rate,
            duration=duration,
            latent_dim=latent_dim,
            device=device if device != "auto" else None,
        )
        generator = AdvancedAudioGenerator(config)

    # Load training data
    logger.info("Loading training data...")
    samples = generator.load_dataset(data_dir, recursive=True)

    if len(samples) == 0:
        raise click.ClickException(f"No audio files found in {data_dir}")

    logger.info(f"Loaded {len(samples)} audio samples")

    # Create training config
    train_config = TrainingConfig(
        epochs=epochs, batch_size=min(batch_size, len(samples)), learning_rate=lr
    )

    # Train model
    logger.info("Starting model training...")
    try:
        generator.train(samples, train_config)
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise click.ClickException(f"Training failed: {e}")

    # Save model
    output.parent.mkdir(parents=True, exist_ok=True)
    generator.save_model(output)
    logger.info(f"Model saved to: {output}")

    # Launch TensorBoard if requested
    if tensorboard:
        logger.info("Launching TensorBoard...")
        generator.start_tensorboard()

    click.echo(f"✅ Training complete! Model saved to {output}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("generated_audio"),
    help="Output directory for generated audio",
)
@click.option("--count", "-n", type=int, default=10, help="Number of audio files to generate")
@click.option(
    "--temperature", "-t", type=float, default=1.0, help="Generation temperature (randomness)"
)
@click.option(
    "--prefix", "-p", type=str, default="generated", help="Filename prefix for generated audio"
)
@click.pass_context
def generate(ctx, model_path, output_dir, count, temperature, prefix):
    """Generate audio using a trained model.

    MODEL_PATH should point to a trained .pth model file.
    """
    logger.info(f"Loading model from: {model_path}")

    # Create generator and load model
    if ctx.obj["config"]:
        generator = create_generator_from_config(ctx.obj["config"])
    else:
        generator = AdvancedAudioGenerator()

    generator.load_model(model_path)
    logger.info("Model loaded successfully")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate audio
    logger.info(f"Generating {count} audio samples...")

    audios = generator.generate_random(n_samples=count, config={"temperature": temperature})

    # Save audio files
    for i, audio in enumerate(tqdm(audios, desc="Saving audio")):
        filename = f"{prefix}_{i:03d}.wav"
        filepath = output_dir / filename
        generator.save_audio(audio, filepath)

    logger.info(f"Generated {count} audio files in {output_dir}")
    click.echo(f"✅ Generated {count} audio files in {output_dir}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--vectors", "-v", type=str, required=True, help="Latent vectors as JSON array or file path"
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("latent_audio.wav"),
    help="Output audio file path",
)
@click.pass_context
def synthesize(ctx, model_path, vectors, output):
    """Synthesize audio from specific latent vectors.

    VECTORS can be a JSON array like "[0.1, 0.2, ...]" or a path to a JSON file
    containing the latent vector.
    """
    logger.info(f"Loading model from: {model_path}")

    # Create generator and load model
    if ctx.obj["config"]:
        generator = create_generator_from_config(ctx.obj["config"])
    else:
        generator = AdvancedAudioGenerator()

    generator.load_model(model_path)

    # Parse latent vectors
    try:
        if vectors.startswith("["):  # JSON array
            z = np.array(json.loads(vectors))
        else:  # File path
            vector_file = Path(vectors)
            if not vector_file.exists():
                raise click.BadParameter(f"Vector file not found: {vectors}")

            with open(vector_file, "r") as f:
                data = json.load(f)

            if isinstance(data, list):
                z = np.array(data)
            elif isinstance(data, dict) and "latent_vector" in data:
                z = np.array(data["latent_vector"])
            else:
                raise click.BadParameter("Invalid vector file format")

    except (json.JSONDecodeError, ValueError) as e:
        raise click.BadParameter(f"Invalid latent vectors: {e}")

    # Validate vector dimensions
    if z.ndim != 1 or z.shape[0] != generator.config.latent_dim:
        raise click.BadParameter(f"Expected {generator.config.latent_dim}D vector, got {z.shape}")

    # Generate audio
    logger.info("Generating audio from latent vector...")
    audio = generator.generate_from_latent(z)

    # Save audio
    output.parent.mkdir(parents=True, exist_ok=True)
    generator.save_audio(audio, output)

    logger.info(f"Audio saved to: {output}")
    click.echo(f"✅ Audio synthesized and saved to {output}")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option("--preset-name", "-p", type=str, help="Preset name to save/load")
@click.option("--latent-vector", "-v", type=str, help="Latent vector as JSON array (for saving)")
@click.option("--description", type=str, help="Preset description")
@click.option("--tags", type=str, help="Comma-separated preset tags")
@click.option("--list", "list_presets", is_flag=True, help="List all presets")
@click.option("--find-tag", type=str, help="Find presets with specific tag")
@click.pass_context
def presets(ctx, model_path, preset_name, latent_vector, description, tags, list_presets, find_tag):
    """Manage model presets.

    MODEL_PATH should point to a trained .pth model file.
    """
    # Create generator and load model
    if ctx.obj["config"]:
        generator = create_generator_from_config(ctx.obj["config"])
    else:
        generator = AdvancedAudioGenerator()

    generator.load_model(model_path)

    if list_presets:
        # List all presets
        presets = generator.list_presets()
        if not presets:
            click.echo("No presets found")
            return

        click.echo("Available presets:")
        for name in presets:
            info = generator.get_preset_info(name)
            desc = (
                info.description[:50] + "..."
                if info.description and len(info.description) > 50
                else info.description
            )
            tags_str = ", ".join(info.tags) if info.tags else "no tags"
            click.echo(f"  • {name}: {desc or 'no description'} [{tags_str}]")

    elif find_tag:
        # Find presets by tag
        matching = generator.find_presets_by_tag(find_tag)
        if not matching:
            click.echo(f"No presets found with tag '{find_tag}'")
            return

        click.echo(f"Presets with tag '{find_tag}':")
        for name in matching:
            click.echo(f"  • {name}")

    elif preset_name and latent_vector:
        # Save preset
        try:
            z = np.array(json.loads(latent_vector))
            tag_list = [t.strip() for t in tags.split(",")] if tags else None

            generator.save_preset(
                name=preset_name, latent_vector=z, description=description, tags=tag_list
            )

            click.echo(f"✅ Preset '{preset_name}' saved")

        except (json.JSONDecodeError, ValueError) as e:
            raise click.BadParameter(f"Invalid latent vector: {e}")

    elif preset_name:
        # Load preset
        z = generator.get_preset(preset_name)
        if z is None:
            raise click.BadParameter(f"Preset '{preset_name}' not found")

        click.echo(f"Latent vector for preset '{preset_name}':")
        click.echo(json.dumps(z.tolist(), indent=2))

    else:
        click.echo("Use --help to see available preset operations")


@cli.command()
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=Path("latent_walk.wav"),
    help="Output audio file path",
)
@click.option("--steps", "-s", type=int, default=8, help="Number of walk steps")
@click.option("--step-size", type=float, default=0.4, help="Size of each step")
@click.option("--momentum", type=float, default=0.5, help="Persistence of direction (0.0 to 1.0)")
@click.option("--origin-pull", type=float, default=0.1, help="Subtle pull toward the center")
@click.option("--start-preset", type=str, help="Starting preset name")
@click.pass_context
def walk(ctx, model_path, output, steps, step_size, momentum, origin_pull, start_preset):
    """Generate a random walk through latent space.

    MODEL_PATH should point to a trained .pth model file.
    """
    logger.info(f"Loading model from: {model_path}")

    # Create generator and load model
    if ctx.obj["config"]:
        generator = create_generator_from_config(ctx.obj["config"])
    else:
        generator = AdvancedAudioGenerator()

    generator.load_model(model_path)

    # Get starting vector
    if start_preset:
        start_vector = generator.get_preset(start_preset)
        if start_vector is None:
            raise click.BadParameter(f"Starting preset '{start_preset}' not found")
        logger.info(f"Starting walk from preset '{start_preset}'")
    else:
        start_vector = None
        logger.info("Starting walk from origin")

    # Generate walk
    logger.info(f"Generating {steps}-step random walk...")
    walk_path = generator.random_walk(
        start_vector=start_vector,
        n_steps=steps,
        step_size=step_size,
        momentum=momentum,
        origin_pull=origin_pull,
    )

    # Generate audio for each step
    logger.info("Generating audio for walk...")
    audios = []
    for z in tqdm(walk_path, desc="Generating audio"):
        audio = generator.generate_from_latent(z)
        audios.append(audio)

    # Concatenate with gaps
    gap_samples = int(0.1 * generator.config.sample_rate)  # 100ms gap
    gap = np.zeros(gap_samples)

    full_sequence = []
    for audio in audios:
        full_sequence.extend([audio, gap])

    walk_audio = np.concatenate(full_sequence)

    # Save audio
    output.parent.mkdir(parents=True, exist_ok=True)
    generator.save_audio(walk_audio, output)

    logger.info(f"Walk audio saved to: {output}")
    click.echo(f"✅ Generated {steps}-step walk, saved to {output}")


@cli.command()
@click.argument("input_model", type=click.Path(exists=True, path_type=Path))
@click.argument("output_model", type=click.Path(path_type=Path))
@click.option("--optimize", is_flag=True, help="Apply TorchScript optimization")
@click.option("--quantize", type=click.Choice(["dynamic", "static"]), help="Apply quantization")
@click.pass_context
def convert(ctx, input_model, output_model, optimize, quantize):
    """Convert and optimize models for deployment.

    INPUT_MODEL: Source model file (.pth)
    OUTPUT_MODEL: Output model file (.pth or .pt)
    """
    logger.info(f"Converting model: {input_model} -> {output_model}")

    # Create generator and load model
    if ctx.obj["config"]:
        generator = create_generator_from_config(ctx.obj["config"])
    else:
        generator = AdvancedAudioGenerator()

    generator.load_model(input_model)
    logger.info("Model loaded for conversion")

    if optimize:
        logger.info("Applying TorchScript optimization...")
        # Note: Full TorchScript optimization would require more implementation
        # This is a placeholder for the concept
        click.echo("⚠️ TorchScript optimization not yet implemented")

    if quantize:
        logger.info(f"Applying {quantize} quantization...")
        # Note: Quantization would require additional implementation
        click.echo("⚠️ Quantization not yet implemented")

    # For now, just copy the model (placeholder)
    import shutil

    output_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_model, output_model)

    logger.info(f"Model converted and saved to: {output_model}")
    click.echo(f"✅ Model converted to {output_model}")


@cli.command()
@click.option("--port", type=int, default=6006, help="TensorBoard port")
@click.option("--host", type=str, default="localhost", help="TensorBoard host")
@click.option(
    "--logdir",
    type=click.Path(path_type=Path),
    default=DEFAULT_MODELS_DIR,
    help="Log directory to serve",
)
def tensorboard(port, host, logdir):
    """Launch TensorBoard server.

    Serves TensorBoard logs from the specified directory.
    """
    from .core.training import TrainingLogger

    logger.info(f"Starting TensorBoard on {host}:{port}")
    logger.info(f"Log directory: {logdir}")

    # Create a temporary logger just to launch TensorBoard
    temp_logger = TrainingLogger(log_dir=logdir, experiment_name="cli_temp")
    temp_logger.start_tensorboard(port=port, host=host)

    click.echo(f"✅ TensorBoard launched at http://{host}:{port}")
    click.echo("Press Ctrl+C to stop the server")


@cli.command()
def info():
    """Show information about LatentAudio installation."""
    click.echo("LatentAudio v1.0.0")
    click.echo("Direct Neural Audio Generation and Exploration")
    click.echo("")
    click.echo("Core Features:")
    click.echo("  * VAE-based audio synthesis")
    click.echo("  * Latent space exploration")
    click.echo("  * Real-time generation")
    click.echo("  * TensorBoard monitoring")
    click.echo("")
    click.echo("Available Commands:")
    click.echo("  train      - Train new models")
    click.echo("  generate   - Generate random audio")
    click.echo("  synthesize - Generate from latent vectors")
    click.echo("  presets    - Manage model presets")
    click.echo("  walk       - Generate latent space walks")
    click.echo("  convert    - Convert/optimize models")
    click.echo("  tensorboard- Launch TensorBoard")
    click.echo("")
    click.echo("Use 'latentaudio <command> --help' for more info")


def main():
    """Main CLI entry point."""
    try:
        cli()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

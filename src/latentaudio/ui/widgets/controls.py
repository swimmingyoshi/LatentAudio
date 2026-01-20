# controls.py - Audio generation and playback controls
"""Audio generation and playback control widgets."""

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import pyqtSignal

from ..theme import BUTTON_STYLE, TEXT_SECONDARY, NORMAL_FONT


class GenerationControls(QWidget):
    """Controls for audio generation."""

    generate_requested = pyqtSignal()
    random_generate_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the generation controls."""
        layout = QHBoxLayout(self)

        self.generate_btn = QPushButton("🎵 Generate Sound")
        self.generate_btn.clicked.connect(self.generate_requested.emit)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.generate_btn)

        self.random_generate_btn = QPushButton("🎲 Generate Random")
        self.random_generate_btn.clicked.connect(self.random_generate_requested.emit)
        self.random_generate_btn.setEnabled(False)
        self.random_generate_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.random_generate_btn)

    def set_generation_enabled(self, enabled: bool):
        """Enable or disable generation buttons."""
        self.generate_btn.setEnabled(enabled)
        self.random_generate_btn.setEnabled(enabled)


class PlaybackControls(QWidget):
    """Controls for audio playback and saving."""

    play_requested = pyqtSignal()
    save_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the playback controls."""
        layout = QHBoxLayout(self)

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play_requested.emit)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.play_btn)

        self.save_btn = QPushButton("💾 Save WAV")
        self.save_btn.clicked.connect(self.save_requested.emit)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.save_btn)

    def set_playback_enabled(self, enabled: bool):
        """Enable or disable playback controls."""
        self.play_btn.setEnabled(enabled)
        self.save_btn.setEnabled(enabled)


class StatusWidget(QWidget):
    """Status display widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        """Set up the status display."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.status_label = QLabel("Ready - Load or train a model to begin")
        self.status_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        self.status_label.setFont(NORMAL_FONT)
        layout.addWidget(self.status_label)

    def set_status(self, text: str):
        """Set the status text."""
        self.status_label.setText(text)

    def set_ready_status(self):
        """Set status to ready state."""
        self.set_status("Ready")

    def set_training_status(self):
        """Set status to training state."""
        self.set_status("Training model...")

    def set_generating_status(self):
        """Set status to generating state."""
        self.set_status("Generating audio...")

    def set_loading_status(self):
        """Set status to loading state."""
        self.set_status("Loading model...")

    def set_error_status(self, error: str):
        """Set status to error state."""
        self.set_status(f"Error: {error}")

    def set_training_progress(self, epoch: int, total_epochs: int, loss: float):
        """Set status to show training progress."""
        self.set_status(f"Training Epoch {epoch}/{total_epochs} - Loss: {loss:.6f}")
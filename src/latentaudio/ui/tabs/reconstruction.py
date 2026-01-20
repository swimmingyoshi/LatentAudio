# reconstruction.py - VAE Reconstruction testing tab
"""Tab for testing how well the model reconstructs existing audio samples."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor

from ..widgets.visualizer import WaveformVisualizer
from ..theme import (
    BUTTON_STYLE, GROUP_BOX_STYLE, ACCENT_PRIMARY,
    TEXT_SECONDARY, NORMAL_FONT
)


class ReconstructionTab(QWidget):
    """Tab for encoding and reconstructing existing audio samples."""

    # Signal emitted when a vector is pushed to the main UI
    vector_pushed = pyqtSignal(np.ndarray)

    def __init__(self, generator=None, latent_widget=None, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.latent_widget = latent_widget
        
        self.original_audio = None
        self.recon_audio = None
        self.current_vector = None
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the reconstruction interface."""
        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel("Test how well the model replicates real audio")
        info.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        info.setFont(NORMAL_FONT)
        layout.addWidget(info)

        # Load button
        self.load_btn = QPushButton("📁 Load Reference Sample")
        self.load_btn.clicked.connect(self.load_reference)
        self.load_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.load_btn)

        # Reference section
        ref_group = QGroupBox("Original Reference")
        ref_group.setStyleSheet(GROUP_BOX_STYLE)
        ref_layout = QVBoxLayout()
        
        self.ref_waveform = WaveformVisualizer(color=QColor(100, 150, 255))
        ref_layout.addWidget(self.ref_waveform)
        
        ref_controls = QHBoxLayout()
        self.play_ref_btn = QPushButton("▶ Play Original")
        self.play_ref_btn.clicked.connect(lambda: self.play_audio(self.original_audio))
        self.play_ref_btn.setEnabled(False)
        self.play_ref_btn.setStyleSheet(BUTTON_STYLE)
        ref_controls.addWidget(self.play_ref_btn)
        
        ref_layout.addLayout(ref_controls)
        ref_group.setLayout(ref_layout)
        layout.addWidget(ref_group)

        # Reconstruction section
        recon_group = QGroupBox("Model Reconstruction")
        recon_group.setStyleSheet(GROUP_BOX_STYLE)
        recon_layout = QVBoxLayout()
        
        self.recon_waveform = WaveformVisualizer()
        recon_layout.addWidget(self.recon_waveform)
        
        recon_controls = QHBoxLayout()
        self.play_recon_btn = QPushButton("▶ Play Reconstruction")
        self.play_recon_btn.clicked.connect(lambda: self.play_audio(self.recon_audio))
        self.play_recon_btn.setEnabled(False)
        self.play_recon_btn.setStyleSheet(BUTTON_STYLE)
        recon_controls.addWidget(self.play_recon_btn)
        
        recon_layout.addLayout(recon_controls)
        recon_group.setLayout(recon_layout)
        layout.addWidget(recon_group)

        # Action buttons
        actions_layout = QHBoxLayout()
        
        self.push_btn = QPushButton("🧬 Push to Main Sliders")
        self.push_btn.clicked.connect(self.push_to_sliders)
        self.push_btn.setEnabled(False)
        self.push_btn.setStyleSheet(BUTTON_STYLE)
        actions_layout.addWidget(self.push_btn)
        
        layout.addLayout(actions_layout)
        layout.addStretch()

    def load_reference(self):
        """Load an audio file, encode it, and reconstruct it."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded. Train or load a model first!")
            return

        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Reference Audio", "", "Audio Files (*.wav *.mp3 *.flac *.ogg)"
        )
        if not filepath:
            return

        try:
            # 1. Load and process
            self.original_audio = self.generator.load_audio_file(filepath)
            self.ref_waveform.set_audio(self.original_audio)
            self.play_ref_btn.setEnabled(True)

            # 2. Encode
            self.current_vector = self.generator.encode_audio(self.original_audio)
            
            # 3. Reconstruct
            self.recon_audio = self.generator.generate_from_latent(self.current_vector)
            self.recon_waveform.set_audio(self.recon_audio)
            self.play_recon_btn.setEnabled(True)
            self.push_btn.setEnabled(True)

            QMessageBox.information(self, "Success", "Audio encoded and reconstructed successfully!")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Reconstruction failed: {e}")

    def play_audio(self, audio):
        """Play given audio array."""
        if audio is not None:
            try:
                import sounddevice as sd
                sd.play(audio, self.generator.config.sample_rate)
            except ImportError:
                QMessageBox.warning(self, "Error", "Install sounddevice: pip install sounddevice")

    def push_to_sliders(self):
        """Send the current latent vector to the main UI sliders."""
        if self.current_vector is not None:
            self.vector_pushed.emit(self.current_vector)
            QMessageBox.information(self, "Success", "Vector pushed to main sliders!")

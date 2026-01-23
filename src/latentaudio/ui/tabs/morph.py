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
# morph.py - Morphing/Interpolation tab widget
"""Morphing and interpolation tab for creating smooth transitions between sounds."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QSpinBox,
    QComboBox,
    QSlider,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np

from ..widgets.visualizer import WaveformVisualizer
from ..theme import BUTTON_STYLE, GROUP_BOX_STYLE, ACCENT_PRIMARY, TEXT_SECONDARY, NORMAL_FONT


class MorphTab(QWidget):
    """Tab for creating smooth morphs between latent vectors."""

    morph_generated = pyqtSignal(list)  # List of latent vectors

    def __init__(self, generator=None, latent_widget=None, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.latent_widget = latent_widget
        self.z_start = None
        self.z_end = None
        self.current_morph = []
        self.current_audio = []
        self.current_index = 0
        self.setup_ui()

    def setup_ui(self):
        """Set up the morphing interface."""
        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel("Create smooth transitions between sounds")
        info.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        info.setFont(NORMAL_FONT)
        layout.addWidget(info)

        # Capture buttons
        capture_layout = QHBoxLayout()

        self.set_start_btn = QPushButton("Set Start (A)")
        self.set_start_btn.clicked.connect(self.set_start)
        self.set_start_btn.setStyleSheet(BUTTON_STYLE)
        capture_layout.addWidget(self.set_start_btn)

        self.set_end_btn = QPushButton("Set End (B)")
        self.set_end_btn.clicked.connect(self.set_end)
        self.set_end_btn.setStyleSheet(BUTTON_STYLE)
        capture_layout.addWidget(self.set_end_btn)

        layout.addLayout(capture_layout)

        # Status
        self.status_label = QLabel("Set both start and end points")
        self.status_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()}; font-style: italic;")
        self.status_label.setFont(NORMAL_FONT)
        layout.addWidget(self.status_label)

        # Settings
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(3, 50)
        self.steps_spin.setValue(10)
        settings_layout.addWidget(self.steps_spin)

        settings_layout.addWidget(QLabel("Method:"))
        self.method_combo = QComboBox()
        self.method_combo.addItems(["Spherical (SLERP)", "Linear"])
        settings_layout.addWidget(self.method_combo)

        layout.addLayout(settings_layout)

        # Generate button
        self.generate_btn = QPushButton("🎵 Generate Morph")
        self.generate_btn.clicked.connect(self.generate_morph)
        self.generate_btn.setEnabled(False)
        self.generate_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.generate_btn)

        # Preview section
        preview_group = QGroupBox("Preview Morph")
        preview_group.setStyleSheet(GROUP_BOX_STYLE)
        preview_layout = QVBoxLayout()

        # Preview waveform
        preview_label = QLabel("Current Step")
        preview_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()}; font-weight: bold;")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_layout.addWidget(preview_label)

        self.preview_waveform = WaveformVisualizer()
        preview_layout.addWidget(self.preview_waveform)

        # Progress slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Step:"))

        self.preview_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_slider.setMinimum(0)
        self.preview_slider.setMaximum(0)
        self.preview_slider.setValue(0)
        self.preview_slider.setEnabled(False)
        self.preview_slider.valueChanged.connect(self.on_slider_changed)
        slider_layout.addWidget(self.preview_slider)

        self.step_label = QLabel("0/0")
        self.step_label.setMinimumWidth(60)
        self.step_label.setFont(NORMAL_FONT)
        slider_layout.addWidget(self.step_label)

        preview_layout.addLayout(slider_layout)

        # Preview controls
        controls_layout = QHBoxLayout()

        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.play_current)
        self.play_btn.setEnabled(False)
        self.play_btn.setStyleSheet(BUTTON_STYLE)
        controls_layout.addWidget(self.play_btn)

        self.play_all_btn = QPushButton("🔊 Play ALL")
        self.play_all_btn.clicked.connect(self.play_all)
        self.play_all_btn.setEnabled(False)
        self.play_all_btn.setStyleSheet(BUTTON_STYLE)
        controls_layout.addWidget(self.play_all_btn)

        self.save_btn = QPushButton("💾 Save All")
        self.save_btn.clicked.connect(self.save_morph)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(BUTTON_STYLE)
        controls_layout.addWidget(self.save_btn)

        preview_layout.addLayout(controls_layout)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()

    def set_start(self):
        """Capture the current latent vector as start point."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded. Train or load a model first!")
            return

        # Get current vector from latent widget
        if self.latent_widget:
            try:
                self.z_start = self.latent_widget.get_vector().copy()
                self.update_status()
                QMessageBox.information(self, "Success", "Start point (A) captured!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to capture start: {e}")

    def set_end(self):
        """Capture the current latent vector as end point."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded. Train or load a model first!")
            return

        # Get current vector from latent widget
        if self.latent_widget:
            try:
                self.z_end = self.latent_widget.get_vector().copy()
                self.update_status()
                QMessageBox.information(self, "Success", "End point (B) captured!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to capture end: {e}")

    def update_status(self):
        """Update status display based on captured points."""
        has_start = self.z_start is not None
        has_end = self.z_end is not None

        if has_start and has_end:
            self.status_label.setText("Ready to generate morph: A → B")
            self.status_label.setStyleSheet(f"color: {ACCENT_PRIMARY.name()};")
            self.generate_btn.setEnabled(True)
        else:
            parts = []
            if has_start:
                parts.append("A✓")
            if has_end:
                parts.append("B✓")

            self.status_label.setText(
                "Set both points: " + ", ".join(parts) if parts else "Set both start and end points"
            )
            self.status_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
            self.generate_btn.setEnabled(False)

    def generate_morph(self):
        """Generate the morph sequence."""
        if self.z_start is None or self.z_end is None:
            return

        n_steps = self.steps_spin.value()
        method = "spherical" if "Spherical" in self.method_combo.currentText() else "linear"

        try:
            # Generate latent morph
            self.current_morph = self.generator.interpolate(
                self.z_start, self.z_end, n_steps, method
            )

            # Generate audio for each step
            self.current_audio = []
            for z in self.current_morph:
                audio = self.generator.generate_from_latent(z)
                self.current_audio.append(audio)

            # Setup preview
            self.current_index = 0
            self.preview_slider.setMaximum(len(self.current_morph) - 1)
            self.preview_slider.setValue(0)
            self.preview_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.play_all_btn.setEnabled(True)
            self.save_btn.setEnabled(True)

            self.update_preview()
            self.morph_generated.emit(self.current_morph)

            QMessageBox.information(
                self,
                "Success",
                f"Generated {len(self.current_morph)}-step morph!\nUse slider to preview each step.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate morph: {e}")

    def on_slider_changed(self, value):
        """Handle slider value change."""
        self.current_index = value
        self.update_preview()

    def update_preview(self):
        """Update the preview display."""
        if self.current_audio and 0 <= self.current_index < len(self.current_audio):
            self.preview_waveform.set_audio(self.current_audio[self.current_index])
            self.step_label.setText(f"{self.current_index + 1}/{len(self.current_morph)}")

    def play_current(self):
        """Play the current step's audio."""
        if self.current_audio and 0 <= self.current_index < len(self.current_audio):
            try:
                import sounddevice as sd

                audio = self.current_audio[self.current_index]
                sd.play(audio, self.generator.config.sample_rate)
            except ImportError:
                QMessageBox.warning(self, "Error", "Install sounddevice: pip install sounddevice")

    def play_all(self):
        """Play the entire morph sequence as one audio clip."""
        if not self.current_audio:
            return

        try:
            import sounddevice as sd

            # Concatenate all steps
            full_audio = np.concatenate(self.current_audio)
            # Play
            sd.play(full_audio, self.generator.config.sample_rate)
        except ImportError:
            QMessageBox.warning(self, "Error", "Install sounddevice: pip install sounddevice")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to play all: {e}")

    def save_morph(self):
        """Save the morph sequence to files."""
        if not self.current_audio:
            return

        folder = QFileDialog.getExistingDirectory(self, "Save Morph Sequence")
        if not folder:
            return

        try:
            for i, audio in enumerate(self.current_audio):
                filename = f"{folder}/morph_{i:03d}.wav"
                self.generator.save_audio(audio, filename)

            QMessageBox.information(
                self, "Success", f"Saved {len(self.current_audio)} files to {folder}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save morph: {e}")

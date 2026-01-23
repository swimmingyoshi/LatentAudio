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
# walk.py - Random walk exploration tab widget
"""Random walk exploration tab for discovering new sounds."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QFileDialog,
    QMessageBox,
)
from PyQt6.QtCore import Qt

from ..widgets.visualizer import WaveformVisualizer
from ..theme import BUTTON_STYLE, GROUP_BOX_STYLE, TEXT_SECONDARY, NORMAL_FONT


class WalkTab(QWidget):
    """Tab for random walk exploration through latent space."""

    def __init__(self, generator=None, latent_widget=None, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.latent_widget = latent_widget
        self.current_walk = []
        self.current_audio = []
        self.current_index = 0
        self.setup_ui()

    def setup_ui(self):
        """Set up the walk exploration interface."""
        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel("Explore latent space with random walks")
        info.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        info.setFont(NORMAL_FONT)
        layout.addWidget(info)

        # Settings
        settings_layout = QHBoxLayout()

        settings_layout.addWidget(QLabel("Steps:"))
        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(3, 20)
        self.steps_spin.setValue(8)
        settings_layout.addWidget(self.steps_spin)

        settings_layout.addWidget(QLabel("Step Size:"))
        self.step_size_spin = QDoubleSpinBox()
        self.step_size_spin.setRange(0.1, 10.0)
        self.step_size_spin.setValue(0.8)
        self.step_size_spin.setSingleStep(0.2)
        settings_layout.addWidget(self.step_size_spin)

        settings_layout.addWidget(QLabel("Momentum:"))
        self.momentum_spin = QDoubleSpinBox()
        self.momentum_spin.setRange(0.0, 1.0)
        self.momentum_spin.setValue(0.5)
        self.momentum_spin.setSingleStep(0.1)
        settings_layout.addWidget(self.momentum_spin)

        settings_layout.addWidget(QLabel("Origin Pull:"))
        self.pull_spin = QDoubleSpinBox()
        self.pull_spin.setRange(0.0, 1.0)
        self.pull_spin.setValue(0.1)
        self.pull_spin.setSingleStep(0.05)
        settings_layout.addWidget(self.pull_spin)

        layout.addLayout(settings_layout)

        # Generate button
        self.generate_btn = QPushButton("🚶 Generate Walk")
        self.generate_btn.clicked.connect(self.generate_walk)
        self.generate_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.generate_btn)

        # Status
        self.status_label = QLabel("Ready to explore")
        self.status_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()}; font-style: italic;")
        self.status_label.setFont(NORMAL_FONT)
        layout.addWidget(self.status_label)

        # Preview section
        preview_group = QGroupBox("Preview Walk")
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

        self.play_all_btn = QPushButton("▶▶ Play All")
        self.play_all_btn.clicked.connect(self.play_all)
        self.play_all_btn.setEnabled(False)
        self.play_all_btn.setStyleSheet(BUTTON_STYLE)
        controls_layout.addWidget(self.play_all_btn)

        self.save_btn = QPushButton("💾 Save All")
        self.save_btn.clicked.connect(self.save_walk)
        self.save_btn.setEnabled(False)
        self.save_btn.setStyleSheet(BUTTON_STYLE)
        controls_layout.addWidget(self.save_btn)

        preview_layout.addLayout(controls_layout)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()

    def generate_walk(self):
        """Generate a random walk through latent space."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded. Train or load a model first!")
            return

        n_steps = self.steps_spin.value()
        step_size = self.step_size_spin.value()
        momentum = self.momentum_spin.value()
        origin_pull = self.pull_spin.value()

        # Get starting point from latent widget if available
        start_vector = None
        if self.latent_widget:
            start_vector = self.latent_widget.get_vector()

        try:
            # Generate latent walk
            self.current_walk = self.generator.random_walk(
                start_vector=start_vector,
                n_steps=n_steps,
                step_size=step_size,
                momentum=momentum,
                origin_pull=origin_pull,
            )

            # Generate audio for each step
            self.current_audio = []
            for z in self.current_walk:
                audio = self.generator.generate_from_latent(z)
                self.current_audio.append(audio)

            # Setup preview
            self.current_index = 0
            self.preview_slider.setMaximum(len(self.current_walk) - 1)
            self.preview_slider.setValue(0)
            self.preview_slider.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.play_all_btn.setEnabled(True)
            self.save_btn.setEnabled(True)

            self.update_preview()
            self.status_label.setText(f"Generated {len(self.current_walk)}-step walk")

            QMessageBox.information(
                self,
                "Success",
                f"Generated {len(self.current_walk)}-step random walk!\nUse slider to explore each step.",
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate walk: {e}")

    def on_slider_changed(self, value):
        """Handle slider value change."""
        self.current_index = value
        self.update_preview()

    def update_preview(self):
        """Update the preview display."""
        if self.current_audio and 0 <= self.current_index < len(self.current_audio):
            self.preview_waveform.set_audio(self.current_audio[self.current_index])
            self.step_label.setText(f"{self.current_index + 1}/{len(self.current_walk)}")

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
        """Play all steps in sequence."""
        if not self.current_audio:
            return

        try:
            import sounddevice as sd

            # Concatenate all audio with gaps
            gap_samples = int(0.1 * self.generator.config.sample_rate)  # 100ms gap
            gap = np.zeros(gap_samples)

            full_sequence = []
            for audio in self.current_audio:
                full_sequence.extend([audio, gap])

            full_audio = np.concatenate(full_sequence)
            sd.play(full_audio, self.generator.config.sample_rate)

        except ImportError:
            QMessageBox.warning(self, "Error", "Install sounddevice: pip install sounddevice")

    def save_walk(self):
        """Save the walk sequence to files."""
        if not self.current_audio:
            return

        folder = QFileDialog.getExistingDirectory(self, "Save Walk Sequence")
        if not folder:
            return

        try:
            for i, audio in enumerate(self.current_audio):
                filename = f"{folder}/walk_{i:03d}.wav"
                self.generator.save_audio(audio, filename)

            QMessageBox.information(
                self, "Success", f"Saved {len(self.current_audio)} files to {folder}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save walk: {e}")

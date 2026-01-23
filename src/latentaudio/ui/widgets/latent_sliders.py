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
# latent_sliders.py - Latent vector editing widget with sliders
"""Latent vector editing widget with individual dimension sliders."""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QGroupBox,
    QDoubleSpinBox,
    QCheckBox,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
import numpy as np

from ..theme import (
    SLIDER_HEIGHT,
    BUTTON_STYLE,
    SLIDER_STYLE,
    GROUP_BOX_STYLE,
    TEXT_SECONDARY,
    NORMAL_FONT,
)


class LatentVectorWidget(QWidget):
    """Display and edit individual latent dimensions with sliders."""

    vector_changed = pyqtSignal(np.ndarray)

    def __init__(self, latent_dim=128, parent=None):
        super().__init__(parent)
        self.latent_dim = latent_dim
        self.current_vector = np.random.randn(latent_dim) * 0.5
        self.dimension_locks = [False] * min(128, latent_dim)  # Lock state for first 128 dimensions
        self.setup_ui()

    def setup_ui(self):
        """Set up the user interface."""
        layout = QVBoxLayout(self)

        # Control buttons
        controls = QHBoxLayout()

        self.randomize_btn = QPushButton("🎲 Random")
        self.randomize_btn.clicked.connect(self.randomize)
        self.randomize_btn.setStyleSheet(BUTTON_STYLE)
        controls.addWidget(self.randomize_btn)

        self.zero_btn = QPushButton("⊗ Zero")
        self.zero_btn.clicked.connect(self.zero_vector)
        self.zero_btn.setStyleSheet(BUTTON_STYLE)
        controls.addWidget(self.zero_btn)

        controls.addWidget(QLabel("Temp:"))
        self.temp_spin = QDoubleSpinBox()
        self.temp_spin.setRange(0.1, 3.0)
        self.temp_spin.setValue(1.0)
        self.temp_spin.setSingleStep(0.1)
        self.temp_spin.setFixedWidth(60)
        controls.addWidget(self.temp_spin)

        layout.addLayout(controls)

        # Info label
        self.info_label = QLabel(f"{self.latent_dim}D vector")
        self.info_label.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        self.info_label.setFont(NORMAL_FONT)
        layout.addWidget(self.info_label)

        # Scrollable sliders area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(SLIDER_HEIGHT)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Create widget to hold the scroll content
        scroll_widget = QWidget()
        sliders_layout = QVBoxLayout(scroll_widget)

        sliders_group = QGroupBox(f"Fine Control (all {self.latent_dim} dimensions)")
        sliders_group.setStyleSheet(GROUP_BOX_STYLE)
        slider_controls_layout = QVBoxLayout()

        self.sliders = []
        max_dims = min(
            self.latent_dim, self.latent_dim
        )  # Limit to 32 dimensions for UI performance

        for i in range(max_dims):
            # Extend dimension_locks if needed
            if i >= len(self.dimension_locks):
                self.dimension_locks.append(False)

            row = QHBoxLayout()

            # Lock checkbox
            lock_checkbox = QCheckBox("Lock")
            lock_checkbox.setChecked(self.dimension_locks[i])
            lock_checkbox.stateChanged.connect(
                lambda state, idx=i: self.on_lock_changed(idx, state)
            )
            row.addWidget(lock_checkbox)

            # Dimension label
            label = QLabel(f"z[{i:02d}]:")
            label.setMinimumWidth(60)
            label.setFont(NORMAL_FONT)
            row.addWidget(label)

            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-300)
            slider.setMaximum(300)
            slider.setValue(0)
            slider.valueChanged.connect(self.on_slider_changed)
            slider.setStyleSheet(SLIDER_STYLE)
            row.addWidget(slider)

            # Value label
            value_label = QLabel("0.00")
            value_label.setMinimumWidth(50)
            value_label.setFont(NORMAL_FONT)
            row.addWidget(value_label)

            self.sliders.append((slider, value_label, lock_checkbox, i))
            slider_controls_layout.addLayout(row)

        sliders_group.setLayout(slider_controls_layout)
        sliders_layout.addWidget(sliders_group)

        # Add scroll area to the main layout
        scroll_area.setWidget(scroll_widget)
        layout.addWidget(scroll_area)

    def on_lock_changed(self, idx, state):
        """Called when a lock checkbox changes state."""
        self.dimension_locks[idx] = state == 2  # Qt.CheckState.Checked == 2

    def on_slider_changed(self):
        """Called when any slider value changes."""
        for slider, label, lock_checkbox, idx in self.sliders:
            value = slider.value() / 100.0
            label.setText(f"{value:.2f}")
            self.current_vector[idx] = value

        self.vector_changed.emit(self.current_vector.copy())

    def randomize(self):
        """Randomize unlocked dimensions."""
        temp = self.temp_spin.value()

        # Generate new random values for unlocked dimensions only
        for i in range(min(128, self.latent_dim)):
            if not self.dimension_locks[i]:  # Only randomize if NOT locked
                self.current_vector[i] = np.random.randn() * temp

        # Update ALL sliders
        for slider, label, lock_checkbox, idx in self.sliders:
            if idx < len(self.current_vector):  # Safety check
                value = int(self.current_vector[idx] * 100)
                slider.blockSignals(True)  # Prevent triggering valueChanged
                slider.setValue(np.clip(value, -300, 300))
                slider.blockSignals(False)
                label.setText(f"{self.current_vector[idx]:.2f}")

        self.vector_changed.emit(self.current_vector.copy())

    def zero_vector(self):
        """Set all dimensions to zero."""
        self.current_vector = np.zeros(self.latent_dim)

        for slider, label, lock_checkbox, idx in self.sliders:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)
            label.setText("0.00")

        self.vector_changed.emit(self.current_vector.copy())

    def set_vector(self, z: np.ndarray):
        """Set the current vector to the given values."""
        self.current_vector = z.copy()

        for slider, label, lock_checkbox, idx in self.sliders:
            if idx < len(z):
                value = int(z[idx] * 100)
                slider.blockSignals(True)
                slider.setValue(np.clip(value, -300, 300))
                slider.blockSignals(False)
                label.setText(f"{z[idx]:.2f}")

        self.vector_changed.emit(self.current_vector.copy())

    def get_vector(self) -> np.ndarray:
        """Get the current vector values."""
        return self.current_vector.copy()

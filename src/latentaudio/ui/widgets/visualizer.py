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
# visualizer.py - Waveform visualization widget
"""Waveform visualization widget with real-time audio display."""

from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor
import numpy as np

from ..theme import (
    BG_DARK,
    ACCENT_PRIMARY,
    GRID_COLOR,
    CENTER_LINE_COLOR,
    WAVEFORM_HEIGHT_MIN,
    WAVEFORM_HEIGHT_MAX,
    FONT_SIZE_SMALL,
    get_font,
)


class WaveformVisualizer(QWidget):
    """Real-time waveform visualization widget."""

    def __init__(self, color=None, parent=None):
        super().__init__(parent)
        self.audio_data = None
        self.color = color if color else ACCENT_PRIMARY
        self.setMinimumHeight(WAVEFORM_HEIGHT_MIN)
        self.setMaximumHeight(WAVEFORM_HEIGHT_MAX)

    def set_color(self, color: QColor):
        """Update the waveform color."""
        self.color = color
        self.update()

    def set_audio(self, audio: np.ndarray):
        """Update the waveform display with new audio data."""
        self.audio_data = audio
        self.update()

    def clear(self):
        """Clear the waveform display."""
        self.audio_data = None
        self.update()

    def paintEvent(self, event):
        """Paint the waveform visualization."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QBrush(BG_DARK))

        if self.audio_data is None or len(self.audio_data) == 0:
            # Draw "No audio" message
            painter.setPen(QPen(GRID_COLOR, 1))
            painter.drawText(event.rect(), Qt.AlignmentFlag.AlignCenter, "No audio generated yet")
            return

        center_y = height // 2

        # Draw grid lines
        painter.setPen(QPen(GRID_COLOR, 1))
        for i in range(5):
            y = int(height * i / 4)
            painter.drawLine(0, y, width, y)

        # Draw center line (thicker)
        painter.setPen(QPen(CENTER_LINE_COLOR, 2))
        painter.drawLine(0, center_y, width, center_y)

        # Draw waveform
        painter.setPen(QPen(self.color, 2))

        # Downsample audio to fit width
        step = max(1, len(self.audio_data) // width)

        points = []
        for i in range(0, len(self.audio_data) - step, step):
            x = int(i / len(self.audio_data) * width)

            # Get max amplitude in this window for better visibility
            window = self.audio_data[i : i + step]
            sample = np.max(np.abs(window)) * np.sign(np.mean(window))

            y = int(center_y - (sample * center_y * 0.85))
            points.append((x, y))

        # Draw connected lines
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            painter.drawLine(x1, y1, x2, y2)

        # Draw stats
        painter.setPen(QPen(QColor(150, 150, 150), 1))
        font = get_font(FONT_SIZE_SMALL)
        painter.setFont(font)

        peak = np.max(np.abs(self.audio_data))
        rms = np.sqrt(np.mean(self.audio_data**2))

        stats_text = f"Peak: {peak:.3f} | RMS: {rms:.3f} | Samples: {len(self.audio_data)}"
        painter.drawText(5, height - 5, stats_text)

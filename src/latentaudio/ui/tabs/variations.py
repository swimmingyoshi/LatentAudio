# variations.py - Latent variations exploration tab
"""Tab for exploring local variations around a central latent vector."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QSlider, QGridLayout, QMessageBox, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal

from ..widgets.visualizer import WaveformVisualizer
from ..theme import (
    BUTTON_STYLE, GROUP_BOX_STYLE, TEXT_SECONDARY, NORMAL_FONT, ACCENT_PRIMARY
)


class VariationCell(QFrame):
    """A single cell in the variation grid."""
    
    play_requested = pyqtSignal(np.ndarray)
    select_requested = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("VariationCell { border: 1px solid #333; border-radius: 4px; background: #1a1a1a; }")
        
        self.audio = None
        self.vector = None
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        
        self.visualizer = WaveformVisualizer()
        self.visualizer.setMinimumHeight(80)
        layout.addWidget(self.visualizer)
        
        btn_layout = QHBoxLayout()
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(30)
        self.play_btn.clicked.connect(self.on_play)
        self.play_btn.setEnabled(False)
        btn_layout.addWidget(self.play_btn)
        
        self.select_btn = QPushButton("🎯 Center")
        self.select_btn.clicked.connect(self.on_select)
        self.select_btn.setEnabled(False)
        self.select_btn.setFont(NORMAL_FONT)
        btn_layout.addWidget(self.select_btn)
        
        layout.addLayout(btn_layout)

    def set_data(self, audio, vector):
        self.audio = audio
        self.vector = vector
        self.visualizer.set_audio(audio)
        self.play_btn.setEnabled(True)
        self.select_btn.setEnabled(True)

    def on_play(self):
        if self.audio is not None:
            self.play_requested.emit(self.audio)

    def on_select(self):
        if self.vector is not None:
            self.select_requested.emit(self.vector)


class VariationsTab(QWidget):
    """Tab for exploring local variations in latent space."""

    # Signal to update the main latent sliders
    vector_selected = pyqtSignal(np.ndarray)

    def __init__(self, generator=None, latent_widget=None, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.latent_widget = latent_widget
        
        self.center_vector = None
        self.variation_vectors = []
        self.variation_audios = []
        
        self.setup_ui()

    def setup_ui(self):
        """Set up the variations interface."""
        layout = QVBoxLayout(self)

        # Controls
        controls_group = QGroupBox("Grid Settings")
        controls_group.setStyleSheet(GROUP_BOX_STYLE)
        controls_layout = QVBoxLayout()

        # Row 1: Source
        source_layout = QHBoxLayout()
        self.sync_btn = QPushButton("🔄 Sync from Sliders")
        self.sync_btn.clicked.connect(self.sync_from_sliders)
        self.sync_btn.setStyleSheet(BUTTON_STYLE)
        source_layout.addWidget(self.sync_btn)
        
        self.random_btn = QPushButton("🎲 Random Center")
        self.random_btn.clicked.connect(self.random_center)
        self.random_btn.setStyleSheet(BUTTON_STYLE)
        source_layout.addWidget(self.random_btn)
        controls_layout.addLayout(source_layout)

        # Row 2: Deviation
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("Variation Intensity:"))
        self.dev_slider = QSlider(Qt.Orientation.Horizontal)
        self.dev_slider.setRange(1, 100)
        self.dev_slider.setValue(20)
        dev_layout.addWidget(self.dev_slider)
        
        self.dev_label = QLabel("0.20")
        self.dev_label.setFixedWidth(40)
        self.dev_slider.valueChanged.connect(lambda v: self.dev_label.setText(f"{v/100:.2f}"))
        dev_layout.addWidget(self.dev_label)
        controls_layout.addLayout(dev_layout)

        # Row 3: Generate
        self.generate_btn = QPushButton("✨ Generate Grid")
        self.generate_btn.clicked.connect(self.generate_grid)
        self.generate_btn.setStyleSheet(BUTTON_STYLE + "font-weight: bold; background-color: #2e4a3e;")
        controls_layout.addWidget(self.generate_btn)

        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)

        # Grid area
        self.grid_layout = QGridLayout()
        self.cells = []
        for r in range(3):
            for c in range(3):
                cell = VariationCell()
                cell.play_requested.connect(self.play_audio)
                cell.select_requested.connect(self.on_cell_selected)
                self.grid_layout.addWidget(cell, r, c)
                self.cells.append(cell)
        
        layout.addLayout(self.grid_layout)
        layout.addStretch()

    def sync_from_sliders(self):
        """Get the current vector from main UI."""
        if self.latent_widget:
            self.center_vector = self.latent_widget.get_vector()
            QMessageBox.information(self, "Sync", "Center vector synced from sliders!")

    def random_center(self):
        """Generate a random center vector."""
        if not self.generator:
            return
        dim = self.generator.config.latent_dim
        self.center_vector = np.random.randn(dim) * 0.5
        self.generate_grid()

    def generate_grid(self):
        """Generate 9 variations and update the grid."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded!")
            return

        if self.center_vector is None:
            if self.latent_widget:
                self.center_vector = self.latent_widget.get_vector()
            else:
                dim = self.generator.config.latent_dim
                self.center_vector = np.zeros(dim)

        deviation = self.dev_slider.value() / 100.0
        dim = len(self.center_vector)
        
        try:
            for i, cell in enumerate(self.cells):
                # The middle cell (index 4) is the exact center
                if i == 4:
                    z = self.center_vector.copy()
                else:
                    # Random variation
                    noise = np.random.randn(dim)
                    z = self.center_vector + noise * deviation
                
                audio = self.generator.generate_from_latent(z)
                cell.set_data(audio, z)
                
                # Highlight center cell
                if i == 4:
                    cell.setStyleSheet("VariationCell { border: 2px solid #00FF00; border-radius: 4px; background: #1a2a1a; }")
                else:
                    cell.setStyleSheet("VariationCell { border: 1px solid #333; border-radius: 4px; background: #1a1a1a; }")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Grid generation failed: {e}")

    def on_cell_selected(self, vector):
        """Handle 'Set as Center' click."""
        self.center_vector = vector.copy()
        self.vector_selected.emit(vector)
        self.generate_grid()

    def play_audio(self, audio):
        """Play given audio array."""
        if audio is not None:
            try:
                import sounddevice as sd
                sd.play(audio, self.generator.config.sample_rate)
            except ImportError:
                QMessageBox.warning(self, "Error", "Install sounddevice: pip install sounddevice")

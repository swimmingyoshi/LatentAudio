# attributes.py - Feature Discovery / Latent Arithmetic Tab
"""Interface for discovering and applying latent directions."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QListWidget, QLineEdit, QDoubleSpinBox, QMessageBox,
    QProgressBar, QListWidgetItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from typing import List, Optional

from ..theme import (
    BUTTON_STYLE, GROUP_BOX_STYLE, TEXT_SECONDARY, NORMAL_FONT
)

class AttributesTab(QWidget):
    """
    Tab for 'Latent Arithmetic'.
    
    Users can select positive and negative samples to calculate a 
    directional vector representing a specific sonic quality.
    """
    
    direction_discovered = pyqtSignal(str)  # Emits name of direction
    apply_requested = pyqtSignal()  # Emits when an attribute is applied
    
    def __init__(self, generator=None, latent_widget=None, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.latent_widget = latent_widget
        
        self.pos_vectors = []
        self.neg_vectors = []
        
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Info
        info = QLabel("Discover sonic qualities by comparing groups of sounds")
        info.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        info.setFont(NORMAL_FONT)
        layout.addWidget(info)
        
        # Comparison Workspace
        work_layout = QHBoxLayout()
        
        # Positive Samples
        pos_group = QGroupBox("Target Quality Samples (Positive)")
        pos_group.setStyleSheet(GROUP_BOX_STYLE)
        pos_layout = QVBoxLayout()
        self.pos_list = QListWidget()
        pos_layout.addWidget(self.pos_list)
        
        pos_btns = QHBoxLayout()
        self.add_pos_btn = QPushButton("+ Add Current")
        self.add_pos_btn.clicked.connect(self.add_positive)
        pos_btns.addWidget(self.add_pos_btn)
        
        self.clear_pos_btn = QPushButton("Clear")
        self.clear_pos_btn.clicked.connect(self.clear_positive)
        pos_btns.addWidget(self.clear_pos_btn)
        
        pos_layout.addLayout(pos_btns)
        pos_group.setLayout(pos_layout)
        work_layout.addWidget(pos_group)
        
        # Negative Samples
        neg_group = QGroupBox("Opposite Quality Samples (Negative)")
        neg_group.setStyleSheet(GROUP_BOX_STYLE)
        neg_layout = QVBoxLayout()
        self.neg_list = QListWidget()
        neg_layout.addWidget(self.neg_list)
        
        neg_btns = QHBoxLayout()
        self.add_neg_btn = QPushButton("+ Add Current")
        self.add_neg_btn.clicked.connect(self.add_negative)
        neg_btns.addWidget(self.add_neg_btn)
        
        self.clear_neg_btn = QPushButton("Clear")
        self.clear_neg_btn.clicked.connect(self.clear_negative)
        neg_btns.addWidget(self.clear_neg_btn)
        
        neg_layout.addLayout(neg_btns)
        neg_group.setLayout(neg_layout)
        work_layout.addWidget(neg_group)
        
        layout.addLayout(work_layout)
        
        # Discovery Controls
        discovery_group = QGroupBox("Discovery")
        discovery_group.setStyleSheet(GROUP_BOX_STYLE)
        discovery_layout = QVBoxLayout()
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Attribute Name:"))
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("e.g. Brightness, Crunch, Airy")
        name_layout.addWidget(self.name_edit)
        discovery_layout.addLayout(name_layout)
        
        self.discover_btn = QPushButton("✨ Discover Latent Direction")
        self.discover_btn.clicked.connect(self.discover_direction)
        self.discover_btn.setStyleSheet(BUTTON_STYLE)
        discovery_layout.addWidget(self.discover_btn)
        
        discovery_group.setLayout(discovery_layout)
        layout.addWidget(discovery_group)
        
        # Discovered Attributes (Applied Sliders)
        self.attr_group = QGroupBox("Apply Attributes")
        self.attr_group.setStyleSheet(GROUP_BOX_STYLE)
        self.attr_layout = QVBoxLayout()
        self.attr_group.setLayout(self.attr_layout)
        layout.addWidget(self.attr_group)
        
        layout.addStretch()

    def add_positive(self):
        if not self.latent_widget: return
        vector = self.latent_widget.get_vector()
        self.pos_vectors.append(vector.copy())
        self.pos_list.addItem(f"Sample {len(self.pos_vectors)}")

    def add_negative(self):
        if not self.latent_widget: return
        vector = self.latent_widget.get_vector()
        self.neg_vectors.append(vector.copy())
        self.neg_list.addItem(f"Sample {len(self.neg_vectors)}")

    def clear_positive(self):
        self.pos_vectors = []
        self.pos_list.clear()

    def clear_negative(self):
        self.neg_vectors = []
        self.neg_list.clear()

    def discover_direction(self):
        name = self.name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Error", "Please enter a name for the attribute.")
            return
        
        if not self.pos_vectors or not self.neg_vectors:
            QMessageBox.warning(self, "Error", "Please provide both positive and negative samples.")
            return
            
        try:
            direction = self.generator.discover_direction(name, self.pos_vectors, self.neg_vectors)
            self.add_attribute_control(name)
            self.direction_discovered.emit(name)
            QMessageBox.information(self, "Success", f"Discovered direction: {name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Discovery failed: {e}")

    def add_attribute_control(self, name: str):
        row = QHBoxLayout()
        row.addWidget(QLabel(f"<b>{name}</b>"))
        
        slider = QDoubleSpinBox()
        slider.setRange(-10.0, 10.0)
        slider.setValue(0.0)
        slider.setSingleStep(0.5)
        
        apply_btn = QPushButton("Apply")
        
        def apply_attr():
            if not self.generator or not self.latent_widget: return
            base = self.latent_widget.get_vector()
            new_v = self.generator.apply_attribute(base, name, slider.value())
            self.latent_widget.set_vector(new_v)
            self.apply_requested.emit()
                
        apply_btn.clicked.connect(apply_attr)
        
        row.addWidget(slider)
        row.addWidget(apply_btn)
        self.attr_layout.addLayout(row)

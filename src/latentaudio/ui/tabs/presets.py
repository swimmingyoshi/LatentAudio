# presets.py - Preset management tab widget
"""Preset management tab for saving and loading latent vectors."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,
    QListWidgetItem, QInputDialog, QMessageBox, QLabel
)
from PyQt6.QtCore import pyqtSignal

from ..theme import (
    BUTTON_STYLE, LIST_WIDGET_STYLE, TEXT_SECONDARY, NORMAL_FONT
)


class PresetManager(QWidget):
    """Manage saved latent presets."""

    preset_selected = pyqtSignal(object)  # LatentVector

    def __init__(self, generator=None, latent_widget=None, parent=None):
        super().__init__(parent)
        self.generator = generator
        self.latent_widget = latent_widget
        self.setup_ui()

    def setup_ui(self):
        """Set up the preset management interface."""
        layout = QVBoxLayout(self)

        # Instructions
        info = QLabel("Save and manage favorite sounds")
        info.setStyleSheet(f"color: {TEXT_SECONDARY.name()};")
        info.setFont(NORMAL_FONT)
        layout.addWidget(info)

        # Preset list
        self.preset_list = QListWidget()
        self.preset_list.itemDoubleClicked.connect(self.load_preset)
        self.preset_list.setStyleSheet(LIST_WIDGET_STYLE)
        layout.addWidget(self.preset_list)

        # Buttons
        btn_layout = QHBoxLayout()

        self.save_btn = QPushButton("💾 Save Current")
        self.save_btn.clicked.connect(self.save_preset)
        self.save_btn.setStyleSheet(BUTTON_STYLE)
        btn_layout.addWidget(self.save_btn)

        self.delete_btn = QPushButton("🗑️ Delete")
        self.delete_btn.clicked.connect(self.delete_preset)
        self.delete_btn.setStyleSheet(BUTTON_STYLE)
        btn_layout.addWidget(self.delete_btn)

        layout.addLayout(btn_layout)

        self.refresh_list()

    def refresh_list(self):
        """Refresh the preset list from the generator."""
        self.preset_list.clear()
        if self.generator:
            for name in self.generator.list_presets():
                item = QListWidgetItem(name)
                preset_info = self.generator.get_preset_info(name)
                if preset_info and preset_info.description:
                    item.setToolTip(preset_info.description)
                self.preset_list.addItem(item)

    def save_preset(self):
        """Save current latent vector as a preset."""
        if not self.generator:
            QMessageBox.warning(self, "Error", "No generator available")
            return

        # Get current vector from the latent widget
        current_vector = None
        if self.latent_widget:
            current_vector = self.latent_widget.get_vector()

        if current_vector is None:
            QMessageBox.warning(self, "Error", "No current vector to save")
            return

        name, ok = QInputDialog.getText(self, 'Save Preset', 'Preset name:')
        if ok and name:
            try:
                self.generator.save_preset(name, current_vector)
                self.refresh_list()
                QMessageBox.information(self, "Success", f"Preset '{name}' saved!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save preset: {e}")

    def load_preset(self, item):
        """Load a preset by double-clicking."""
        name = item.text()
        try:
            vector = self.generator.get_preset(name)
            if vector is not None:
                self.preset_selected.emit(vector)
            else:
                QMessageBox.warning(self, "Error", f"Preset '{name}' not found")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load preset: {e}")

    def delete_preset(self):
        """Delete the currently selected preset."""
        current_item = self.preset_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Info", "Please select a preset to delete")
            return

        name = current_item.text()

        reply = QMessageBox.question(
            self, 'Delete Preset',
            f"Are you sure you want to delete preset '{name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                if self.generator.delete_preset(name):
                    self.refresh_list()
                    QMessageBox.information(self, "Success", f"Preset '{name}' deleted!")
                else:
                    QMessageBox.warning(self, "Error", f"Failed to delete preset '{name}'")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete preset: {e}")
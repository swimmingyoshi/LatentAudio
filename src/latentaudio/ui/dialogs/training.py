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
# training.py - Training configuration dialog with ANTI-COLLAPSE presets
"""Training configuration dialog for setting up model training."""

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QPushButton,
    QLabel,
    QDialogButtonBox,
    QCheckBox,
)
from PyQt6.QtCore import Qt

from ..theme import (
    BUTTON_STYLE,
    GROUP_BOX_STYLE,
    TEXT_SECONDARY,
    NORMAL_FONT,
    TITLE_FONT,
    ACCENT_PRIMARY,
)
from ...config import (
    DEFAULT_EPOCHS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_BETA_KL,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_GRAD_CLIP,
    DEFAULT_SCHEDULER_PATIENCE,
    DEFAULT_SCHEDULER_FACTOR,
    LOG_BATCH_INTERVAL,
    STFT_LOSS_SKIP_INTERVAL,
    LATENT_DIM,
    DEFAULT_GRADIENT_ACCUMULATION_STEPS,
)


class TrainingDialog(QDialog):
    """Dialog for configuring training parameters."""

    def __init__(self, n_samples=0, parent=None, resume_from=None):
        super().__init__(parent)
        self.n_samples = n_samples
        self.resume_from = resume_from
        self.setup_ui()

    def setup_ui(self):
        """Set up the dialog interface."""
        self.setWindowTitle("Training Configuration" + (" - Resuming" if self.resume_from else ""))
        self.setMinimumWidth(500)
        self.resize(600, 750)

        layout = QVBoxLayout(self)

        # Header with dataset info
        header_text = f"Dataset: {self.n_samples} samples loaded"
        if self.resume_from:
            total_trained = self.resume_from.get("total_epochs_trained", 0)
            best_loss = self.resume_from.get("best_loss", float("inf"))
            header_text += f"\n\n▶️ CONTINUING TRAINING\nStarting from absolute epoch {total_trained + 1}\nBest previous loss: {best_loss:.6f}"

        info_label = QLabel(header_text)
        info_label.setFont(TITLE_FONT)
        info_label.setStyleSheet(f"color: {ACCENT_PRIMARY.name()}; padding: 10px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # Basic settings
        basic_group = QGroupBox("Basic Settings")
        basic_group.setStyleSheet(GROUP_BOX_STYLE)
        basic_layout = QFormLayout()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(50, 5000)
        self.epochs_spin.setValue(DEFAULT_EPOCHS)
        self.epochs_spin.setSingleStep(100)
        basic_layout.addRow("Epochs:", self.epochs_spin)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, min(128, max(1, self.n_samples)))
        self.batch_spin.setValue(min(DEFAULT_BATCH_SIZE, max(1, self.n_samples)))
        basic_layout.addRow("Batch Size:", self.batch_spin)

        self.batch_log_spin = QSpinBox()
        self.batch_log_spin.setRange(0, 100)
        self.batch_log_spin.setValue(LOG_BATCH_INTERVAL)
        self.batch_log_spin.setSpecialValueText("Disabled")
        basic_layout.addRow("Log Every N Batches:", self.batch_log_spin)

        self.stft_skip_spin = QSpinBox()
        self.stft_skip_spin.setRange(1, 20)
        self.stft_skip_spin.setValue(STFT_LOSS_SKIP_INTERVAL)
        basic_layout.addRow("STFT Skip Interval:", self.stft_skip_spin)

        basic_group.setLayout(basic_layout)
        layout.addWidget(basic_group)

        # Model Architecture settings
        model_group = QGroupBox("Model Architecture")
        model_group.setStyleSheet(GROUP_BOX_STYLE)
        model_layout = QFormLayout()

        self.latent_dim_spin = QSpinBox()
        self.latent_dim_spin.setRange(8, 128)
        self.latent_dim_spin.setValue(LATENT_DIM)
        model_layout.addRow("Latent Dimension:", self.latent_dim_spin)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Optimizer settings
        optimizer_group = QGroupBox("Optimizer Settings")
        optimizer_group.setStyleSheet(GROUP_BOX_STYLE)
        optimizer_layout = QFormLayout()

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 0.1)
        self.lr_spin.setValue(DEFAULT_LEARNING_RATE)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        optimizer_layout.addRow("Learning Rate:", self.lr_spin)

        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.01)
        self.weight_decay_spin.setValue(DEFAULT_WEIGHT_DECAY)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setSingleStep(0.00001)
        optimizer_layout.addRow("Weight Decay:", self.weight_decay_spin)

        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.1, 10.0)
        self.grad_clip_spin.setValue(DEFAULT_GRAD_CLIP)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setSingleStep(0.1)
        optimizer_layout.addRow("Gradient Clipping:", self.grad_clip_spin)

        self.grad_accum_spin = QSpinBox()
        self.grad_accum_spin.setRange(1, 32)
        self.grad_accum_spin.setValue(DEFAULT_GRADIENT_ACCUMULATION_STEPS)
        optimizer_layout.addRow("Grad Accumulation:", self.grad_accum_spin)

        optimizer_group.setLayout(optimizer_layout)
        layout.addWidget(optimizer_group)

        # VAE settings - ANTI-COLLAPSE VERSION
        vae_group = QGroupBox("VAE Settings (Anti-Collapse)")
        vae_group.setStyleSheet(GROUP_BOX_STYLE)
        vae_layout = QFormLayout()

        self.beta_kl_spin = QDoubleSpinBox()
        self.beta_kl_spin.setRange(0.0, 0.5)
        self.beta_kl_spin.setValue(0.002)  # NEW DEFAULT - stable latent space
        self.beta_kl_spin.setDecimals(6)
        self.beta_kl_spin.setSingleStep(0.001)
        vae_layout.addRow("Beta KL (β):", self.beta_kl_spin)

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 1000)
        self.warmup_spin.setValue(100)  # NEW DEFAULT - slower warmup
        vae_layout.addRow("KL Warmup Epochs:", self.warmup_spin)

        # Add cyclical annealing option
        self.cyclical_checkbox = QCheckBox("Use Cyclical Annealing (exploration)")
        self.cyclical_checkbox.setChecked(False)
        vae_layout.addRow("", self.cyclical_checkbox)

        vae_group.setLayout(vae_layout)
        layout.addWidget(vae_group)

        # Scheduler settings
        scheduler_group = QGroupBox("Learning Rate Scheduler")
        scheduler_group.setStyleSheet(GROUP_BOX_STYLE)
        scheduler_layout = QFormLayout()

        self.scheduler_patience_spin = QSpinBox()
        self.scheduler_patience_spin.setRange(5, 500)
        self.scheduler_patience_spin.setValue(DEFAULT_SCHEDULER_PATIENCE)
        scheduler_layout.addRow("Patience:", self.scheduler_patience_spin)

        self.scheduler_factor_spin = QDoubleSpinBox()
        self.scheduler_factor_spin.setRange(0.1, 0.9)
        self.scheduler_factor_spin.setValue(DEFAULT_SCHEDULER_FACTOR)
        self.scheduler_factor_spin.setDecimals(2)
        self.scheduler_factor_spin.setSingleStep(0.1)
        scheduler_layout.addRow("Reduction Factor:", self.scheduler_factor_spin)

        scheduler_group.setLayout(scheduler_layout)
        layout.addWidget(scheduler_group)

        # Quick presets - ANTI-COLLAPSE VERSIONS
        presets_group = QGroupBox("Quick Presets (Anti-Collapse)")
        presets_group.setStyleSheet(GROUP_BOX_STYLE)
        presets_layout = QVBoxLayout()

        presets_row1 = QHBoxLayout()
        test_btn = QPushButton("🧪 Test (50 epochs)")
        test_btn.setToolTip("Quick test: 50 epochs, 16D latent, fast warmup")
        test_btn.clicked.connect(lambda: self.apply_preset("test"))
        presets_row1.addWidget(test_btn)

        simple_btn = QPushButton("⚡ Simple (300 epochs)")
        simple_btn.setToolTip("Light training: 300 epochs, 64D latent")
        simple_btn.clicked.connect(lambda: self.apply_preset("simple"))
        presets_row1.addWidget(simple_btn)

        fast_btn = QPushButton("⚡ Fast (500 epochs)")
        fast_btn.setToolTip("Fast quality: 500 epochs, 128D latent, β=0.005")
        fast_btn.clicked.connect(lambda: self.apply_preset("fast"))
        presets_row1.addWidget(fast_btn)

        presets_layout.addLayout(presets_row1)

        presets_row2 = QHBoxLayout()
        balanced_btn = QPushButton("⚖️ Balanced (1000 epochs) ⭐")
        balanced_btn.setToolTip("RECOMMENDED: 1000 epochs, 128D, β=0.005, 30 epoch warmup")
        balanced_btn.clicked.connect(lambda: self.apply_preset("balanced"))
        presets_row2.addWidget(balanced_btn)

        quality_btn = QPushButton("💎 Quality (2000 epochs)")
        quality_btn.setToolTip("High quality: 2000 epochs, slower STFT")
        quality_btn.clicked.connect(lambda: self.apply_preset("quality"))
        presets_row2.addWidget(quality_btn)

        diverse_btn = QPushButton("🎨 Diverse (high β=0.008)")
        diverse_btn.setToolTip("Maximum diversity: Higher beta, cyclical annealing")
        diverse_btn.clicked.connect(lambda: self.apply_preset("diverse"))
        presets_row2.addWidget(diverse_btn)

        presets_layout.addLayout(presets_row2)
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Help text - UPDATED
        help_text = QLabel(
            "💡 Anti-Collapse Tips:\n"
            "• ⭐ START WITH 'BALANCED' PRESET (recommended)\n"
            "• Beta=0.002 prevents collapse for SimpleVAE\n"
            "• 100 epoch warmup prevents early latent death\n"
            "• Use 'Diverse' if sounds are too similar (β=0.004)\n"
            "• Use 'Quality' for final long training runs\n"
            "• Monitor TensorBoard for 'Active Dims' warnings\n"
            "• Healthy KL should stay above 10-20 after warmup"
        )
        help_text.setStyleSheet(f"color: {TEXT_SECONDARY.name()}; padding: 10px; font-size: 10px;")
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        # Dialog buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def apply_preset(self, preset_name: str):
        """Apply ANTI-COLLAPSE preset configurations."""

        # ALL PRESETS: Use new anti-collapse defaults
        BASE_LR = DEFAULT_LEARNING_RATE  # 0.0001
        BASE_WD = 0.001
        BASE_GRAD_CLIP = DEFAULT_GRAD_CLIP  # 0.5
        BASE_GRAD_ACCUM = DEFAULT_GRADIENT_ACCUMULATION_STEPS  # 2

        if preset_name == "test":
            # Quick smoke test
            self.epochs_spin.setValue(50)
            self.batch_spin.setValue(DEFAULT_BATCH_SIZE)
            self.latent_dim_spin.setValue(16)
            self.lr_spin.setValue(BASE_LR)
            self.beta_kl_spin.setValue(0.001)
            self.weight_decay_spin.setValue(BASE_WD)
            self.warmup_spin.setValue(10)  # Fast warmup for testing
            self.stft_skip_spin.setValue(10)  # Skip more for speed
            self.grad_accum_spin.setValue(BASE_GRAD_ACCUM)
            self.grad_clip_spin.setValue(BASE_GRAD_CLIP)
            self.cyclical_checkbox.setChecked(False)

        elif preset_name == "simple":
            # Light training
            self.epochs_spin.setValue(300)
            self.batch_spin.setValue(DEFAULT_BATCH_SIZE)
            self.latent_dim_spin.setValue(64)
            self.lr_spin.setValue(BASE_LR)
            self.beta_kl_spin.setValue(0.002)
            self.weight_decay_spin.setValue(BASE_WD)
            self.warmup_spin.setValue(100)
            self.stft_skip_spin.setValue(5)
            self.grad_accum_spin.setValue(BASE_GRAD_ACCUM)
            self.grad_clip_spin.setValue(BASE_GRAD_CLIP)
            self.cyclical_checkbox.setChecked(False)

        elif preset_name == "fast":
            # Fast but quality
            self.epochs_spin.setValue(500)
            self.batch_spin.setValue(DEFAULT_BATCH_SIZE)
            self.latent_dim_spin.setValue(128)
            self.lr_spin.setValue(BASE_LR)
            self.beta_kl_spin.setValue(0.002)
            self.weight_decay_spin.setValue(BASE_WD)
            self.warmup_spin.setValue(100)
            self.stft_skip_spin.setValue(5)
            self.grad_accum_spin.setValue(BASE_GRAD_ACCUM)
            self.grad_clip_spin.setValue(BASE_GRAD_CLIP)
            self.cyclical_checkbox.setChecked(False)

        elif preset_name == "balanced":
            # ⭐ RECOMMENDED DEFAULT
            self.epochs_spin.setValue(1000)
            self.batch_spin.setValue(DEFAULT_BATCH_SIZE)
            self.latent_dim_spin.setValue(128)
            self.lr_spin.setValue(BASE_LR)
            self.beta_kl_spin.setValue(0.002)  # Anti-collapse beta
            self.weight_decay_spin.setValue(BASE_WD)
            self.warmup_spin.setValue(100)  # Slower warmup
            self.stft_skip_spin.setValue(5)
            self.grad_accum_spin.setValue(BASE_GRAD_ACCUM)
            self.grad_clip_spin.setValue(BASE_GRAD_CLIP)
            self.cyclical_checkbox.setChecked(False)

        elif preset_name == "quality":
            # Long high-quality training
            self.epochs_spin.setValue(2000)
            self.batch_spin.setValue(DEFAULT_BATCH_SIZE)
            self.latent_dim_spin.setValue(128)
            self.lr_spin.setValue(BASE_LR)
            self.beta_kl_spin.setValue(0.003)  # Slightly higher for long training
            self.weight_decay_spin.setValue(BASE_WD)
            self.warmup_spin.setValue(200)
            self.stft_skip_spin.setValue(2)  # More frequent STFT
            self.grad_accum_spin.setValue(BASE_GRAD_ACCUM)
            self.grad_clip_spin.setValue(BASE_GRAD_CLIP)
            self.cyclical_checkbox.setChecked(False)

        elif preset_name == "diverse":
            # Maximum diversity - highest safe beta
            self.epochs_spin.setValue(1000)
            self.batch_spin.setValue(DEFAULT_BATCH_SIZE)
            self.latent_dim_spin.setValue(128)
            self.lr_spin.setValue(BASE_LR)
            self.beta_kl_spin.setValue(0.004)  # Higher beta for diversity
            self.weight_decay_spin.setValue(BASE_WD)
            self.warmup_spin.setValue(100)
            self.stft_skip_spin.setValue(5)
            self.grad_accum_spin.setValue(BASE_GRAD_ACCUM)
            self.grad_clip_spin.setValue(BASE_GRAD_CLIP)
            self.cyclical_checkbox.setChecked(True)  # Enable cyclical annealing

    def get_config(self):
        """Get training configuration from dialog values."""
        from latentaudio.types import TrainingConfig

        return TrainingConfig(
            epochs=self.epochs_spin.value(),
            batch_size=self.batch_spin.value(),
            learning_rate=self.lr_spin.value(),
            beta_kl=self.beta_kl_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
            scheduler_patience=self.scheduler_patience_spin.value(),
            scheduler_factor=self.scheduler_factor_spin.value(),
            grad_clip=self.grad_clip_spin.value(),
            latent_dim=self.latent_dim_spin.value(),
            batch_log_interval=self.batch_log_spin.value(),
            stft_skip_interval=self.stft_skip_spin.value(),
            kl_warmup_epochs=self.warmup_spin.value(),
            gradient_accumulation_steps=self.grad_accum_spin.value(),
            use_cyclical_annealing=self.cyclical_checkbox.isChecked(),
        )

# app.py - Main LatentAudio application window
"""Main application window for LatentAudio exploration."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QTabWidget,
    QFileDialog, QMessageBox, QPushButton, QLabel, QProgressDialog,
    QStackedWidget, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject

import numpy as np
from ..logging import logger
from ..core.generator import AdvancedAudioGenerator
from ..types import GeneratorConfig
from .widgets import (
    WaveformVisualizer, LatentVectorWidget,
    GenerationControls, PlaybackControls, StatusWidget,
    SoundMapWidget
)
from .tabs import PresetManager, MorphTab, WalkTab, ReconstructionTab, VariationsTab, AttributesTab
from .dialogs import TrainingDialog
from .theme import (
    WINDOW_TITLE, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    DEFAULT_SPLITTER_SIZES, TITLE_FONT, BUTTON_STYLE
)


class TrainingWorker(QObject):
    """Worker class for running training in a separate thread."""

    progress_updated = pyqtSignal(int, str)
    training_finished = pyqtSignal(bool, str)
    epoch_completed = pyqtSignal(int, float, float, float, float)
    training_started = pyqtSignal()

    def __init__(self, generator, samples, config, resume_from=None):
        super().__init__()
        self.generator = generator
        self.samples = samples
        self.config = config
        self.resume_from = resume_from
        self._cancel_requested = False

        self.generator._training_started_callback = self.emit_training_started

    def emit_training_started(self):
        self.training_started.emit()

    def request_cancel(self):
        self._cancel_requested = True

    def run_training(self):
        self._cancel_requested = False

        try:
            original_callback = self.config.callback

            def progress_callback(epoch, loss, metrics):
                if self._cancel_requested:
                    raise RuntimeError("Training cancelled by user")

                status_msg = f"Epoch {epoch}: Loss={loss:.6f}"
                self.progress_updated.emit(epoch, status_msg)

                if metrics:
                    self.epoch_completed.emit(
                        epoch, loss,
                        metrics.get('reconstruction_loss', 0),
                        metrics.get('kl_loss', 0),
                        metrics.get('learning_rate', 0)
                    )

                if original_callback:
                    original_callback(epoch, loss, metrics)

            self.config.callback = progress_callback
            self.generator._cancel_training = self._cancel_requested

            # Run training with resume state
            result = self.generator.train(self.samples, self.config, resume_from=self.resume_from)

            self.training_finished.emit(True, "Training completed successfully!")

        except RuntimeError as e:
            error_msg = str(e)
            print(f"Training error details: {error_msg}")  # Print to console

            # Persist state even on cancellation
            self.generator._total_epochs_trained = self.generator._total_epochs_trained
            self.generator._best_loss = self.generator._best_loss
            self.generator._training_history = self.generator._training_history

            if "cancelled" in error_msg.lower():
                self.training_finished.emit(True, "Training stopped by user (model preserved)")
            elif "out of memory" in error_msg.lower():
                self.training_finished.emit(False, f"GPU Out of Memory Error.\n\nTry reducing the Batch Size (e.g., to 8 or 4).\n\nDetails: {error_msg}")
            else:
                self.training_finished.emit(False, f"Training error: {error_msg}")

        except Exception as e:
            error_msg = str(e)
            print(f"Training failed details: {error_msg}") # Print to console

            # Attempt to preserve state on crash
            if hasattr(self.generator, '_total_epochs_trained'):
                self.generator._total_epochs_trained = self.generator._total_epochs_trained
            if hasattr(self.generator, '_best_loss'):
                self.generator._best_loss = self.generator._best_loss
            if hasattr(self.generator, '_training_history'):
                self.generator._training_history = self.generator._training_history

            self.training_finished.emit(False, f"Training failed: {error_msg}")


class LatentExplorerApp(QWidget):
    """Main application window for latent space audio exploration."""

    def __init__(self):
        super().__init__()
        self.generator = None
        self.samples = []
        self.current_audio = None

        # Training thread and worker
        self.training_thread = None
        self.training_worker = None
        self.progress_dialog = None
        self.training_total_epochs = 0
        self._training_finished_popup_shown = False

        self.setup_generator()
        self.setup_ui()
        self.connect_signals()

    def setup_generator(self):
        """Initialize the audio generator."""
        try:
            config = GeneratorConfig()
            self.generator = AdvancedAudioGenerator(config)
            # Update UI to reflect generator's latent dimension
            self.latent_widget = LatentVectorWidget(
                latent_dim=self.generator.config.latent_dim
            )
            self.generation_controls = GenerationControls()
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error",
                               f"Failed to initialize generator: {e}")
            self.generator = None

    def setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)

        layout = QVBoxLayout(self)

        # Title
        title = QLabel("🧠 Latent Space Explorer")
        title.setFont(TITLE_FONT)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Direct neural network control - no fake parameters")
        subtitle.setStyleSheet("color: #888; margin-bottom: 10px;")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Controls
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # Right panel - Tabs
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        splitter.setSizes(DEFAULT_SPLITTER_SIZES)
        layout.addWidget(splitter)

        # Bottom controls
        bottom_layout = self.create_bottom_controls()
        layout.addLayout(bottom_layout)

        # Status bar
        self.status_widget = StatusWidget()
        layout.addWidget(self.status_widget)

    def create_left_panel(self):
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Sound Map Widget (Initialize early so buttons can connect)
        self.sound_map = SoundMapWidget()
        self.sound_map.point_clicked.connect(self.on_map_clicked)

        # View Switcher
        switch_layout = QHBoxLayout()
        self.view_stack = QStackedWidget()
        
        self.toggle_view_btn = QPushButton("🗺️ Switch to Sound Map")
        self.toggle_view_btn.clicked.connect(self.toggle_latent_view)
        self.toggle_view_btn.setStyleSheet(BUTTON_STYLE)
        switch_layout.addWidget(self.toggle_view_btn)

        # Sound Map Source Buttons
        self.map_data_btn = QPushButton("📁 Data")
        self.map_data_btn.setToolTip("Project map from loaded training dataset")
        self.map_data_btn.clicked.connect(self._map_from_dataset)
        self.map_data_btn.setStyleSheet(BUTTON_STYLE)
        self.map_data_btn.setVisible(False)
        switch_layout.addWidget(self.map_data_btn)

        self.map_model_btn = QPushButton("🧠 Model")
        self.map_model_btn.setToolTip("Restore map from coordinates stored in model file")
        self.map_model_btn.clicked.connect(self._map_from_cache)
        self.map_model_btn.setStyleSheet(BUTTON_STYLE)
        self.map_model_btn.setVisible(False)
        switch_layout.addWidget(self.map_model_btn)

        self.synth_map_btn = QPushButton("✨ Synth")
        self.synth_map_btn.setToolTip("Generate map by random sampling (imagination)")
        self.synth_map_btn.clicked.connect(lambda: self.generate_synthetic_map())
        self.synth_map_btn.setStyleSheet(BUTTON_STYLE)
        self.synth_map_btn.setVisible(False)
        switch_layout.addWidget(self.synth_map_btn)

        self.reset_map_btn = QPushButton("🏠 Reset")
        self.reset_map_btn.setToolTip("Reset zoom and pan")
        self.reset_map_btn.clicked.connect(self.sound_map.reset_view)
        self.reset_map_btn.setStyleSheet(BUTTON_STYLE)
        self.reset_map_btn.setVisible(False)
        switch_layout.addWidget(self.reset_map_btn)
        
        self.snap_check = QCheckBox("Snap")
        self.snap_check.setToolTip("Snap to nearest real point (checked) or blend between points (unchecked)")
        self.snap_check.setChecked(True)
        self.snap_check.setVisible(False)
        self.snap_check.setStyleSheet("color: #ccc; font-size: 10px;")
        switch_layout.addWidget(self.snap_check)

        layout.addLayout(switch_layout)

        # Latent vector controls (Sliders)
        self.latent_widget = LatentVectorWidget(
            latent_dim=self.generator.config.latent_dim if self.generator else 128
        )
        
        self.view_stack.addWidget(self.latent_widget)
        self.view_stack.addWidget(self.sound_map)
        layout.addWidget(self.view_stack)

        # Generation controls
        self.generation_controls = GenerationControls()
        layout.addWidget(self.generation_controls)

        # Waveform display
        waveform_label = QLabel("Waveform Preview")
        waveform_label.setStyleSheet("color: #888; margin-top: 10px;")
        layout.addWidget(waveform_label)

        self.waveform_viz = WaveformVisualizer()
        layout.addWidget(self.waveform_viz)

        # Playback controls
        self.playback_controls = PlaybackControls()
        layout.addWidget(self.playback_controls)

        layout.addStretch()
        return panel

    def toggle_latent_view(self):
        """Toggle between slider view and map view."""
        current = self.view_stack.currentIndex()
        if current == 0:
            self.view_stack.setCurrentIndex(1)
            self.toggle_view_btn.setText("🎚️ Sliders")
            
            # Show map controls
            self.map_data_btn.setVisible(True)
            self.map_model_btn.setVisible(True)
            self.synth_map_btn.setVisible(True)
            self.reset_map_btn.setVisible(True)
            self.snap_check.setVisible(True)
            
            # Update button enabled states
            self.update_map_button_states()
        else:
            self.view_stack.setCurrentIndex(0)
            self.toggle_view_btn.setText("🗺️ Map")
            self.map_data_btn.setVisible(False)
            self.map_model_btn.setVisible(False)
            self.synth_map_btn.setVisible(False)
            self.reset_map_btn.setVisible(False)
            self.snap_check.setVisible(False)


    def update_map_button_states(self):
        """Enable/disable map buttons based on availability."""
        has_generator = self.generator is not None and self.generator.model is not None
        self.map_data_btn.setEnabled(has_generator and bool(self.samples))
        self.map_model_btn.setEnabled(has_generator and self.generator.cached_latents is not None)
        self.synth_map_btn.setEnabled(has_generator)
        self.reset_map_btn.setEnabled(self.sound_map.points.size > 0)

    def on_map_clicked(self, coords):
        """Handle clicks on the sound map."""
        if not self.generator or not self.generator.model:
            return
            
        if self.sound_map.points.size == 0 or self._latent_samples is None:
            return
            
        # Calculate distances to all points
        dists = np.linalg.norm(self.sound_map.points - coords, axis=1)
        
        if self.snap_check.isChecked():
            # SNAP MODE: Nearest neighbor
            idx = np.argmin(dists)
            vector = self._latent_samples[idx]
        else:
            # BLEND MODE: Inverse Distance Weighting of nearest K neighbors
            K = 5
            # Find indices of K nearest neighbors
            near_indices = np.argsort(dists)[:K]
            near_dists = dists[near_indices]
            near_vectors = self._latent_samples[near_indices]
            
            # Calculate weights (inverse distance squared)
            epsilon = 1e-6
            weights = 1.0 / (near_dists**2 + epsilon)
            weights /= np.sum(weights)
            
            # Compute weighted average
            vector = np.sum(near_vectors * weights[:, np.newaxis], axis=0)
        
        self.latent_widget.set_vector(vector)
        self.generate_audio()

    def update_sound_map(self):
        """Auto-detect the best available source for the map."""
        if self.samples and self.generator and self.generator.model:
            self._map_from_dataset()
        elif self.generator and self.generator.cached_latents is not None:
            self._map_from_cache()

    def _map_from_dataset(self):
        """Project currently loaded training samples."""
        if not self.samples or not self.generator or not self.generator.model:
            return
        try:
            self.status_widget.set_status("Projecting dataset to map...")
            self.generator.model.eval()
            latent_list = []
            for s in self.samples:
                z = self.generator.encode_audio(s)
                latent_list.append(z)
            
            self._latent_samples = np.array(latent_list)
            self.generator.cached_latents = self._latent_samples
            
            self.generator.fit_projector(self._latent_samples)
            points_2d = self.generator.project_to_2d(self._latent_samples)
            self.sound_map.set_points(points_2d)
            self.status_widget.set_status("Map built from dataset")
            self.update_map_button_states()
        except Exception as e:
            logger.error(f"Dataset projection failed: {e}")
            self.status_widget.set_error_status(f"Data map error: {e}")

    def _map_from_cache(self):
        """Restore map from coordinates stored in model file."""
        if not self.generator or self.generator.cached_latents is None:
            return
        try:
            self.status_widget.set_status("Restoring map from model cache...")
            self._latent_samples = self.generator.cached_latents
            self.generator.fit_projector(self._latent_samples)
            points_2d = self.generator.project_to_2d(self._latent_samples)
            self.sound_map.set_points(points_2d)
            self.status_widget.set_status("Map restored from model cache")
            self.update_map_button_states()
        except Exception as e:
            logger.error(f"Cache restoration failed: {e}")
            self.status_widget.set_error_status(f"Cache map error: {e}")

    def generate_synthetic_map(self, n_points: int = 500):
        """Generate a map by hallucinating random points from the model."""
        if not self.generator or not self.generator.model:
            return
            
        try:
            self.status_widget.set_status(f"Generating synthetic map ({n_points} points)...")
            self.generator.model.eval()
            
            # Generate random latent vectors
            import torch
            with torch.no_grad():
                # Use getattr to avoid LSP warning on custom model method
                sample_func = getattr(self.generator.model, 'sample_latent')
                z_tensor = sample_func(n_points, str(self.generator.device))
                self._latent_samples = z_tensor.cpu().numpy()
            
            self.generator.fit_projector(self._latent_samples)
            points_2d = self.generator.project_to_2d(self._latent_samples)
            self.sound_map.set_points(points_2d)
            self.status_widget.set_status("Synthetic sound map generated")
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic map: {e}")
            self.status_widget.set_error_status(f"Synthetic map error: {e}")


    def create_right_panel(self):
        """Create the right tab panel."""
        self.tabs = QTabWidget()

        # Preset tab
        self.preset_tab = PresetManager(self.generator, latent_widget=self.latent_widget)
        self.tabs.addTab(self.preset_tab, "💾 Presets")

        # Morph tab
        self.morph_tab = MorphTab(self.generator, latent_widget=self.latent_widget)
        self.tabs.addTab(self.morph_tab, "🔄 Morph")

        # Walk tab
        self.walk_tab = WalkTab(self.generator, latent_widget=self.latent_widget)
        self.tabs.addTab(self.walk_tab, "🚶 Walk")

        # Reconstruction tab
        self.recon_tab = ReconstructionTab(self.generator, latent_widget=self.latent_widget)
        self.tabs.addTab(self.recon_tab, "🔬 Recon")

        # Variations tab
        self.variations_tab = VariationsTab(self.generator, latent_widget=self.latent_widget)
        self.tabs.addTab(self.variations_tab, "🎨 Variations")

        # Attributes tab
        self.attr_tab = AttributesTab(self.generator, latent_widget=self.latent_widget)
        self.tabs.addTab(self.attr_tab, "✨ Attributes")

        return self.tabs

    def create_bottom_controls(self):
        """Create bottom control buttons."""
        layout = QHBoxLayout()

        # Data loading
        self.load_samples_btn = QPushButton("📁 Load Training Data :3")
        self.load_samples_btn.clicked.connect(self.load_samples)
        self.load_samples_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.load_samples_btn)

        # Model training
        self.train_btn = QPushButton("🎓 Train Model")
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.train_btn)

        # Model management
        self.save_model_btn = QPushButton("💾 Save Model")
        self.save_model_btn.clicked.connect(self.save_model)
        self.save_model_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.save_model_btn)

        self.load_model_btn = QPushButton("📂 Load Model")
        self.load_model_btn.clicked.connect(self.load_model)
        self.load_model_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.load_model_btn)

        # Unload model
        self.unload_model_btn = QPushButton("⏏️ Unload Model")
        self.unload_model_btn.clicked.connect(self.unload_model)
        self.unload_model_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.unload_model_btn)

        # TensorBoard
        self.tensorboard_btn = QPushButton("📊 TensorBoard")
        self.tensorboard_btn.clicked.connect(self.toggle_tensorboard)
        self.tensorboard_btn.setStyleSheet(BUTTON_STYLE)
        layout.addWidget(self.tensorboard_btn)

        return layout

    def connect_signals(self):
        """Connect widget signals."""
        # Generation controls
        self.generation_controls.generate_requested.connect(self.generate_audio)
        self.generation_controls.random_generate_requested.connect(self.generate_random_audio)

        # Playback controls
        self.playback_controls.play_requested.connect(self.play_audio)
        self.playback_controls.save_requested.connect(self.save_audio)

        # Latent vector changes
        self.latent_widget.vector_changed.connect(self.on_vector_changed)

        # Tab signals
        self.preset_tab.preset_selected.connect(self.load_preset)
        self.recon_tab.vector_pushed.connect(self.load_preset)
        self.variations_tab.vector_selected.connect(self.load_preset)
        self.attr_tab.apply_requested.connect(self.generate_audio)

    def on_vector_changed(self, vector):
        """Handle latent vector changes."""
        # Could add live preview here if desired
        pass

    def generate_audio(self):
        """Generate audio from current latent vector."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded. Train or load a model first!")
            return

        try:
            self.status_widget.set_generating_status()
            z = self.latent_widget.get_vector()
            self.current_audio = self.generator.generate_from_latent(z)

            self.waveform_viz.set_audio(self.current_audio)
            self.playback_controls.set_playback_enabled(True)
            self.status_widget.set_ready_status()

        except Exception as e:
            self.status_widget.set_error_status(str(e))
            QMessageBox.critical(self, "Generation Error", f"Failed to generate audio: {e}")

    def generate_random_audio(self):
        """Generate random audio and update UI."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model loaded. Train or load a model first!")
            return

        try:
            self.status_widget.set_generating_status()

            # Generate random vector and update widget
            self.latent_widget.randomize()
            z = self.latent_widget.get_vector()

            self.current_audio = self.generator.generate_from_latent(z)
            self.waveform_viz.set_audio(self.current_audio)
            self.playback_controls.set_playback_enabled(True)
            self.status_widget.set_ready_status()

            # Play immediately
            self.play_audio()

        except Exception as e:
            self.status_widget.set_error_status(str(e))
            QMessageBox.critical(self, "Generation Error", f"Failed to generate audio: {e}")

    def play_audio(self):
        """Play the current audio."""
        if self.current_audio is not None:
            try:
                import sounddevice as sd
                sd.play(self.current_audio, self.generator.config.sample_rate)
                self.status_widget.set_status("▶ Playing...")
            except ImportError:
                QMessageBox.warning(self, "Error", "Install sounddevice: pip install sounddevice")
        else:
            QMessageBox.information(self, "Info", "No audio to play. Generate audio first.")

    def save_audio(self):
        """Save current audio to file."""
        if self.current_audio is None:
            QMessageBox.information(self, "Info", "No audio to save. Generate audio first.")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Audio", "", "WAV (*.wav)")
        if filepath:
            try:
                self.generator.save_audio(self.current_audio, filepath)
                QMessageBox.information(self, "Success", f"Audio saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save audio: {e}")

    def load_preset(self, vector):
        """Load a preset vector into the latent widget."""
        self.latent_widget.set_vector(vector)
        self.generate_audio()

    def load_samples(self):
        """Load training samples."""
        try:
            self.status_widget.set_status("Loading samples...")
            folder = QFileDialog.getExistingDirectory(self, "Select Audio Folder")
            if not folder:
                return

            self.samples = self.generator.load_dataset(folder, recursive=True)
            success_msg = f"Loaded {len(self.samples)} training samples"
            
            self.update_map_button_states()
            QMessageBox.information(self, "Success", success_msg)
            self.status_widget.set_status(success_msg)

        except Exception as e:
            error_msg = f"Failed to load samples: {e}"
            self.status_widget.set_error_status(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def train_model(self):
        """Train the model with user configuration."""
        if not self.generator:
            QMessageBox.critical(self, "Error", "Generator not initialized")
            return

        if not self.samples:
            QMessageBox.warning(self, "Error", "No training samples loaded. Load samples first!")
            return

        # Reset popup flag for new training session
        self._training_finished_popup_shown = False

        # Check if there's a loaded training state that can be resumed
        resume_from = None
        total_trained = 0
        
        # Determine potential resume state
        state_candidate = None
        
        # Priority 1: Current generator state (check first to see if it has history)
        gen_state = self.generator.get_training_state()
        if gen_state.get('can_resume', False):
            state_candidate = gen_state
            
        # Priority 2: Explicitly loaded state (fallback)
        elif hasattr(self, '_loaded_training_state') and self._loaded_training_state:
            state_candidate = self._loaded_training_state

        if state_candidate and state_candidate.get('can_resume', False):
            total_trained = state_candidate.get('total_epochs_trained', 0)
            best_loss = state_candidate.get('best_loss', float('inf'))
            
            # Simplified more helpful message
            msg = (
                f"Previous training found: {total_trained} epochs completed.\n"
                f"Best loss achieved: {best_loss:.6f}\n\n"
                f"Continue training from epoch {total_trained + 1}?\n\n"
                f"Click 'Yes' to resume or 'No' to start over."
            )
            
            reply = QMessageBox.question(
                self, "Resume Training?", msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply == QMessageBox.StandardButton.Yes:
                resume_from = state_candidate

        # Show training dialog
        dialog = TrainingDialog(len(self.samples), self, resume_from=resume_from)

        # DEBUG: Log beta KL value right after dialog creation
        try:
            with open('latentaudio_debug_after_create.txt', 'w') as f:
                f.write(f"AFTER dialog creation: beta_kl = {dialog.beta_kl_spin.value()}\n")
        except:
            pass

        # DEBUG: Log beta KL value before dialog execution
        try:
            with open('latentaudio_debug_before_exec.txt', 'w') as f:
                f.write(f"BEFORE dialog.exec(): beta_kl = {dialog.beta_kl_spin.value()}\n")
        except:
            pass

        if not dialog.exec():
            return

        config = dialog.get_config()
        self.training_total_epochs = config.epochs

        # Build model if not already built
        if self.generator.model is None:
            self.generator.build_model()

        # Create progress dialog
        start_epoch = resume_from.get('total_epochs_trained', 0) if resume_from else 0
        self.progress_dialog = QProgressDialog(
            f"Training... (resuming from epoch {start_epoch + 1})" if resume_from else "Training...",
            "Cancel",
            0, self.training_total_epochs,
            self
        )
        self.progress_dialog.setWindowTitle("Training Progress")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        self.progress_dialog.setMinimumWidth(400)
        self.progress_dialog.setMinimumDuration(0)  # Show immediately
        self.progress_dialog.show()

        # Create training worker and thread
        self.training_worker = TrainingWorker(self.generator, self.samples, config, resume_from)
        self.training_thread = QThread()

        # Move worker to thread
        self.training_worker.moveToThread(self.training_thread)

        # Connect signals
        self.training_thread.started.connect(self.training_worker.run_training)
        self.training_worker.training_started.connect(self.on_training_started)
        self.training_worker.progress_updated.connect(self.on_training_progress)
        self.training_worker.epoch_completed.connect(self.on_epoch_completed)
        self.training_worker.training_finished.connect(self.on_training_finished)
        self.training_worker.training_finished.connect(self.training_thread.quit)
        self.training_worker.training_finished.connect(self.training_worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)

        # Handle cancellation
        self.progress_dialog.canceled.connect(self.cancel_training)

        # Update UI
        self.status_widget.set_training_status()
        self.train_btn.setEnabled(False)

        # Start training thread
        self.training_thread.start()

    def on_training_progress(self, epoch, message):
        """Handle training progress updates."""
        if self.progress_dialog:
            self.progress_dialog.setValue(epoch)
            self.progress_dialog.setLabelText(message)
            self.status_widget.set_status(message)

    def on_epoch_completed(self, epoch, loss, recon_loss, kl_loss, lr):
        """Handle epoch completion."""
        # Update status widget with detailed info
        self.status_widget.set_training_progress(epoch, self.training_total_epochs, loss)

    def on_training_finished(self, success, message):
        """Handle training completion."""
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None

        # Update UI
        self.train_btn.setEnabled(True)

        # Check if a model exists in memory
        model_loaded = self.generator is not None and self.generator.model is not None

        if success and model_loaded:
            # Clear loaded state so next train checks for generator's updated state
            self._loaded_training_state = None

            # Enable generation
            self.generation_controls.set_generation_enabled(True)
            self.status_widget.set_ready_status()

            # Show success popup only once
            if not self._training_finished_popup_shown:
                QMessageBox.information(self, "Success", message)
                self._training_finished_popup_shown = True

            # Refresh preset list in case any were loaded
            if hasattr(self, 'preset_tab'):
                self.preset_tab.refresh_list()

        elif model_loaded:
            # Training was stopped/cancelled, but model is still loaded
            # Enable generation controls so user can still use them
            self.generation_controls.set_generation_enabled(True)
            self.status_widget.set_status("Ready (Model Loaded, Training Stopped)")

            # Show cancellation message only once
            if not self._training_finished_popup_shown:
                QMessageBox.information(self, "Training Stopped", message)
                self._training_finished_popup_shown = True

        else:
            self.status_widget.set_error_status(message)

            # Show error popup only once
            if not self._training_finished_popup_shown:
                QMessageBox.critical(self, "Training Error", message)
                self._training_finished_popup_shown = True

        # Clean up
        self.training_worker = None
        self.training_thread = None

    def on_training_started(self):
        """Handle training startup - launch TensorBoard."""
        try:
            self.generator.start_tensorboard()
            self.status_widget.set_status("TensorBoard launched - monitor training progress")
            self.tensorboard_btn.setText("✅ TB Running")
        except Exception as e:
            # Don't fail training if TensorBoard fails
            self.status_widget.set_status("Training started (TensorBoard launch failed)")

    def toggle_tensorboard(self):
        """Launch TensorBoard."""
        try:
            if not self.generator:
                QMessageBox.warning(self, "No Generator",
                                  "Initialize the generator first.")
                return

            # Launch TensorBoard - the training logger handles process management
            self.generator.start_tensorboard()
            self.status_widget.set_status("TensorBoard launched - check your browser")
            self.tensorboard_btn.setText("✅ TB Running")

        except Exception as e:
            self.status_widget.set_error_status(f"Failed to launch TensorBoard: {e}")
            QMessageBox.critical(self, "TensorBoard Error", str(e))

    def closeEvent(self, a0):
        """Handle application closing."""
        # TensorBoard processes are cleaned up automatically by the training logger
        a0.accept()

    def cancel_training(self):
        """Cancel the current training."""
        if self.training_worker and self.training_thread and self.training_thread.isRunning():
            # Request cancellation
            self.training_worker.request_cancel()
            self.status_widget.set_status("Cancelling training...")
        else:
            QMessageBox.information(self, "No Training", "No training is currently running.")

    def save_model(self):
        """Save the trained model."""
        if not self.generator or not self.generator.model:
            QMessageBox.warning(self, "Error", "No model to save. Train a model first!")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "Save Model", "", "PyTorch Models (*.pth)")
        if filepath:
            try:
                # Get training state to save
                training_state = self.generator.get_training_state()
                self.generator.save_model(
                    filepath,
                    optimizer=None,  # We don't have access to optimizer after training
                    scheduler=None,  # We don't have access to scheduler after training
                    current_epoch=training_state.get('total_epochs_trained', 0),
                    total_epochs_trained=training_state.get('total_epochs_trained', 0),
                    best_loss=training_state.get('best_loss', float('inf')),
                    training_history=training_state.get('training_history', {})
                )
                QMessageBox.information(self, "Success", f"Model saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save model: {e}")

    def load_model(self):
        """Load a trained model."""
        try:
            filepath, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Models (*.pth)")
            if not filepath:
                return

            self.status_widget.set_loading_status()

            # Load model and get training state
            training_state = self.generator.load_model(filepath)

            # Store training state for potential resuming
            self._loaded_training_state = training_state

            # Enable generation
            self.generation_controls.set_generation_enabled(True)

            # Refresh preset list
            self.preset_tab.refresh_list()

            self.update_map_button_states()
            self.status_widget.set_ready_status()

            # Show message about loaded model
            if training_state.get('can_resume', False):
                epochs = training_state.get('total_epochs_trained', 0)
                QMessageBox.information(
                    self, "Model Loaded",
                    f"Model loaded successfully!\n\n"
                    f"Training history found:\n"
                    f"- Total epochs trained: {epochs}\n"
                    f"- Best loss: {training_state.get('best_loss', 'N/A'):.6f}\n\n"
                    f"Click 'Train Model' to continue training from where it left off."
                )
            else:
                QMessageBox.information(self, "Success", "Model loaded successfully!")

        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self.status_widget.set_error_status(error_msg)
            QMessageBox.critical(self, "Error", error_msg)

    def unload_model(self):
        """Unload the current model."""
        if not self.generator or not self.generator.model:
            QMessageBox.information(self, "Info", "No model currently loaded.")
            return

        reply = QMessageBox.question(
            self, "Unload Model",
            "Are you sure you want to unload the current model?\nUnsaved changes will be lost.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Offload model to RAM instead of deleting to save VRAM but keep ready
            if self.generator:
                self.generator.offload_to_ram()
            
            # Clear UI states
            self.current_audio = None
            self._loaded_training_state = None
            
            # Reset widgets
            self.waveform_viz.clear()
            self.generation_controls.set_generation_enabled(False)
            self.status_widget.set_status("Model offloaded to RAM")
            
            QMessageBox.information(self, "Success", "Model offloaded to CPU RAM. VRAM cleared.")


def main():
    """Main entry point for the LatentAudio GUI application."""
    import sys
    from PyQt6.QtWidgets import QApplication
    import os

    # Check if we have a display
    if os.name == 'nt':  # Windows
        # On Windows, assume we have display
        pass
    else:  # Unix-like
        if not os.environ.get('DISPLAY'):
            print("No DISPLAY environment variable found. GUI cannot run in headless mode.")
            sys.exit(1)

    # Create QApplication
    app = QApplication(sys.argv)

    # Set application properties
    app.setApplicationName("LatentAudio")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("LatentAudio Team")

    try:
        # Create and show main window
        window = LatentExplorerApp()
        window.show()

        # Run event loop
        sys.exit(app.exec())
    except Exception as e:
        print(f"UI failed to start: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
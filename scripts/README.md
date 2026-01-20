# LatentAudio Scripts

**Note Some of these where made during the creation and debugging process so they might not even be useful anymore, but they are here just in case they help**

Utilities for data preprocessing, training diagnostics, model verification, and latent space exploration.

## 📊 Data Preparation & Analysis

*   **`analyze_audio.py`**: Analyzes audio files for length consistency and detects long silences.
*   **`diagnose_data.py`**: Thoroughly checks datasets for NaNs, Infs, normalization issues, clipping, and DC offsets before training.
*   **`prepare_dataset.py`**: Advanced DSP pipeline for phase-locked transient alignment, peak normalization, and polarity correction.
*   **`cleanup_silent_files.py`**: Recursively removes silent segments from preprocessed datasets using configurable RMS thresholds.

## 📈 Training Monitoring & Log Analysis

*   **`check_health.py`**: Real-time training auditor. Reads TensorBoard logs to report loss trends, KL divergence health, and posterior collapse risks.
*   **`summarize_logs.py`**: Generates a comprehensive `TRAINING_SUMMARY.md` report from the most recent training session.

## 🔬 Model Verification & Testing

*   **`preflight_check.py`**: Basic sanity check for model architectures (Simple vs Complex) and output shapes.
*   **`verify_skip_connections.py`**: Detailed verification of U-Net skip connection geometry and SkipGenerator consistency.
*   **`verify_upgrade.py`**: Tests model initialization silence and the full reconstruction pipeline functionality.
*   **`verify_annealing.py`**: Visualizes and validates KL beta annealing schedules (Linear/Sigmoid).
*   **`stress_test_hd.py`**: Stress tests the Phase 7 High-Fidelity engine, verifying gradients and output headroom.
*   **`debug_model.py`**: Loads the latest checkpoint and generates sample stats to verify model health.

## 🛠️ Model Refinement & Exploration

*   **`polish_model.py`**: Final stage refinement with micro-learning rates and high waveform weights for maximum fidelity.
*   **`deep_clean_v2.py`**: Signal-aware cleaning using energy-gating to preserve 808 tails while scrubbing static from silence.
*   **`deep_clean_v3.py`**: The "No-Cheat" Fidelity Bridge. Forces training using hallucinated skips to ensure zero discrepancy between training and generation.
*   **`identify_latent_attributes.py`**: Correlates latent dimensions with mathematical audio features (Brightness, Noisiness, etc.).
*   **`test_walk_geometry.py`**: Benchmarks latent space exploration efficiency (Momentum Walk vs. Random Walk).

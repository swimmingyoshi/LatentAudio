# LatentAudio Scripts

**Note Some of these where made during the creation and debugging process so they might not even be useful anymore, but they are here just in case they help**

Utilities for data preprocessing, training diagnostics, model verification, and latent space exploration.

## 📊 Data Preparation & Analysis

*   **`analyze_audio.py`**: Analyzes audio files for length consistency and detects long silences.
*   **`diagnose_data.py`**: Thoroughly checks datasets for NaNs, Infs, normalization issues, clipping, and DC offsets before training.
*   **`prepare_dataset.py`**: Advanced DSP pipeline for phase-locked transient alignment, peak normalization, and polarity correction.
*   **`cleanup_silent_files.py`**: Recursively removes silent segments from preprocessed datasets using configurable RMS thresholds.
*   **`download_esc50.py`**: Downloads and extracts the ESC-50 environmental sound dataset.
*   **`preprocess_esc50.py`**: Slices ESC-50 5s clips into 0.5s training segments.
*   **`preprocess_kicks.py`**: Normalizes kick drum samples to exactly 1.0s.
*   **`preprocess_synths.py`**: Extracts the first 1.0s of synth samples for training.

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

## 🎛️ Exporting a Model to the VST

The pipeline from a trained checkpoint to a folder the LatentAudioVST plugin can load:

```bash
# 1. Strip training state -> writes MyModel_PRUNED_FP32.pth and MyModel_PRUNED_FP16.pth next to it
python scripts/prune_model.py path/to/MyModel.pth

# 2. (optional) Benchmark load time / decode latency
python scripts/test_pruned_models.py --fp32 path/to/MyModel_PRUNED_FP32.pth --fp16 path/to/MyModel_PRUNED_FP16.pth

# 3. Export the decoder to ONNX + build the 2D sound map
python scripts/export_for_vst.py path/to/MyModel_PRUNED_FP16.pth -o path/to/MyModelVST
```

Then click **Load Model Folder** in the plugin and pick the output folder; it contains `latent_decoder.onnx` and `sound_map.json`.

*   **`benchmark_inference.py`**: Measures model load time and decode speed for a checkpoint.

#!/usr/bin/env python3
"""
Latent Feature Discovery Tool for LatentAudio

This script identifies which latent dimensions correlate with specific audio features
like Brightness, Loudness, and Noisiness.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from scipy import stats
from loguru import logger

from latentaudio import AdvancedAudioGenerator
from latentaudio.types import GeneratorConfig

def extract_features(audio, sr):
    """Extract mathematical audio features."""
    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = audio.squeeze()
    
    # Brightness (Spectral Centroid)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    brightness = np.mean(centroid)
    
    # Loudness (RMS Energy)
    rms = librosa.feature.rms(y=audio)[0]
    loudness = np.mean(rms)
    
    # Noisiness (Zero Crossing Rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio)[0]
    noisiness = np.mean(zcr)
    
    # Spectral Rolloff (Frequency below which 85% of energy lies)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    rolloff_val = np.mean(rolloff)
    
    return {
        "Brightness": brightness,
        "Loudness": loudness,
        "Noisiness": noisiness,
        "SpectralRolloff": rolloff_val
    }

def main():
    parser = argparse.ArgumentParser(description="Identify latent attributes by correlating dimensions with audio features.")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model (.pth)")
    parser.add_argument("--audio-dir", type=str, required=True, help="Path to directory of audio files")
    parser.add_argument("--output", type=str, default="latent_correlations.csv", help="Output CSV path")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top dimensions to show per feature")
    
    args = parser.parse_args()
    
    model_path = Path(args.model)
    audio_dir = Path(args.audio_dir)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return

    if not audio_dir.exists():
        logger.error(f"Audio directory not found: {audio_dir}")
        return

    # 1. Load Model
    logger.info(f"Loading model from {model_path}...")
    generator = AdvancedAudioGenerator()
    checkpoint_info = generator.load_model(model_path)
    latent_dim = generator.config.latent_dim
    sr = generator.config.sample_rate
    
    logger.info(f"Model loaded. Latent Dim: {latent_dim}, Sample Rate: {sr}")
    
    # 2. Collect Audio Files
    audio_files = list(audio_dir.rglob("*.wav")) + list(audio_dir.rglob("*.mp3"))
    if not audio_files:
        logger.error("No audio files found in directory.")
        return
    
    logger.info(f"Found {len(audio_files)} audio files. Processing...")
    
    # 3. Extract Latent Vectors and Audio Features
    latent_data = []
    feature_data = []
    
    for f in tqdm(audio_files, desc="Analyzing dataset"):
        try:
            # Load and preprocess audio
            audio = generator.load_audio_file(f)
            
            # Encode to latent space
            z = generator.encode_audio(audio)
            
            # Extract features
            features = extract_features(audio, sr)
            
            latent_data.append(z)
            feature_data.append(features)
        except Exception as e:
            logger.warning(f"Failed to process {f.name}: {e}")
            
    if not latent_data:
        logger.error("No data processed successfully.")
        return
        
    Z = np.array(latent_data)  # (N, latent_dim)
    F_df = pd.DataFrame(feature_data)  # (N, num_features)
    
    logger.info(f"Analysis complete on {len(Z)} samples. Calculating correlations...")
    
    # 4. Calculate Correlations
    results = []
    features = F_df.columns
    
    for feature in features:
        feature_vals = F_df[feature].values
        correlations = []
        
        for d in range(latent_dim):
            z_dim_vals = Z[:, d]
            
            # Use Pearson correlation
            if np.std(z_dim_vals) < 1e-6:
                corr, p_val = 0.0, 1.0
            else:
                corr, p_val = stats.pearsonr(z_dim_vals, feature_vals)
            
            correlations.append({
                "dimension": d,
                "correlation": corr,
                "abs_correlation": abs(corr),
                "p_value": p_val
            })
            
        # Sort by absolute correlation
        corr_df = pd.DataFrame(correlations).sort_values(by="abs_correlation", ascending=False)
        
        logger.info(f"\nTop dimensions for {feature}:")
        for i in range(min(args.top_n, len(corr_df))):
            row = corr_df.iloc[i]
            direction = "positive" if row['correlation'] > 0 else "negative"
            print(f"  Dimension {int(row['dimension']):3d} -> {row['abs_correlation']*100:5.1f}% correlation ({direction}, p={row['p_value']:.2e})")
            
        # Store all for export
        for i, row in corr_df.iterrows():
            results.append({
                "feature": feature,
                "dimension": int(row['dimension']),
                "correlation": row['correlation'],
                "p_value": row['p_value']
            })
            
    # 5. Export to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    logger.info(f"Full correlation matrix saved to {args.output}")

if __name__ == "__main__":
    main()

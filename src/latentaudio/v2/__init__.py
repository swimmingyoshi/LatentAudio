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
"""LatentAudio v2: spectral-domain VAE with a small, fast, ONNX-exportable decoder.

Compared with the v1 SimpleFastVAE:
  * the decoder works at STFT frame rate (ConvNeXt blocks conditioned on z) and
    synthesises audio with an inverse STFT implemented as a fixed transposed
    convolution - roughly 50x fewer parameters and far less compute;
  * no encoder->decoder skip connections, so training and generation use the
    exact same path (v1 had to "hallucinate" skips at inference);
  * spectral losses on every step instead of every 5th batch.
"""

from .model import LatentAudioV2, V2Config

__all__ = ["LatentAudioV2", "V2Config"]

"""Model Compression, Pruning & Quantization Utilities."""

import sys
import numpy as np
from typing import Dict, Any
from logger import get_logger

logger = get_logger("ModelCompressor")


class ModelCompressor:
    """Reduces model memory footprint and optimizes inference overhead."""

    @staticmethod
    def quantize_coefficients(weights: np.ndarray, precision: str = "float16") -> np.ndarray:
        """Downcasts floating point weights to lower precision (e.g. float16)."""
        logger.info(f"Quantizing weight array of shape {weights.shape} to {precision}...")
        original_bytes = weights.nbytes
        quantized = weights.astype(precision)
        new_bytes = quantized.nbytes

        compression_ratio = round(original_bytes / new_bytes, 2)
        logger.info(f"Quantization complete: {original_bytes}B -> {new_bytes}B ({compression_ratio}x reduction)")
        return quantized

    @staticmethod
    def prune_near_zero_weights(weights: np.ndarray, threshold: float = 1e-3) -> np.ndarray:
        """Prunes small weight values by zeroing them out to enhance sparsity."""
        pruned = np.where(np.abs(weights) < threshold, 0.0, weights)
        sparsity = round(np.mean(pruned == 0.0) * 100, 2)
        logger.info(f"Pruned weights below {threshold}. Resulting sparsity: {sparsity}%")
        return pruned

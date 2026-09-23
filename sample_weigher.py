"""Sample Weighting Engine for ML-Project-01.

Computes sample weights based on temporal exponential decay half-life, inverse class frequency,
and density-based outlier down-weighting.
"""

from typing import Optional, Union
import numpy as np
import pandas as pd
from logger import logger


class SampleWeigher:
    """Calculates custom instance training weights to prioritize recent or rare observations."""

    @staticmethod
    def compute_temporal_decay_weights(
        timestamps: Union[pd.Series, np.ndarray], half_life_days: float = 30.0
    ) -> np.ndarray:
        """Assigns exponential decay weights so recent samples have higher importance."""
        ts = pd.to_datetime(timestamps)
        max_time = ts.max()
        time_deltas_days = (max_time - ts).dt.total_seconds() / (24 * 3600.0)

        # Weight = 2^(-delta / half_life)
        weights = 2.0 ** (-time_deltas_days / half_life_days)
        weights_norm = weights / weights.mean()
        logger.info(f"Temporal decay weights computed (half_life={half_life_days}d). Mean={weights_norm.mean():.2f}")
        return weights_norm.values

    @staticmethod
    def compute_inverse_frequency_weights(
        categories: Union[pd.Series, np.ndarray], power: float = 1.0
    ) -> np.ndarray:
        """Assigns higher weights to minority categorical classes or targets."""
        series = pd.Series(categories)
        counts = series.value_counts()
        total = len(series)

        class_weights = {k: (total / (v + 1e-8)) ** power for k, v in counts.items()}
        weights = series.map(class_weights).values
        return weights / np.mean(weights)

    @staticmethod
    def compute_density_downweights(
        errors: np.ndarray, threshold_std: float = 2.5
    ) -> np.ndarray:
        """Downweights extreme residual outliers using Huber-like weights."""
        errors = np.asarray(errors)
        std = np.std(errors) + 1e-8
        z_scores = np.abs(errors) / std

        weights = np.where(z_scores <= threshold_std, 1.0, threshold_std / (z_scores + 1e-8))
        return weights / np.mean(weights)

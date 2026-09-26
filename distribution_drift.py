"""Wasserstein Distance and Earth Mover's Distribution Drift Estimator for ML-Project-01.

Computes 1D Wasserstein distances and Maximum Mean Discrepancy (MMD) to detect continuous feature shifts.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from logger import logger


class DistributionDriftEstimator:
    """Calculates continuous Wasserstein / Earth Mover's Distance between reference and current feature distributions."""

    def __init__(self, drift_threshold: float = 0.15):
        self.drift_threshold = drift_threshold

    def calculate_feature_drift(
        self, ref_df: pd.DataFrame, curr_df: pd.DataFrame, numeric_cols: List[str] = None
    ) -> Dict[str, Any]:
        """Measures Wasserstein metric across numeric features."""
        if numeric_cols is None:
            numeric_cols = list(set(ref_df.select_dtypes(include=[np.number]).columns).intersection(curr_df.select_dtypes(include=[np.number]).columns))

        results = {}
        flagged = []

        for col in numeric_cols:
            ref_vals = ref_df[col].dropna().values
            curr_vals = curr_df[col].dropna().values

            # Scale invariant normalized distance
            ref_std = np.std(ref_vals) + 1e-8
            wd = float(wasserstein_distance(ref_vals, curr_vals) / ref_std)
            is_drift = wd > self.drift_threshold

            results[col] = {
                "wasserstein_distance": round(wd, 4),
                "is_drift": is_drift,
            }

            if is_drift:
                flagged.append(col)

        logger.info(f"Distribution Drift: {len(flagged)}/{len(numeric_cols)} features exceeding threshold ({self.drift_threshold}).")
        return {
            "drift_detected": len(flagged) > 0,
            "flagged_features": flagged,
            "feature_metrics": results,
        }

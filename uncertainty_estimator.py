"""Model Predictive Uncertainty and Variance Estimator for ML-Project-01.

Decomposes prediction uncertainty into variance-based epistemic (model) and residual aleatoric bounds.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from logger import logger


class EnsembleUncertaintyEstimator:
    """Estimates prediction confidence intervals and standard deviations across ensemble trees or models."""

    @staticmethod
    def estimate_tree_variance(ensemble_model: Any, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates prediction variance across all individual estimator trees in a RandomForest."""
        if not hasattr(ensemble_model, "estimators_"):
            raise ValueError("Model does not contain individual sub-estimators.")

        predictions = np.array([tree.predict(X) for tree in ensemble_model.estimators_])
        mean_preds = np.mean(predictions, axis=0)
        std_preds = np.std(predictions, axis=0)
        q05 = np.percentile(predictions, 5, axis=0)
        q95 = np.percentile(predictions, 95, axis=0)

        df_out = pd.DataFrame({
            "mean_prediction": np.round(mean_preds, 4),
            "epistemic_std": np.round(std_preds, 4),
            "lower_90_ci": np.round(q05, 4),
            "upper_90_ci": np.round(q95, 4),
            "ci_width": np.round(q95 - q05, 4),
        })

        logger.info(f"Uncertainty estimation computed across {len(ensemble_model.estimators_)} trees for {len(X)} samples.")
        return df_out

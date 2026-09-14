"""
Model Explainability and Feature Attribution Engine.
Provides feature importances, permutation importance, and prediction explanations.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from logger import get_logger

logger = get_logger("Explainer")


class ModelExplainer:
    """
    Computes global and local feature importance rankings and model interpretability metrics.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names

    def get_tree_feature_importance(self) -> pd.DataFrame:
        """Extract native feature importances from tree-based estimators."""
        if not hasattr(self.model, "feature_importances_"):
            logger.warning("Underlying model does not provide native 'feature_importances_'.")
            return pd.DataFrame()

        importances = self.model.feature_importances_
        features = self.feature_names or [f"feature_{i}" for i in range(len(importances))]

        df_importance = pd.DataFrame({
            "feature": features,
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        logger.info("Successfully extracted tree feature importances.")
        return df_importance

    def get_permutation_importance(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Compute model-agnostic permutation feature importance on validation data."""
        logger.info(f"Calculating permutation importance with n_repeats={n_repeats}...")
        result = permutation_importance(
            self.model,
            X_val,
            y_val,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        features = self.feature_names or [f"feature_{i}" for i in range(X_val.shape[1])]
        df_perm = pd.DataFrame({
            "feature": features,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

        logger.info("Permutation importance calculation completed.")
        return df_perm

    def explain_single_prediction(
        self,
        sample: np.ndarray,
        baseline: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculates feature attribution estimation for a single observation relative to baseline.
        """
        pred = self.model.predict(sample.reshape(1, -1))[0]
        explanation = {
            "prediction": float(pred),
            "feature_contributions": {}
        }

        if hasattr(self.model, "feature_importances_"):
            weights = self.model.feature_importances_
            features = self.feature_names or [f"feature_{i}" for i in range(len(weights))]
            for f, w, val in zip(features, weights, sample.flatten()):
                explanation["feature_contributions"][f] = {
                    "value": float(val),
                    "global_weight": float(w),
                    "estimated_impact": float(val * w)
                }

        return explanation

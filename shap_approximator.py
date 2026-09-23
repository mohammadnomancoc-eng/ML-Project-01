"""Fast Shapley Attribution and Kernel Approximator for ML-Project-01.

Implements Monte Carlo Shapley value estimation and marginal feature attribution
for black-box models without external binary dependencies.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from logger import logger


class ShapleyApproximator:
    """Computes exact and Monte-Carlo approximated Shapley value feature contributions."""

    def __init__(self, model: Any, background_data: pd.DataFrame, n_samples: int = 50, random_state: int = 42):
        self.model = model
        self.background_data = background_data.select_dtypes(include=[np.number]).fillna(0)
        self.n_samples = n_samples
        self.random_state = random_state
        self.feature_names = self.background_data.columns.tolist()

    def explain_instance(self, instance: pd.Series) -> Dict[str, float]:
        """Calculates Shapley attribution value for each feature of a single instance."""
        np.random.seed(self.random_state)
        instance_num = instance[self.feature_names].values.astype(float)
        bg_samples = self.background_data.sample(
            n=min(self.n_samples, len(self.background_data)),
            replace=True,
            random_state=self.random_state,
        ).values

        n_feats = len(self.feature_names)
        shap_values = np.zeros(n_feats)

        # Baseline expected value
        base_preds = self.model.predict(pd.DataFrame(bg_samples, columns=self.feature_names))
        expected_val = float(np.mean(base_preds))

        # Permutation-based Monte-Carlo Shapley estimator
        for z in bg_samples:
            perm = np.random.permutation(n_feats)
            x_prev = z.copy()
            pred_prev = float(self.model.predict(pd.DataFrame([x_prev], columns=self.feature_names))[0])

            for feat_idx in perm:
                x_curr = x_prev.copy()
                x_curr[feat_idx] = instance_num[feat_idx]
                pred_curr = float(self.model.predict(pd.DataFrame([x_curr], columns=self.feature_names))[0])

                shap_values[feat_idx] += (pred_curr - pred_prev)
                x_prev = x_curr
                pred_prev = pred_curr

        shap_values /= len(bg_samples)

        result = {
            feat: round(float(val), 4)
            for feat, val in zip(self.feature_names, shap_values)
        }
        logger.info(f"Shapley attribution computed for instance across {n_feats} features.")
        return result

    def compute_global_importance(self, df_sample: pd.DataFrame) -> Dict[str, float]:
        """Computes mean absolute Shapley value across a sample of instances."""
        all_shaps = []
        for _, row in df_sample.iterrows():
            shaps = self.explain_instance(row)
            all_shaps.append(list(shaps.values()))

        mean_abs_shaps = np.mean(np.abs(all_shaps), axis=0)
        global_importance = {
            feat: round(float(val), 4)
            for feat, val in sorted(zip(self.feature_names, mean_abs_shaps), key=lambda x: x[1], reverse=True)
        }
        return global_importance

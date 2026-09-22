"""Counterfactual and Minimal Perturbation Explainer for ML-Project-01.

Finds the minimal feature changes needed to flip a model prediction or achieve a desired target threshold.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from logger import logger


class CounterfactualExplainer:
    """Generates actionable counterfactual explanations via gradient-free feature search."""

    def __init__(self, model: Any, feature_names: List[str], step_size: float = 0.05):
        self.model = model
        self.feature_names = feature_names
        self.step_size = step_size

    def explain_instance(
        self,
        instance: pd.Series,
        desired_target_delta: float,
        max_iterations: int = 100,
        feature_bounds: Optional[Dict[str, tuple]] = None,
    ) -> Dict[str, Any]:
        """Calculates minimal perturbations needed to change output by desired_target_delta."""
        current_x = instance.copy()
        initial_pred = float(self.model.predict(pd.DataFrame([current_x], columns=self.feature_names))[0])
        target_pred = initial_pred + desired_target_delta

        numeric_cols = [c for c in self.feature_names if isinstance(current_x[c], (int, float, np.number))]
        changes = {}

        # Greedy directional coordinate descent for minimal perturbation
        for _ in range(max_iterations):
            current_pred = float(self.model.predict(pd.DataFrame([current_x], columns=self.feature_names))[0])
            error = target_pred - current_pred
            if abs(error) < 0.05 * abs(desired_target_delta) + 1e-4:
                break

            best_feature = None
            best_improvement = 0.0
            best_new_val = None

            direction = 1.0 if error > 0 else -1.0

            for col in numeric_cols:
                original_val = float(current_x[col])
                test_step = self.step_size * (abs(original_val) + 1.0) * direction
                test_val = original_val + test_step

                if feature_bounds and col in feature_bounds:
                    min_v, max_v = feature_bounds[col]
                    test_val = max(min_v, min(max_v, test_val))

                temp_x = current_x.copy()
                temp_x[col] = test_val

                test_pred = float(self.model.predict(pd.DataFrame([temp_x], columns=self.feature_names))[0])
                improvement = abs(target_pred - current_pred) - abs(target_pred - test_pred)

                if improvement > best_improvement:
                    best_improvement = improvement
                    best_feature = col
                    best_new_val = test_val

            if best_feature is not None and best_improvement > 0:
                current_x[best_feature] = best_new_val
                changes[best_feature] = {
                    "original": round(float(instance[best_feature]), 4),
                    "counterfactual": round(float(best_new_val), 4),
                    "delta": round(float(best_new_val - instance[best_feature]), 4),
                }
            else:
                break

        final_pred = float(self.model.predict(pd.DataFrame([current_x], columns=self.feature_names))[0])
        logger.info(f"Counterfactual generated with {len(changes)} feature modifications.")

        return {
            "initial_prediction": round(initial_pred, 4),
            "target_prediction": round(target_pred, 4),
            "achieved_prediction": round(final_pred, 4),
            "modified_features": changes,
        }

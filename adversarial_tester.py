"""Adversarial Robustness and Input Perturbation Stress Tester for ML-Project-01.

Simulates noise injection, feature corruption, missingness shocks, and adversarial perturbations
to stress-test trained model stability.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from logger import logger


class AdversarialRobustnessTester:
    """Stress tests model predictions under adversarial perturbations and noise environments."""

    def __init__(self, model: Any, baseline_X: pd.DataFrame, baseline_y: pd.Series):
        self.model = model
        self.baseline_X = baseline_X.copy()
        self.baseline_y = baseline_y.copy()
        self.baseline_preds = self.model.predict(self.baseline_X)
        self.baseline_r2 = float(r2_score(self.baseline_y, self.baseline_preds))
        self.baseline_mse = float(mean_squared_error(self.baseline_y, self.baseline_preds))

    def evaluate_gaussian_noise_robustness(
        self, noise_levels: List[float] = [0.01, 0.05, 0.1, 0.2, 0.5]
    ) -> List[Dict[str, Any]]:
        """Injects relative Gaussian noise into numeric features and records performance degradation."""
        results = []
        std_devs = self.baseline_X.std(numeric_only=True)

        for noise_std in noise_levels:
            perturbed_X = self.baseline_X.copy()
            for col in std_devs.index:
                noise = np.random.normal(0, std_devs[col] * noise_std, size=len(perturbed_X))
                perturbed_X[col] += noise

            noisy_preds = self.model.predict(perturbed_X)
            r2 = float(r2_score(self.baseline_y, noisy_preds))
            mse = float(mean_squared_error(self.baseline_y, noisy_preds))
            drop_pct = ((self.baseline_r2 - r2) / (abs(self.baseline_r2) + 1e-8)) * 100

            results.append({
                "noise_level_sigma": noise_std,
                "perturbed_r2": round(r2, 4),
                "perturbed_mse": round(mse, 4),
                "r2_drop_percentage": round(drop_pct, 2),
            })

        logger.info(f"Gaussian stress testing complete across {len(noise_levels)} noise levels.")
        return results

    def evaluate_feature_dropout_impact(self) -> Dict[str, float]:
        """Evaluates model vulnerability by zeroing out individual features one-by-one."""
        feature_impacts = {}

        for col in self.baseline_X.columns:
            perturbed_X = self.baseline_X.copy()
            perturbed_X[col] = 0.0  # Zero out feature

            preds = self.model.predict(perturbed_X)
            r2 = float(r2_score(self.baseline_y, preds))
            degradation = self.baseline_r2 - r2
            feature_impacts[col] = round(degradation, 4)

        # Sort features by highest vulnerability
        sorted_impacts = dict(sorted(feature_impacts.items(), key=lambda item: item[1], reverse=True))
        logger.info(f"Feature dropout stress test completed for {len(sorted_impacts)} features.")
        return sorted_impacts

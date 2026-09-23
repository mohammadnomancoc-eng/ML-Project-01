"""Conformal Prediction and Coverage Guarantee Estimator for ML-Project-01.

Implements split conformal regression intervals with exact finite-sample coverage guarantees (1 - alpha).
"""

from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
from logger import logger


class ConformalPredictor:
    """Provides non-parametric distribution-free prediction intervals with exact statistical guarantees."""

    def __init__(self, coverage_level: float = 0.90):
        self.coverage_level = coverage_level
        self.alpha = 1.0 - coverage_level
        self.quantile_residual_: float = 0.0
        self.is_calibrated: bool = False

    def calibrate(self, y_val_true: np.ndarray, y_val_pred: np.ndarray) -> "ConformalPredictor":
        """Computes nonconformity scores (residuals) on held-out calibration fold."""
        y_val_true = np.asarray(y_val_true).ravel()
        y_val_pred = np.asarray(y_val_pred).ravel()
        n = len(y_val_true)

        residuals = np.abs(y_val_true - y_val_pred)
        # Finite sample quantile calculation (1 - alpha) * (1 + 1/n)
        q_level = np.clip(np.ceil((n + 1) * (1 - self.alpha)) / n, 0.0, 1.0)
        self.quantile_residual_ = float(np.quantile(residuals, q_level))
        self.is_calibrated = True

        logger.info(
            f"ConformalPredictor calibrated: residual margin = {self.quantile_residual_:.4f} for {self.coverage_level * 100:.1f}% coverage guarantee."
        )
        return self

    def predict_intervals(self, y_pred: np.ndarray) -> pd.DataFrame:
        """Returns point predictions alongside lower and upper conformal prediction bounds."""
        if not self.is_calibrated:
            raise RuntimeError("ConformalPredictor must be calibrated before predicting intervals.")

        y_pred = np.asarray(y_pred).ravel()
        lower_bound = y_pred - self.quantile_residual_
        upper_bound = y_pred + self.quantile_residual_

        return pd.DataFrame({
            "point_prediction": np.round(y_pred, 4),
            "lower_bound": np.round(lower_bound, 4),
            "upper_bound": np.round(upper_bound, 4),
            "interval_width": round(2 * self.quantile_residual_, 4),
        })

    def evaluate_empirical_coverage(
        self, y_test_true: np.ndarray, y_test_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Calculates actual empirical coverage rate on test dataset."""
        intervals = self.predict_intervals(y_test_pred)
        y_test_true = np.asarray(y_test_true).ravel()

        covered = (y_test_true >= intervals["lower_bound"]) & (y_test_true <= intervals["upper_bound"])
        empirical_cov = float(np.mean(covered))

        return {
            "target_coverage": self.coverage_level,
            "empirical_coverage": round(empirical_cov, 4),
            "interval_margin": round(self.quantile_residual_, 4),
            "status": "VALIDATED" if empirical_cov >= (self.coverage_level - 0.05) else "UNDERCOVERED",
        }

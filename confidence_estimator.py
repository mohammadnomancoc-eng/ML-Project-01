"""Prediction Confidence & Conformal Uncertainty Interval Estimator."""

import numpy as np
from typing import Dict, Any, Tuple
from logger import get_logger

logger = get_logger("ConfidenceEstimator")


class ConfidenceEstimator:
    """Calculates conformal prediction error bounds and confidence intervals for regression."""

    def __init__(self, confidence_level: float = 0.90):
        self.confidence_level = confidence_level
        self.q_hat_: float = 0.0

    def fit_calibration(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Computes empirical nonconformity quantiles from a held-out calibration set."""
        residuals = np.abs(y_true - y_pred)
        n = len(residuals)
        quantile_idx = int(np.ceil((n + 1) * self.confidence_level) / n)
        self.q_hat_ = float(np.quantile(residuals, min(1.0, self.confidence_level)))
        logger.info(f"Fitted Conformal Estimator: Q-hat margin = ±{self.q_hat_:.4f} at {int(self.confidence_level*100)}% confidence.")

    def predict_with_intervals(self, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns lower and upper prediction confidence intervals."""
        lower_bound = y_pred - self.q_hat_
        upper_bound = y_pred + self.q_hat_
        return np.round(lower_bound, 4), np.round(upper_bound, 4)

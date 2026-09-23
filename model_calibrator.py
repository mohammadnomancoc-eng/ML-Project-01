"""Probability and Confidence Margin Calibrator for ML-Project-01.

Implements Platt scaling (logistic calibration), Isotonic regression, and regression error interval calibration.
"""

from typing import Dict, Optional, Tuple, Any
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from logger import logger


class ModelCalibrator:
    """Calibrates model predictions into well-calibrated probabilities or confidence distributions."""

    def __init__(self, method: str = "sigmoid"):
        if method not in ["sigmoid", "isotonic"]:
            raise ValueError(f"Unknown calibration method '{method}'. Choose 'sigmoid' or 'isotonic'.")
        self.method = method
        self.calibrator = None

    def fit(self, raw_scores: np.ndarray, y_true: np.ndarray) -> "ModelCalibrator":
        """Fits calibration curve against ground truth labels."""
        raw_scores = np.asarray(raw_scores).ravel()
        y_true = np.asarray(y_true).ravel()

        if self.method == "sigmoid":
            self.calibrator = LogisticRegression(solver="lbfgs")
            self.calibrator.fit(raw_scores.reshape(-1, 1), y_true)
        else:
            self.calibrator = IsotonicRegression(out_of_bounds="clip")
            self.calibrator.fit(raw_scores, y_true)

        logger.info(f"ModelCalibrator fitted with method='{self.method}' across {len(raw_scores)} samples.")
        return self

    def calibrate(self, raw_scores: np.ndarray) -> np.ndarray:
        """Transforms uncalibrated predictions into well-calibrated confidence scores."""
        if self.calibrator is None:
            raise RuntimeError("ModelCalibrator is not fitted yet. Call fit() first.")

        raw_scores = np.asarray(raw_scores).ravel()
        if self.method == "sigmoid":
            probs = self.calibrator.predict_proba(raw_scores.reshape(-1, 1))[:, 1]
        else:
            probs = self.calibrator.predict(raw_scores)

        return np.clip(probs, 0.0, 1.0)

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10
    ) -> float:
        """Calculates the Expected Calibration Error (ECE) metric."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)

        for i in range(n_bins):
            bin_lower, bin_upper = bin_boundaries[i], bin_boundaries[i + 1]
            in_bin = (y_probs >= bin_lower) & (y_probs < bin_upper if i < n_bins - 1 else y_probs <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                acc_in_bin = np.mean(y_true[in_bin])
                conf_in_bin = np.mean(y_probs[in_bin])
                ece += (bin_size / n_total) * abs(acc_in_bin - conf_in_bin)

        return round(float(ece), 4)

"""Streaming Concept Drift and Residual Error Degradation Detector for ML-Project-01.

Implements Page-Hinkley test and rolling window statistical drift tests on model error streams.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from logger import logger


class ConceptDriftDetector:
    """Detects concept drift and sudden or gradual performance degradation in real-time prediction streams."""

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.99,
        window_size: int = 100,
    ):
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        self.window_size = window_size
        self.reset()

    def reset(self) -> None:
        """Resets tracking state."""
        self.mean: float = 0.0
        self.cumulative_sum: float = 0.0
        self.min_cumulative_sum: float = float("inf")
        self.n_samples: int = 0
        self.error_history: List[float] = []
        self.drift_detected_indices: List[int] = []

    def update(self, y_true: float, y_pred: float) -> bool:
        """Processes a single streaming prediction error and checks for concept drift."""
        error = abs(float(y_true) - float(y_pred))
        self.n_samples += 1
        self.error_history.append(error)

        if self.n_samples == 1:
            self.mean = error
        else:
            self.mean = self.alpha * self.mean + (1 - self.alpha) * error

        self.cumulative_sum += (error - self.mean - self.delta)
        if self.cumulative_sum < self.min_cumulative_sum:
            self.min_cumulative_sum = self.cumulative_sum

        ph_statistic = self.cumulative_sum - self.min_cumulative_sum
        has_drift = ph_statistic > self.threshold

        if has_drift:
            logger.warning(
                f"Concept Drift Detected at sample #{self.n_samples}! PH Statistic: {ph_statistic:.2f} > Threshold: {self.threshold}"
            )
            self.drift_detected_indices.append(self.n_samples)
            self.cumulative_sum = 0.0
            self.min_cumulative_sum = 0.0

        return has_drift

    def evaluate_batch(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> Dict[str, Any]:
        """Runs concept drift analysis over an ordered sequence of predictions."""
        self.reset()
        drift_events = []

        for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            drift_flag = self.update(yt, yp)
            if drift_flag:
                drift_events.append(idx)

        recent_window = self.error_history[-self.window_size :] if len(self.error_history) >= self.window_size else self.error_history
        early_window = self.error_history[: self.window_size] if len(self.error_history) >= self.window_size else self.error_history

        recent_mae = float(np.mean(recent_window)) if recent_window else 0.0
        baseline_mae = float(np.mean(early_window)) if early_window else 0.0
        mae_ratio = (recent_mae / (baseline_mae + 1e-8))

        return {
            "total_samples": len(y_true),
            "drift_event_count": len(drift_events),
            "drift_indices": drift_events,
            "baseline_mae": round(baseline_mae, 4),
            "recent_mae": round(recent_mae, 4),
            "mae_degradation_ratio": round(mae_ratio, 4),
            "drift_status": "CRITICAL_DRIFT" if len(drift_events) > 0 else "STABLE",
        }

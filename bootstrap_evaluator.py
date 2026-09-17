"""Non-Parametric Bootstrap Evaluation & Confidence Intervals."""

import numpy as np
from typing import Dict, Any, Callable
from sklearn.metrics import r2_score, mean_squared_error
from logger import get_logger

logger = get_logger("Bootstrap")


class BootstrapEvaluator:
    """Computes statistical confidence intervals for evaluation metrics via bootstrap resampling."""

    def __init__(self, n_bootstraps: int = 1000, confidence_level: float = 0.95, random_state: int = 42):
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.rng = np.random.RandomState(random_state)

    def evaluate_metric_ci(self, y_true: np.ndarray, y_pred: np.ndarray, metric_fn: Callable = r2_score) -> Dict[str, float]:
        """Calculates point estimate, standard error, and 95% confidence intervals."""
        n_samples = len(y_true)
        bootstrapped_scores = []

        for _ in range(self.n_bootstraps):
            indices = self.rng.randint(0, n_samples, size=n_samples)
            score = metric_fn(y_true[indices], y_pred[indices])
            bootstrapped_scores.append(score)

        bootstrapped_scores = np.array(bootstrapped_scores)
        alpha = (1.0 - self.confidence_level) / 2.0
        ci_lower = np.percentile(bootstrapped_scores, alpha * 100)
        ci_upper = np.percentile(bootstrapped_scores, (1.0 - alpha) * 100)

        report = {
            "point_estimate": round(float(metric_fn(y_true, y_pred)), 4),
            "mean_bootstrapped": round(float(np.mean(bootstrapped_scores)), 4),
            "std_error": round(float(np.std(bootstrapped_scores)), 4),
            "ci_lower": round(float(ci_lower), 4),
            "ci_upper": round(float(ci_upper), 4),
            "confidence_level": self.confidence_level,
        }

        logger.info(f"Bootstrap Metric: {report['point_estimate']} (95% CI: [{report['ci_lower']}, {report['ci_upper']}])")
        return report

"""Decision Threshold Optimizer for Classification and Regression Binarization in ML-Project-01.

Optimizes classification cutoff thresholds for F-beta score, Youden's J-statistic,
cost-benefit utility matrices, and precision-recall constraints.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve, fbeta_score
from logger import logger


class ThresholdOptimizer:
    """Calculates optimal decision thresholds tailored to specific business utility constraints."""

    @staticmethod
    def optimize_fbeta(
        y_true: np.ndarray, y_probs: np.ndarray, beta: float = 1.0, n_thresholds: int = 100
    ) -> Dict[str, Any]:
        """Finds the decision threshold maximizing the F-beta score."""
        thresholds = np.linspace(0.01, 0.99, n_thresholds)
        best_threshold = 0.5
        best_score = -1.0

        for thresh in thresholds:
            y_pred = (y_probs >= thresh).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = thresh

        logger.info(f"Optimal F-{beta} Threshold: {best_threshold:.4f} (Score: {best_score:.4f})")
        return {
            "optimal_threshold": round(float(best_threshold), 4),
            "metric_name": f"f_{beta}_score",
            "best_score": round(float(best_score), 4),
        }

    @staticmethod
    def optimize_youdens_j(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
        """Finds optimal threshold maximizing Sensitivity + Specificity - 1 (ROC curve)."""
        fpr, tpr, thresholds = roc_curve(y_true, y_probs)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
        best_j = j_scores[best_idx]

        return {
            "optimal_threshold": round(float(best_threshold), 4),
            "youdens_j": round(float(best_j), 4),
            "sensitivity": round(float(tpr[best_idx]), 4),
            "specificity": round(float(1 - fpr[best_idx]), 4),
        }

    @staticmethod
    def optimize_cost_matrix(
        y_true: np.ndarray,
        y_probs: np.ndarray,
        cost_fp: float = 10.0,
        cost_fn: float = 50.0,
        cost_tp: float = 0.0,
        cost_tn: float = 0.0,
    ) -> Dict[str, Any]:
        """Finds threshold minimizing total expected business costs based on confusion matrix penalty weights."""
        thresholds = np.linspace(0.01, 0.99, 100)
        min_cost = float("inf")
        best_thresh = 0.5

        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)

        for thresh in thresholds:
            preds = (y_probs >= thresh).astype(int)
            tp = np.sum((y_true == 1) & (preds == 1))
            fp = np.sum((y_true == 0) & (preds == 1))
            tn = np.sum((y_true == 0) & (preds == 0))
            fn = np.sum((y_true == 1) & (preds == 0))

            total_cost = (fp * cost_fp) + (fn * cost_fn) + (tp * cost_tp) + (tn * cost_tn)
            if total_cost < min_cost:
                min_cost = total_cost
                best_thresh = thresh

        return {
            "optimal_threshold": round(float(best_thresh), 4),
            "minimum_total_cost": round(float(min_cost), 2),
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
        }

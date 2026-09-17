"""Model Evaluation and Performance Metrics Suite."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
)
from logger import logger


class ModelEvaluator:
    """Computes benchmark evaluation metrics and residual statistics."""

    @staticmethod
    def evaluate_model(
        y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model", n_features: int = None
    ) -> Dict[str, float]:
        """Calculates regression performance metrics including Adjusted R2 and Max Error."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        exp_var = explained_variance_score(y_true, y_pred)
        max_err = max_error(y_true, y_pred)

        # Adjusted R-squared
        n = len(y_true)
        p = n_features if n_features else 1
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        metrics = {
            "rmse": round(float(rmse), 4),
            "mae": round(float(mae), 4),
            "mse": round(float(mse), 4),
            "r2_score": round(float(r2), 4),
            "adjusted_r2": round(float(adj_r2), 4),
            "max_error": round(float(max_err), 4),
            "explained_variance": round(float(exp_var), 4),
        }

        logger.info(f"[{model_name}] Eval — RMSE: {metrics['rmse']} | MAE: {metrics['mae']} | R2: {metrics['r2_score']} | Adj R2: {metrics['adjusted_r2']}")
        return metrics

    @staticmethod
    def compare_models(evaluations: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Constructs a leaderboard comparison table across all evaluated models."""
        df = pd.DataFrame.from_dict(evaluations, orient="index")
        return df.sort_values(by="r2_score", ascending=False)

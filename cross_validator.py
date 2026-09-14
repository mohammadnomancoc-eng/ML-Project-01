"""Cross-Validation and Robust Generalization Assessment."""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from logger import get_logger

logger = get_logger("CrossValidator")


class CrossValidator:
    """Evaluates estimator stability across multiple cross-validation folds."""

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        self.kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        self.scoring = {
            "r2": "r2",
            "neg_rmse": make_scorer(
                lambda y, pred: -np.sqrt(mean_squared_error(y, pred)),
                greater_is_better=True,
            ),
        }

    def evaluate(self, estimator: Any, X: pd.DataFrame, y: pd.Series, model_name: str = "Model") -> Dict[str, Any]:
        """Runs k-fold cross-validation and reports mean and variance of metrics."""
        logger.info(f"Running {self.kfold.n_splits}-fold CV for [{model_name}]...")
        cv_results = cross_validate(estimator, X, y, cv=self.kfold, scoring=self.scoring, return_train_score=True)

        r2_test = cv_results["test_r2"]
        rmse_test = -cv_results["test_neg_rmse"]

        summary = {
            "model": model_name,
            "mean_r2": round(float(np.mean(r2_test)), 4),
            "std_r2": round(float(np.std(r2_test)), 4),
            "mean_rmse": round(float(np.mean(rmse_test)), 4),
            "std_rmse": round(float(np.std(rmse_test)), 4),
            "fit_time_mean": round(float(np.mean(cv_results["fit_time"])), 4),
        }

        logger.info(f"[{model_name}] CV R2: {summary['mean_r2']} ± {summary['std_r2']} | RMSE: {summary['mean_rmse']}")
        return summary

"""Hyperparameter Optimization using Grid and Randomized Search."""

import pandas as pd
from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from config import config
from logger import logger


class HyperparameterTuner:
    """Tunes machine learning estimators via cross-validated parameter sweeps."""

    def __init__(self, cv: int = 5, n_iter: int = 10, scoring: str = "r2"):
        self.cv = cv
        self.n_iter = n_iter
        self.scoring = scoring
        self.best_params_: Dict[str, Any] = {}
        self.best_scores_: Dict[str, float] = {}

    def tune_grid_search(
        self, estimator: Any, param_grid: Dict[str, list], X: pd.DataFrame, y: pd.Series, model_name: str
    ) -> Any:
        """Runs exhaustive GridSearchCV optimization."""
        logger.info(f"Running GridSearchCV for [{model_name}] with {len(param_grid)} hyperparameter keys...")
        grid = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            cv=self.cv,
            scoring=self.scoring,
            n_jobs=-1,
        )
        grid.fit(X, y)

        self.best_params_[model_name] = grid.best_params_
        self.best_scores_[model_name] = round(float(grid.best_score_), 4)
        logger.info(f"[{model_name}] Best Score: {self.best_scores_[model_name]} | Params: {self.best_params_[model_name]}")
        return grid.best_estimator_

    def tune_random_search(
        self, estimator: Any, param_distributions: Dict[str, list], X: pd.DataFrame, y: pd.Series, model_name: str
    ) -> Any:
        """Runs RandomizedSearchCV parameter distribution sweep."""
        logger.info(f"Running RandomizedSearchCV for [{model_name}] (n_iter={self.n_iter})...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=self.n_iter,
            cv=self.cv,
            scoring=self.scoring,
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X, y)

        self.best_params_[model_name] = search.best_params_
        self.best_scores_[model_name] = round(float(search.best_score_), 4)
        logger.info(f"[{model_name}] RandomSearch Best Score: {self.best_scores_[model_name]} | Params: {self.best_params_[model_name]}")
        return search.best_estimator_

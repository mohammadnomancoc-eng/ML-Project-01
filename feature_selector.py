"""Automated Feature Selection Module."""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
)
from sklearn.linear_model import Ridge
from logger import get_logger

logger = get_logger("FeatureSelector")


class FeatureSelector:
    """Selects high-impact features using variance thresholds, statistical tests, and RFE."""

    def __init__(self, variance_threshold: float = 0.01, k_best: int = 6):
        self.variance_threshold = variance_threshold
        self.k_best = k_best
        self.selected_features_: List[str] = []

    def remove_low_variance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes features with variance below threshold."""
        selector = VarianceThreshold(threshold=self.variance_threshold)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        selector.fit(df[numeric_cols])

        kept_cols = numeric_cols[selector.get_support()].tolist()
        logger.info(f"Variance filter: kept {len(kept_cols)}/{len(numeric_cols)} features.")
        return df[kept_cols]

    def select_k_best_features(
        self, X: pd.DataFrame, y: pd.Series, method: str = "f_regression"
    ) -> List[str]:
        """Selects top K features using univariate statistical tests."""
        score_func = mutual_info_regression if method == "mutual_info" else f_regression
        selector = SelectKBest(score_func=score_func, k=min(self.k_best, X.shape[1]))
        selector.fit(X, y)

        self.selected_features_ = X.columns[selector.get_support()].tolist()
        logger.info(f"Selected Top {len(self.selected_features_)} features via {method}: {self.selected_features_}")
        return self.selected_features_

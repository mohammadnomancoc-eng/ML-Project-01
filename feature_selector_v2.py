"""Advanced Recursive Feature Elimination (RFE) & Cross-Validation Selector for ML-Project-01.

Selects the most predictive feature subset using greedy backward recursive elimination.
"""

from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Ridge
from logger import logger


class RecursiveFeatureSelector:
    """Automates cross-validated feature elimination to maximize generalization performance."""

    def __init__(self, min_features_to_select: int = 3, cv_folds: int = 5):
        self.min_features_to_select = min_features_to_select
        self.cv_folds = cv_folds
        self.rfecv = None
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RecursiveFeatureSelector":
        """Fits RFECV using Ridge regression as base estimator."""
        numeric_X = X.select_dtypes(include=[np.number])
        estimator = Ridge(alpha=1.0)

        self.rfecv = RFECV(
            estimator=estimator,
            step=1,
            cv=self.cv_folds,
            scoring="r2",
            min_features_to_select=self.min_features_to_select,
            n_jobs=-1,
        )
        self.rfecv.fit(numeric_X, y)

        self.selected_features_ = list(numeric_X.columns[self.rfecv.support_])
        logger.info(f"RFECV selected {len(self.selected_features_)}/{numeric_X.shape[1]} optimal features: {self.selected_features_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filters DataFrame to only retain optimal feature columns."""
        if not self.selected_features_:
            raise RuntimeError("RecursiveFeatureSelector is not fitted yet.")
        keep_cols = [c for c in self.selected_features_ if c in X.columns]
        return X[keep_cols]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fits selector and returns transformed DataFrame."""
        return self.fit(X, y).transform(X)

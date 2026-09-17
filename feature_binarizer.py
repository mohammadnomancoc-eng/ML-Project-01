"""Feature Binarization and Quantile Discretization Transformer."""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from sklearn.base import BaseEstimator, TransformerMixin
from logger import get_logger

logger = get_logger("FeatureBinarizer")


class FeatureDiscretizer(BaseEstimator, TransformerMixin):
    """Binarizes numerical continuous variables and creates quantile bin buckets."""

    def __init__(self, n_bins: int = 5, strategy: str = "quantile"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Calculates bin cutoffs for numerical columns."""
        X = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            quantiles = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(X[col].dropna(), quantiles)
            edges[0] = -np.inf
            edges[-1] = np.inf
            self.bin_edges_[col] = np.unique(edges)

        logger.info(f"Fitted FeatureDiscretizer on {len(numeric_cols)} columns with {self.n_bins} bins.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms numeric columns into discrete bucket indices."""
        X = X.copy()
        for col, edges in self.bin_edges_.items():
            if col in X.columns:
                X[f"{col}_binned"] = pd.cut(X[col], bins=edges, labels=False, include_lowest=True)
        return X

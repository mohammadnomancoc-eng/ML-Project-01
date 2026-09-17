"""Feature Hashing Transformer for High-Cardinality Categoricals."""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import FeatureHasher
from logger import get_logger

logger = get_logger("FeatureHasher")


class CategoricalHasher(BaseEstimator, TransformerMixin):
    """Encodes unbounded high-cardinality categorical features using the hashing trick."""

    def __init__(self, n_features: int = 16, input_type: str = "string"):
        self.n_features = n_features
        self.input_type = input_type
        self.hasher = FeatureHasher(n_features=n_features, input_type=input_type)

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """No state required for feature hashing."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Hashes string columns into fixed-width numeric feature matrix."""
        logger.info(f"Hashing categorical columns into {self.n_features} fixed features...")
        string_records = X.astype(str).values.tolist()
        hashed_matrix = self.hasher.transform(string_records).toarray()

        col_names = [f"hash_feat_{i+1}" for i in range(self.n_features)]
        return pd.DataFrame(hashed_matrix, columns=col_names, index=X.index)

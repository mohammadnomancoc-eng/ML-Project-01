"""Fixed-Memory Feature Hashing (Hashing Trick) for High-Cardinality Data in ML-Project-01.

Implements MurmurHash / hashlib hashing trick with sign bit hashing to encode arbitrarily large categorical spaces
into fixed-size memory representations with zero vocabulary dictionary overhead.
"""

import hashlib
from typing import List, Optional
import numpy as np
import pandas as pd
from logger import logger


class FeatureHashEncoder:
    """Encodes high-cardinality categorical features into fixed-width numerical vector spaces."""

    def __init__(self, n_features: int = 16, alternate_sign: bool = True):
        self.n_features = n_features
        self.alternate_sign = alternate_sign
        self.feature_columns_: List[str] = [f"hash_feat_{i}" for i in range(n_features)]

    def _hash_token(self, token: str) -> tuple:
        """Hashes a string token to a column index and sign (-1 or 1)."""
        encoded = str(token).encode("utf-8")
        h = int(hashlib.md5(encoded).hexdigest(), 16)
        col_idx = h % self.n_features
        sign = 1
        if self.alternate_sign:
            sign = 1 if ((h >> 16) & 1) == 0 else -1
        return col_idx, sign

    def transform(self, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Transforms high-cardinality categorical columns into fixed-width hashed feature space."""
        df_out = df.copy()
        if categorical_cols is None:
            categorical_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()

        if not categorical_cols:
            return df_out

        n_samples = len(df_out)
        hashed_matrix = np.zeros((n_samples, self.n_features))

        for col in categorical_cols:
            for row_idx, val in enumerate(df_out[col]):
                if pd.notnull(val):
                    col_idx, sign = self._hash_token(f"{col}={val}")
                    hashed_matrix[row_idx, col_idx] += sign

        hash_df = pd.DataFrame(hashed_matrix, columns=self.feature_columns_, index=df_out.index)
        result = pd.concat([df_out.drop(columns=categorical_cols), hash_df], axis=1)
        logger.info(f"Hashed {len(categorical_cols)} high-cardinality columns into {self.n_features} hashed features.")
        return result

    def fit_transform(self, df: pd.DataFrame, categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Fits encoder (stateless) and transforms DataFrame."""
        return self.transform(df, categorical_cols)

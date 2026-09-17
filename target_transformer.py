"""Target Variable Transformations (Log, Box-Cox, Yeo-Johnson)."""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.preprocessing import PowerTransformer
from logger import get_logger

logger = get_logger("TargetTransformer")


class TargetTransformer:
    """Applies power transforms and log mappings with exact inverse reconstruction."""

    def __init__(self, method: str = "yeo-johnson"):
        self.method = method
        self.transformer = PowerTransformer(method=method, standardize=True) if method in ("box-cox", "yeo-johnson") else None

    def fit_transform(self, y: pd.Series) -> np.ndarray:
        """Fits power transform on target array."""
        logger.info(f"Applying target transformation with method: '{self.method}'...")
        y_arr = y.values.reshape(-1, 1)

        if self.method == "log1p":
            return np.log1p(np.maximum(0, y_arr)).flatten()
        elif self.transformer:
            return self.transformer.fit_transform(y_arr).flatten()
        return y_arr.flatten()

    def inverse_transform(self, y_transformed: np.ndarray) -> np.ndarray:
        """Inversely transforms predictions back to original target domain."""
        y_arr = np.array(y_transformed).reshape(-1, 1)

        if self.method == "log1p":
            return np.expm1(y_arr).flatten()
        elif self.transformer:
            return self.transformer.inverse_transform(y_arr).flatten()
        return y_arr.flatten()

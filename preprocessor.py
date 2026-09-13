"""Data Cleaning and Preprocessing Pipeline."""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from logger import logger


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Handles missing values, outlier clipping, and data sanitation."""

    def __init__(self, iqr_multiplier: float = 1.5):
        self.iqr_multiplier = iqr_multiplier
        self.numeric_medians_: dict = {}
        self.categorical_modes_: dict = {}
        self.iqr_bounds_: dict = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns imputation statistics and IQR bounds from training data."""
        logger.info("Fitting DataPreprocessor on training set...")
        X = X.copy()

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        # Numeric medians & IQR
        for col in numeric_cols:
            self.numeric_medians_[col] = X[col].median()
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.iqr_bounds_[col] = (
                q1 - self.iqr_multiplier * iqr,
                q3 + self.iqr_multiplier * iqr,
            )

        # Categorical modes
        for col in categorical_cols:
            mode_series = X[col].mode()
            self.categorical_modes_[col] = mode_series[0] if not mode_series.empty else "Unknown"

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies imputation and outlier clipping."""
        X = X.copy()

        # Numeric Imputation and Outlier Clipping
        for col, median_val in self.numeric_medians_.items():
            if col in X.columns:
                X[col] = X[col].fillna(median_val)
                lower_bound, upper_bound = self.iqr_bounds_[col]
                X[col] = np.clip(X[col], lower_bound, upper_bound)

        # Categorical Imputation
        for col, mode_val in self.categorical_modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mode_val)

        return X

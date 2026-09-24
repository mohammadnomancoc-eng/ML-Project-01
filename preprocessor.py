"""Data Cleaning and Preprocessing Pipeline."""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from logger import logger


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Handles missing values, infinite values, outlier clipping, and data sanitation."""

    def __init__(self, iqr_multiplier: float = 1.5, outlier_strategy: str = "iqr", drop_constants: bool = True):
        self.iqr_multiplier = iqr_multiplier
        self.outlier_strategy = outlier_strategy
        self.drop_constants = drop_constants
        self.numeric_medians_: dict = {}
        self.categorical_modes_: dict = {}
        self.iqr_bounds_: dict = {}
        self.constant_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Learns imputation statistics and IQR/MAD bounds from training data."""
        logger.info("Fitting DataPreprocessor on training set...")
        X = X.copy()

        # Detect constant columns
        if self.drop_constants:
            self.constant_columns_ = [col for col in X.columns if X[col].nunique() <= 1]
            if self.constant_columns_:
                logger.info(f"Identified {len(self.constant_columns_)} constant columns to drop: {self.constant_columns_}")
            X = X.drop(columns=self.constant_columns_)

        numeric_cols = X.select_dtypes(include=[np.number]).columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns

        # Numeric medians & bounds
        for col in numeric_cols:
            series = X[col].replace([np.inf, -np.inf], np.nan)
            med = series.median()
            self.numeric_medians_[col] = med

            if self.outlier_strategy == "mad":
                mad = np.median(np.abs(series.dropna() - med))
                bound_margin = self.iqr_multiplier * 1.4826 * (mad + 1e-8)
                self.iqr_bounds_[col] = (med - bound_margin, med + bound_margin)
            else:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
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
        """Applies imputation, infinity sanitization, and outlier clipping."""
        X = X.copy()

        # Drop constant columns learned during fit
        if self.constant_columns_:
            X = X.drop(columns=[c for c in self.constant_columns_ if c in X.columns])

        # Numeric Imputation and Outlier Clipping
        for col, median_val in self.numeric_medians_.items():
            if col in X.columns:
                X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(median_val)
                lower_bound, upper_bound = self.iqr_bounds_[col]
                X[col] = np.clip(X[col], lower_bound, upper_bound)

        # Categorical Imputation
        for col, mode_val in self.categorical_modes_.items():
            if col in X.columns:
                X[col] = X[col].fillna(mode_val)

        return X

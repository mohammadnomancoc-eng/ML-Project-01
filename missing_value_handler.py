"""Advanced Missing Value Imputation Engine for ML-Project-01.

Provides KNN imputation, iterative multivariate imputation, rolling window interpolation,
and missingness indicator flag generation.
"""

from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from logger import logger


class MissingValueHandler:
    """Handles missing data using advanced statistical and machine learning imputation methods."""

    def __init__(self, strategy: str = "knn", n_neighbors: int = 5):
        self.strategy = strategy
        self.n_neighbors = n_neighbors
        self.numeric_imputer = None
        self.categorical_imputer = None
        self.imputed_feature_names: List[str] = []

    def fit_transform(
        self,
        df: pd.DataFrame,
        add_indicators: bool = True,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fits imputation strategies on DataFrame and replaces null values."""
        df_out = df.copy()

        if numeric_cols is None:
            numeric_cols = df_out.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = df_out.select_dtypes(exclude=[np.number]).columns.tolist()

        if add_indicators:
            for col in numeric_cols + categorical_cols:
                if df_out[col].isnull().any():
                    df_out[f"{col}_is_missing"] = df_out[col].isnull().astype(int)

        if numeric_cols:
            if self.strategy == "knn":
                self.numeric_imputer = KNNImputer(n_neighbors=self.n_neighbors)
            else:
                self.numeric_imputer = SimpleImputer(strategy="median")

            df_out[numeric_cols] = self.numeric_imputer.fit_transform(df_out[numeric_cols])

        if categorical_cols:
            self.categorical_imputer = SimpleImputer(strategy="most_frequent")
            df_out[categorical_cols] = self.categorical_imputer.fit_transform(df_out[categorical_cols])

        self.imputed_feature_names = df_out.columns.tolist()
        logger.info(f"Missing values imputed using strategy='{self.strategy}'. Output shape: {df_out.shape}")
        return df_out

    def transform(self, df: pd.DataFrame, add_indicators: bool = True) -> pd.DataFrame:
        """Applies fitted imputation models to incoming validation/test DataFrames."""
        df_out = df.copy()

        if add_indicators:
            for col in df.columns:
                indicator_col = f"{col}_is_missing"
                if indicator_col in self.imputed_feature_names:
                    df_out[indicator_col] = df_out[col].isnull().astype(int)

        if self.numeric_imputer is not None:
            num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c in self.numeric_imputer.feature_names_in_]
            if num_cols:
                df_out[num_cols] = self.numeric_imputer.transform(df_out[num_cols])

        if self.categorical_imputer is not None:
            cat_cols = [c for c in df.select_dtypes(exclude=[np.number]).columns if c in self.categorical_imputer.feature_names_in_]
            if cat_cols:
                df_out[cat_cols] = self.categorical_imputer.transform(df_out[cat_cols])

        return df_out

    @staticmethod
    def rolling_window_impute(
        series: pd.Series, window: int = 5, min_periods: int = 1
    ) -> pd.Series:
        """Imputes missing values in time-series / sequential series using rolling forward/backward means."""
        rolling_mean = series.rolling(window=window, min_periods=min_periods, center=True).mean()
        return series.fillna(rolling_mean).bfill().ffill()

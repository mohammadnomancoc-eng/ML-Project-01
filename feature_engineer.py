"""Feature Engineering and Transformation Module."""

import numpy as np
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from config import config
from logger import logger


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Performs numerical scaling, one-hot encoding, and feature interactions."""

    def __init__(self, scaling_method: str = config.SCALING_METHOD):
        self.scaling_method = scaling_method
        self.column_transformer_: Optional[ColumnTransformer] = None
        self.feature_names_out_: List[str] = []

    def _get_scaler(self):
        if self.scaling_method == "minmax":
            return MinMaxScaler()
        elif self.scaling_method == "robust":
            return RobustScaler()
        return StandardScaler()

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        """Fits scalers and encoders on specific feature columns."""
        logger.info(f"Fitting FeatureEngineer with '{self.scaling_method}' scaling...")
        X = X.copy()

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

        transformers = []
        if numeric_features:
            transformers.append(("num", self._get_scaler(), numeric_features))
        if categorical_features:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
            )

        self.column_transformer_ = ColumnTransformer(transformers=transformers)
        self.column_transformer_.fit(X)

        # Store output feature names
        self.feature_names_out_ = self.column_transformer_.get_feature_names_out().tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms features and returns DataFrame with column headers."""
        if self.column_transformer_ is None:
            raise ValueError("FeatureEngineer must be fitted before transforming.")

        transformed_array = self.column_transformer_.transform(X)
        return pd.DataFrame(transformed_array, columns=self.feature_names_out_, index=X.index)

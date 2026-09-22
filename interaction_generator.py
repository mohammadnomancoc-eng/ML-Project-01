"""Feature Interaction and Cross-Term Synthesizer for ML-Project-01.

Generates pairwise polynomial products, ratios, differences, and selective non-linear interaction terms.
"""

from itertools import combinations
from typing import List, Optional
import numpy as np
import pandas as pd
from logger import logger


class FeatureInteractionGenerator:
    """Generates cross-feature interactions and ratio signals for tabular datasets."""

    def __init__(
        self,
        include_products: bool = True,
        include_ratios: bool = True,
        include_differences: bool = False,
        epsilon: float = 1e-6,
    ):
        self.include_products = include_products
        self.include_ratios = include_ratios
        self.include_differences = include_differences
        self.epsilon = epsilon
        self.generated_columns: List[str] = []

    def fit_transform(
        self, df: pd.DataFrame, numeric_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Computes interaction features and appends them to input DataFrame."""
        df_out = df.copy()
        if numeric_columns is None:
            numeric_columns = df_out.select_dtypes(include=[np.number]).columns.tolist()

        new_features = {}

        for col_a, col_b in combinations(numeric_columns, 2):
            if self.include_products:
                prod_col = f"{col_a}_x_{col_b}"
                new_features[prod_col] = df_out[col_a] * df_out[col_b]

            if self.include_ratios:
                ratio_col = f"{col_a}_div_{col_b}"
                new_features[ratio_col] = df_out[col_a] / (df_out[col_b].abs() + self.epsilon)

            if self.include_differences:
                diff_col = f"{col_a}_minus_{col_b}"
                new_features[diff_col] = df_out[col_a] - df_out[col_b]

        interactions_df = pd.DataFrame(new_features, index=df_out.index)
        self.generated_columns = list(new_features.keys())

        result = pd.concat([df_out, interactions_df], axis=1)
        logger.info(f"Generated {len(self.generated_columns)} interaction features. New shape: {result.shape}")
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies pre-calculated interaction definitions to test/validation DataFrames."""
        df_out = df.copy()
        new_features = {}

        for feat_name in self.generated_columns:
            if "_x_" in feat_name:
                col_a, col_b = feat_name.split("_x_")
                if col_a in df_out and col_b in df_out:
                    new_features[feat_name] = df_out[col_a] * df_out[col_b]
            elif "_div_" in feat_name:
                col_a, col_b = feat_name.split("_div_")
                if col_a in df_out and col_b in df_out:
                    new_features[feat_name] = df_out[col_a] / (df_out[col_b].abs() + self.epsilon)
            elif "_minus_" in feat_name:
                col_a, col_b = feat_name.split("_minus_")
                if col_a in df_out and col_b in df_out:
                    new_features[feat_name] = df_out[col_a] - df_out[col_b]

        interactions_df = pd.DataFrame(new_features, index=df_out.index)
        return pd.concat([df_out, interactions_df], axis=1)

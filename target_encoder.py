"""Out-of-Fold Target Encoder with Bayesian Smoothing for ML-Project-01.

Encodes categorical features using smoothed target statistics with K-fold cross-fitting
to prevent target leakage and overfitting.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from logger import logger


class OutOfFoldTargetEncoder:
    """Encodes categorical columns with smoothed target statistics using K-fold out-of-fold scheme."""

    def __init__(self, smoothing: float = 10.0, cv_folds: int = 5, random_state: int = 42):
        self.smoothing = smoothing
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.global_mean_: float = 0.0
        self.category_mappings_: Dict[str, Dict[str, float]] = {}

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, categorical_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Fits target statistics with out-of-fold cross validation and encodes categorical columns."""
        df_out = X.copy()
        if categorical_columns is None:
            categorical_columns = df_out.select_dtypes(exclude=[np.number]).columns.tolist()

        self.global_mean_ = float(y.mean())
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for col in categorical_columns:
            encoded_col = np.zeros(len(df_out))

            # Out of fold calculation
            for train_idx, val_idx in kf.split(df_out):
                train_df = df_out.iloc[train_idx]
                train_y = y.iloc[train_idx]

                # Compute smoothed mean for training fold
                stats = train_y.groupby(train_df[col]).agg(["count", "mean"])
                counts = stats["count"]
                means = stats["mean"]
                smoothed = (counts * means + self.smoothing * self.global_mean_) / (counts + self.smoothing)
                
                # Apply to validation fold
                val_categories = df_out.iloc[val_idx][col]
                encoded_col[val_idx] = val_categories.map(smoothed).fillna(self.global_mean_).values

            df_out[f"{col}_target_enc"] = encoded_col

            # Compute full dataset smoothed stats for future transform() calls
            full_stats = y.groupby(df_out[col]).agg(["count", "mean"])
            full_counts = full_stats["count"]
            full_means = full_stats["mean"]
            self.category_mappings_[col] = (
                (full_counts * full_means + self.smoothing * self.global_mean_) / (full_counts + self.smoothing)
            ).to_dict()

        logger.info(f"Target encoding completed for {len(categorical_columns)} categorical features.")
        return df_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encodes incoming validation/test DataFrames using fitted global smoothed stats."""
        df_out = X.copy()
        for col, mapping in self.category_mappings_.items():
            if col in df_out.columns:
                df_out[f"{col}_target_enc"] = df_out[col].map(mapping).fillna(self.global_mean_)
        return df_out

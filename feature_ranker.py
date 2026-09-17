"""Multi-Method Ensemble Feature Ranking Suite."""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from logger import get_logger

logger = get_logger("FeatureRanker")


class EnsembleFeatureRanker:
    """Combines multiple feature ranking techniques into a normalized consensus rank."""

    @staticmethod
    def compute_consensus_rank(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Computes ranks across Tree Importance, Mutual Info, and Absolute Correlation."""
        logger.info(f"Computing consensus feature rank across {X.shape[1]} features...")
        numeric_X = X.select_dtypes(include=[np.number])

        # 1. Random Forest Importance
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(numeric_X, y)
        rf_ranks = pd.Series(rf.feature_importances_, index=numeric_X.columns).rank(ascending=False)

        # 2. Mutual Information
        mi = mutual_info_regression(numeric_X, y, random_state=42)
        mi_ranks = pd.Series(mi, index=numeric_X.columns).rank(ascending=False)

        # 3. Absolute Pearson Correlation
        corr = numeric_X.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))
        corr_ranks = corr.rank(ascending=False)

        summary_df = pd.DataFrame({
            "rf_rank": rf_ranks,
            "mi_rank": mi_ranks,
            "corr_rank": corr_ranks,
        })
        summary_df["mean_rank"] = summary_df.mean(axis=1)
        summary_df = summary_df.sort_values(by="mean_rank", ascending=True)

        logger.info(f"Top 3 consensus features: {summary_df.index[:3].tolist()}")
        return summary_df

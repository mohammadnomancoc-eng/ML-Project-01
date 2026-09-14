"""Unsupervised Anomaly and Outlier Detection Module."""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from logger import get_logger

logger = get_logger("OutlierDetector")


class OutlierDetector:
    """Detects multi-dimensional anomalies using Isolation Forest and LOF."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination, random_state=random_state, n_jobs=-1
        )
        self.lof = LocalOutlierFactor(contamination=contamination, n_jobs=-1)

    def fit_predict_isolation_forest(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fits Isolation Forest and returns anomaly labels (-1: anomaly, 1: inlier) and anomaly scores."""
        numeric_df = df.select_dtypes(include=[np.number])
        logger.info(f"Running IsolationForest outlier detection on {len(numeric_df)} samples...")
        labels = self.iso_forest.fit_predict(numeric_df)
        scores = self.iso_forest.decision_function(numeric_df)

        n_anomalies = np.sum(labels == -1)
        logger.info(f"IsolationForest detected {n_anomalies} anomalies ({n_anomalies/len(df)*100:.1f}%)")
        return labels, scores

    def filter_clean_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns dataframe with anomalies removed."""
        labels, _ = self.fit_predict_isolation_forest(df)
        return df[labels == 1].reset_index(drop=True)

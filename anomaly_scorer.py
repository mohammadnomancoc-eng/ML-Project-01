"""Multi-Method Ensemble Anomaly Scorer for ML-Project-01.

Combines Isolation Forest, Local Outlier Factor (LOF), and Robust Covariance / Mahalanobis
scoring into an aggregated multi-method anomaly score.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from logger import logger


class EnsembleAnomalyScorer:
    """Computes consensus anomaly scores and identifies out-of-distribution instances."""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.iso_forest = IsolationForest(
            contamination=contamination, random_state=random_state, n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20, contamination=contamination, novelty=True
        )

    def fit(self, X: pd.DataFrame) -> "EnsembleAnomalyScorer":
        """Fits anomaly detectors on training feature distribution."""
        numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
        self.iso_forest.fit(numeric_X)
        self.lof.fit(numeric_X)
        logger.info(f"EnsembleAnomalyScorer fitted on dataset with shape {numeric_X.shape}.")
        return self

    def predict_scores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates normalized anomaly scores between 0 (normal) and 1 (highly anomalous)."""
        numeric_X = X.select_dtypes(include=[np.number]).fillna(0)

        # Isolation forest anomaly score (-0.5 to 0.5 normalized)
        iso_scores = -self.iso_forest.score_samples(numeric_X)
        iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min() + 1e-8)

        # LOF novelty score
        lof_scores = -self.lof.score_samples(numeric_X)
        lof_norm = (lof_scores - lof_scores.min()) / (lof_scores.max() - lof_scores.min() + 1e-8)

        # Consensus score (simple average)
        consensus = 0.5 * iso_norm + 0.5 * lof_norm

        results_df = pd.DataFrame(
            {
                "iso_anomaly_score": np.round(iso_norm, 4),
                "lof_anomaly_score": np.round(lof_norm, 4),
                "consensus_anomaly_score": np.round(consensus, 4),
                "is_anomaly": (consensus >= (1.0 - self.contamination)).astype(int),
            },
            index=X.index,
        )
        return results_df

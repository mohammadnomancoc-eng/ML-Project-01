"""Feature Clustering and Redundancy Reduction Engine for ML-Project-01.

Clusters multicollinear and highly correlated features using correlation distance
and agglomerative hierarchical clustering, and selects representative centroid features.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.cluster import hierarchy
from logger import logger


class FeatureClusterer:
    """Groups correlated features into distinct clusters and selects representative subsets."""

    def __init__(self, correlation_threshold: float = 0.75, linkage_method: str = "average"):
        self.correlation_threshold = correlation_threshold
        self.linkage_method = linkage_method
        self.cluster_assignments_: Dict[str, int] = {}
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame) -> "FeatureClusterer":
        """Fits feature clusters based on pairwise correlation distance matrix."""
        numeric_df = X.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            self.selected_features_ = numeric_df.columns.tolist()
            return self

        corr_matrix = numeric_df.corr().abs().fillna(0)
        dist_matrix = 1.0 - corr_matrix.values
        np.fill_diagonal(dist_matrix, 0)
        dist_matrix = np.clip(dist_matrix, 0, 1)

        condensed_dist = squareform(dist_matrix, checks=False)
        linkage = hierarchy.linkage(condensed_dist, method=self.linkage_method)

        distance_threshold = 1.0 - self.correlation_threshold
        clusters = hierarchy.fcluster(linkage, t=distance_threshold, criterion="distance")

        feature_names = numeric_df.columns.tolist()
        self.cluster_assignments_ = {feat: int(clust) for feat, clust in zip(feature_names, clusters)}

        # Pick feature with highest variance or first in cluster as representative
        selected = []
        unique_clusters = set(clusters)
        for cl in unique_clusters:
            feats_in_cluster = [f for f, c in self.cluster_assignments_.items() if c == cl]
            # Select feature with highest variance
            best_feat = max(feats_in_cluster, key=lambda f: float(numeric_df[f].var()))
            selected.append(best_feat)

        self.selected_features_ = selected
        logger.info(
            f"Feature clustering reduced {len(feature_names)} features into {len(selected)} independent clusters."
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Filters DataFrame to include only representative cluster features."""
        non_numeric = [c for c in X.columns if c not in self.cluster_assignments_]
        keep_cols = [c for c in self.selected_features_ if c in X.columns] + non_numeric
        return X[keep_cols]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fits clustering and transforms the dataset in a single step."""
        return self.fit(X).transform(X)

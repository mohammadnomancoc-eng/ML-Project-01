"""Gaussian Copula Tabular Data Synthesizer for ML-Project-01.

Synthesizes high-fidelity synthetic tabular records while preserving marginal feature distributions,
skewness, and empirical inter-feature covariance structures.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from logger import logger


class TabularCopulaSynthesizer:
    """Synthesizes realistic tabular datasets via empirical probability integral transform & Gaussian Copula."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.empirical_distributions_: Dict[str, np.ndarray] = {}
        self.categorical_probs_: Dict[str, Dict[str, float]] = {}
        self.correlation_matrix_: Optional[np.ndarray] = None

    def fit(self, df: pd.DataFrame) -> "TabularCopulaSynthesizer":
        """Fits empirical cumulative distributions and correlation matrix."""
        np.random.seed(self.random_state)
        self.numeric_cols_ = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = df.select_dtypes(exclude=[np.number]).columns.tolist()

        # Learn empirical distributions
        gaussian_matrix = []
        for col in self.numeric_cols_:
            col_data = df[col].dropna().values
            self.empirical_distributions_[col] = col_data
            
            # Rank transform to uniform, then standard normal
            ranks = stats.rankdata(df[col].fillna(np.median(col_data))) / (len(df) + 1.0)
            z = stats.norm.ppf(ranks)
            gaussian_matrix.append(z)

        if gaussian_matrix:
            self.correlation_matrix_ = np.corrcoef(np.vstack(gaussian_matrix))
        else:
            self.correlation_matrix_ = np.eye(1)

        # Learn categorical marginal probability distributions
        for col in self.categorical_cols_:
            probs = df[col].value_counts(normalize=True).to_dict()
            self.categorical_probs_[col] = probs

        logger.info(
            f"TabularCopulaSynthesizer fitted on {len(self.numeric_cols_)} numeric and {len(self.categorical_cols_)} categorical features."
        )
        return self

    def sample(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generates n_samples synthetic observations respecting original covariance structure."""
        np.random.seed(self.random_state)
        n_num = len(self.numeric_cols_)

        synthetic_dict = {}

        if n_num > 0:
            mean = np.zeros(n_num)
            synthetic_gaussian = np.random.multivariate_normal(
                mean, self.correlation_matrix_, size=n_samples
            )
            # Transform Gaussian copula back to empirical margins via quantile matching
            synthetic_uniform = stats.norm.cdf(synthetic_gaussian)

            for idx, col in enumerate(self.numeric_cols_):
                emp_data = self.empirical_distributions_[col]
                synthetic_dict[col] = np.quantile(emp_data, synthetic_uniform[:, idx])

        for col in self.categorical_cols_:
            categories = list(self.categorical_probs_[col].keys())
            probs = list(self.categorical_probs_[col].values())
            synthetic_dict[col] = np.random.choice(categories, size=n_samples, p=probs)

        sampled_df = pd.DataFrame(synthetic_dict)
        logger.info(f"Synthesized {n_samples} high-fidelity tabular rows with shape {sampled_df.shape}.")
        return sampled_df

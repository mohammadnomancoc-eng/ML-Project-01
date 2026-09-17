"""Exploratory Data Analysis (EDA) and Statistical Profiling."""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from logger import logger


class EDAAnalyzer:
    """Computes dataset statistical summaries, distributions, and collinearity alerts."""

    @staticmethod
    def generate_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Generates comprehensive summary report for a dataframe."""
        logger.info("Generating EDA statistical profile...")

        numeric_df = df.select_dtypes(include=[np.number])
        categorical_df = df.select_dtypes(exclude=[np.number])

        summary = {
            "total_rows": int(len(df)),
            "total_columns": int(df.shape[1]),
            "missing_values": df.isnull().sum().to_dict(),
            "numeric_stats": numeric_df.describe().to_dict(),
            "skewness": numeric_df.skew().to_dict(),
            "kurtosis": numeric_df.kurtosis().to_dict(),
            "categorical_counts": {
                col: categorical_df[col].value_counts().to_dict()
                for col in categorical_df.columns
            },
        }

        logger.info(f"Summary computed: {summary['total_rows']} rows, {summary['total_columns']} cols")
        return summary

    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
        """Computes Pearson correlation matrix for numerical features."""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr(method="pearson")

    @classmethod
    def find_high_correlation_pairs(cls, df: pd.DataFrame, threshold: float = 0.85) -> List[Tuple[str, str, float]]:
        """Identifies collinear feature pairs exceeding threshold."""
        corr = cls.compute_correlation_matrix(df).abs()
        np.fill_diagonal(corr.values, 0)

        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                col1, col2 = corr.columns[i], corr.columns[j]
                val = corr.iloc[i, j]
                if val >= threshold:
                    pairs.append((col1, col2, round(float(val), 4)))

        logger.info(f"Found {len(pairs)} highly correlated feature pairs (threshold >= {threshold}).")
        return pairs

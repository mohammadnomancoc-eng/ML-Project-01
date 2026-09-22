"""Data and Feature Target Leakage Detector for ML-Project-01.

Detects target leakage, train-test contamination, identical rows across splits,
and features with suspicious predictive power.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from logger import logger


class DataLeakageDetector:
    """Automated auditing utility for data leakage and split contamination."""

    @staticmethod
    def detect_train_test_overlap(
        train_df: pd.DataFrame, test_df: pd.DataFrame, subset_cols: List[str] = None
    ) -> Dict[str, Any]:
        """Detects duplicate / overlapping rows between train and test splits."""
        cols = subset_cols if subset_cols else list(set(train_df.columns).intersection(test_df.columns))

        train_sub = train_df[cols].drop_duplicates()
        test_sub = test_df[cols].drop_duplicates()

        merged = pd.merge(train_sub, test_sub, how="inner", on=cols)
        overlap_count = len(merged)
        overlap_pct = (overlap_count / len(test_df)) * 100 if len(test_df) > 0 else 0.0

        is_contaminated = overlap_count > 0
        if is_contaminated:
            logger.warning(
                f"Data Contamination Alert: {overlap_count} identical rows ({overlap_pct:.2f}%) found across splits!"
            )
        else:
            logger.info("Zero overlap detected between train and test splits.")

        return {
            "has_leakage": is_contaminated,
            "duplicate_count": overlap_count,
            "overlap_percentage": round(overlap_pct, 4),
        }

    @staticmethod
    def detect_target_correlation_leakage(
        df: pd.DataFrame, target_column: str, threshold: float = 0.95
    ) -> Dict[str, Any]:
        """Flags features having near-perfect correlation with target."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' missing.")

        numeric_df = df.select_dtypes(include=[np.number])
        if target_column not in numeric_df.columns:
            return {"leaking_features": [], "max_correlation": 0.0}

        correlations = numeric_df.corr()[target_column].abs().drop(target_column)
        leaks = correlations[correlations >= threshold].to_dict()

        for feat, corr in leaks.items():
            logger.warning(f"Target Leak Suspicion: Feature '{feat}' has correlation {corr:.4f} >= {threshold}")

        return {
            "leaking_features": list(leaks.keys()),
            "correlations": {k: round(v, 4) for k, v in leaks.items()},
            "max_correlation": round(float(correlations.max()) if not correlations.empty else 0.0, 4),
        }

    @staticmethod
    def audit_dataset(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
        corr_threshold: float = 0.95,
    ) -> Dict[str, Any]:
        """Runs full suite of leakage audits across datasets."""
        overlap_audit = DataLeakageDetector.detect_train_test_overlap(train_df, test_df)
        corr_audit = DataLeakageDetector.detect_target_correlation_leakage(
            train_df, target_column, threshold=corr_threshold
        )

        overall_clean = (not overlap_audit["has_leakage"]) and (len(corr_audit["leaking_features"]) == 0)
        return {
            "is_clean": overall_clean,
            "overlap_audit": overlap_audit,
            "correlation_leak_audit": corr_audit,
        }

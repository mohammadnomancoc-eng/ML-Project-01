"""Automated Data Quality & Completeness Auditor for ML-Project-01.

Audits DataFrames for missingness patterns, infinite values, cardinality anomalies,
and zero-variance features before feeding into training pipelines.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from logger import logger


class DataQualityAuditor:
    """Computes comprehensive health scores and quality diagnostics for datasets."""

    @staticmethod
    def audit_quality(df: pd.DataFrame, max_null_threshold: float = 0.25) -> Dict[str, Any]:
        """Calculates completeness score, column anomalies, and overall dataset health."""
        n_rows, n_cols = df.shape
        total_cells = n_rows * n_cols
        null_count = int(df.isnull().sum().sum())
        completeness_pct = round((1.0 - (null_count / max(1, total_cells))) * 100, 2)

        column_reports = {}
        anomalies = []

        for col in df.columns:
            series = df[col]
            nulls = int(series.isnull().sum())
            null_ratio = nulls / max(1, n_rows)
            unique_count = int(series.nunique(dropna=True))

            col_info = {
                "dtype": str(series.dtype),
                "null_count": nulls,
                "null_ratio": round(null_ratio, 4),
                "unique_values": unique_count,
            }

            if pd.api.types.is_numeric_dtype(series):
                infs = int(np.isinf(series.dropna()).sum())
                col_info["infinite_count"] = infs
                if infs > 0:
                    anomalies.append(f"Column '{col}' contains {infs} infinite values.")

            if null_ratio > max_null_threshold:
                anomalies.append(f"Column '{col}' exceeds null threshold ({null_ratio:.1%} > {max_null_threshold:.1%}).")

            if unique_count <= 1:
                anomalies.append(f"Column '{col}' is constant (zero variance).")

            column_reports[col] = col_info

        is_passed = len(anomalies) == 0
        logger.info(f"Data Quality Audit: Completeness={completeness_pct}%, Anomalies={len(anomalies)}, Passed={is_passed}")

        return {
            "is_passed": is_passed,
            "completeness_score": completeness_pct,
            "total_rows": n_rows,
            "total_columns": n_cols,
            "total_nulls": null_count,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
            "columns": column_reports,
        }

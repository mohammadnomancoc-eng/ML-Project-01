"""Automated Data Quality & Schema Profiler."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from logger import get_logger

logger = get_logger("DataProfiler")


class DataProfiler:
    """Profiles dataset health, cardinality, memory footprint, and quality issues."""

    @staticmethod
    def profile(df: pd.DataFrame) -> Dict[str, Any]:
        """Generates comprehensive dataset quality and schema diagnostics."""
        logger.info("Profiling dataset health and memory consumption...")

        memory_usage_mb = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
        missing_counts = df.isnull().sum()
        total_rows = len(df)

        column_reports = {}
        for col in df.columns:
            series = df[col]
            missing = int(missing_counts[col])
            col_type = str(series.dtype)
            unique_count = int(series.nunique())

            stats = {
                "dtype": col_type,
                "missing_count": missing,
                "missing_pct": round((missing / total_rows) * 100, 2) if total_rows > 0 else 0,
                "unique_count": unique_count,
                "is_constant": unique_count <= 1,
            }

            if np.issubdtype(series.dtype, np.number):
                stats["min"] = float(series.min()) if not series.empty else None
                stats["max"] = float(series.max()) if not series.empty else None
                stats["mean"] = round(float(series.mean()), 3) if not series.empty else None

            column_reports[col] = stats

        report = {
            "row_count": total_rows,
            "column_count": len(df.columns),
            "memory_mb": memory_usage_mb,
            "total_missing_cells": int(missing_counts.sum()),
            "columns": column_reports,
        }

        logger.info(f"Profiling complete: {total_rows} rows, {len(df.columns)} cols, {memory_usage_mb} MB")
        return report

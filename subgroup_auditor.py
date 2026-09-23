"""Subpopulation Slice Discovery and Subgroup Performance Auditor for ML-Project-01.

Discovers cohort slices with elevated error rates, algorithmic bias, or disparate impact.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from logger import logger


class SubgroupAuditor:
    """Audits model performance across demographic, geographical, and categorical subgroups."""

    @staticmethod
    def audit_categorical_subgroups(
        df: pd.DataFrame,
        subgroup_column: str,
        y_true: pd.Series,
        y_pred: np.ndarray,
        min_slice_size: int = 10,
    ) -> List[Dict[str, Any]]:
        """Computes performance metrics across distinct slices of a categorical feature."""
        if subgroup_column not in df.columns:
            raise ValueError(f"Column '{subgroup_column}' not found in DataFrame.")

        temp_df = pd.DataFrame({
            "slice": df[subgroup_column],
            "y_true": y_true.values if hasattr(y_true, "values") else y_true,
            "y_pred": y_pred,
        })

        overall_mae = float(mean_absolute_error(temp_df["y_true"], temp_df["y_pred"]))
        results = []

        for slice_val, group in temp_df.groupby("slice"):
            if len(group) < min_slice_size:
                continue

            slice_mae = float(mean_absolute_error(group["y_true"], group["y_pred"]))
            slice_rmse = float(np.sqrt(mean_squared_error(group["y_true"], group["y_pred"])))
            disparity_ratio = slice_mae / (overall_mae + 1e-8)

            results.append({
                "subgroup": str(slice_val),
                "sample_count": len(group),
                "mae": round(slice_mae, 4),
                "rmse": round(slice_rmse, 4),
                "disparity_ratio": round(disparity_ratio, 4),
                "status": "UNDERPERFORMING" if disparity_ratio > 1.25 else "NORMAL",
            })

        # Sort slices by highest error
        results.sort(key=lambda x: x["mae"], reverse=True)
        logger.info(f"Subgroup audit completed for '{subgroup_column}' across {len(results)} slices.")
        return results

    @staticmethod
    def find_underperforming_slices(
        df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, threshold_multiplier: float = 1.3
    ) -> List[Dict[str, Any]]:
        """Scans all categorical columns and flags underperforming subpopulation slices."""
        flagged_slices = []
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        for col in cat_cols:
            audit = SubgroupAuditor.audit_categorical_subgroups(df, col, y_true, y_pred)
            for entry in audit:
                if entry["disparity_ratio"] >= threshold_multiplier:
                    entry["feature"] = col
                    flagged_slices.append(entry)

        return flagged_slices

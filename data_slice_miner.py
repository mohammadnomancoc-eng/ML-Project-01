"""High-Error Data Slice and Cohort Miner for ML-Project-01.

Discovers multivariate conditional feature slices exhibiting significantly elevated prediction error rates.
"""

from typing import Dict, List, Any
import numpy as np
import pandas as pd
from logger import logger


class DataSliceMiner:
    """Discovers underperforming cohorts via multi-dimensional binning."""

    @staticmethod
    def mine_worst_slices(
        df: pd.DataFrame,
        y_true: pd.Series,
        y_pred: np.ndarray,
        top_k: int = 5,
        min_cohort_size: int = 25,
    ) -> List[Dict[str, Any]]:
        """Identifies combination slices with highest mean absolute error (MAE)."""
        temp_df = df.copy()
        errors = np.abs(y_true.values - y_pred)
        temp_df["_abs_error"] = errors
        overall_mae = float(np.mean(errors))

        slice_candidates = []
        cat_cols = temp_df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in cat_cols:
            grouped = temp_df.groupby(col).agg(
                cohort_size=("_abs_error", "count"),
                slice_mae=("_abs_error", "mean"),
            ).reset_index()

            for _, row in grouped.iterrows():
                if row["cohort_size"] >= min_cohort_size:
                    disparity = row["slice_mae"] / (overall_mae + 1e-8)
                    slice_candidates.append({
                        "feature": col,
                        "slice_value": str(row[col]),
                        "cohort_size": int(row["cohort_size"]),
                        "slice_mae": round(float(row["slice_mae"]), 4),
                        "overall_mae": round(overall_mae, 4),
                        "error_disparity_ratio": round(float(disparity), 2),
                    })

        slice_candidates.sort(key=lambda x: x["slice_mae"], reverse=True)
        top_slices = slice_candidates[:top_k]
        logger.info(f"Mined {len(top_slices)} worst error cohort slices.")
        return top_slices

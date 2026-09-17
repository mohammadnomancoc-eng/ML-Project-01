"""Algorithmic Fairness and Demographic Parity Evaluator."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from logger import get_logger

logger = get_logger("FairnessEvaluator")


class FairnessEvaluator:
    """Evaluates disparate impact, group error disparities, and fairness metrics."""

    @staticmethod
    def evaluate_subgroup_performance(
        df: pd.DataFrame, group_col: str, y_true_col: str, y_pred_col: str
    ) -> Dict[str, Any]:
        """Calculates performance disparities across distinct demographic groups."""
        logger.info(f"Evaluating fairness metrics across subgroups in column: '{group_col}'...")

        subgroup_metrics = {}
        unique_groups = df[group_col].unique()

        for group in unique_groups:
            sub = df[df[group_col] == group]
            y_true = sub[y_true_col].values
            y_pred = sub[y_pred_col].values

            mae = float(np.mean(np.abs(y_true - y_pred)))
            rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
            mean_prediction = float(np.mean(y_pred))

            subgroup_metrics[str(group)] = {
                "sample_count": len(sub),
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "mean_prediction": round(mean_prediction, 4),
            }

        # Calculate max-to-min disparate impact ratio
        maes = [v["mae"] for v in subgroup_metrics.values()]
        disparity_ratio = round(max(maes) / (min(maes) + 1e-9), 3) if maes else 1.0

        report = {
            "group_column": group_col,
            "groups_evaluated": len(unique_groups),
            "max_disparity_ratio": disparity_ratio,
            "is_fair": disparity_ratio < 1.3,
            "subgroups": subgroup_metrics,
        }

        logger.info(f"Fairness evaluation complete: Disparity Ratio = {disparity_ratio} (Fair: {report['is_fair']})")
        return report

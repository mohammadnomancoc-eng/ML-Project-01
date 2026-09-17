"""Calibration Curves and Reliability Diagnostics for Regression."""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from config import paths
from logger import get_logger

logger = get_logger("CalibrationPlotter")


class CalibrationPlotter:
    """Plots binned decile calibration curves comparing predicted expectations vs true values."""

    @staticmethod
    def plot_regression_calibration(
        y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10, filename: str = "calibration_curve.png"
    ) -> Path:
        """Plots observed vs predicted bucketed deciles."""
        df = pd.DataFrame({"true": y_true, "pred": y_pred})
        df["bin"] = pd.qcut(df["pred"], q=n_bins, duplicates="drop")

        grouped = df.groupby("bin", observed=True).agg({"true": "mean", "pred": "mean"}).reset_index()

        plt.figure(figsize=(7, 6))
        plt.plot(grouped["pred"], grouped["true"], "o-", color="#3E71C0", lw=2, label="Model Calibration")
        min_v = min(df["pred"].min(), df["true"].min())
        max_v = max(df["pred"].max(), df["true"].max())
        plt.plot([min_v, max_v], [min_v, max_v], "r--", label="Perfect Calibration (y=x)")

        plt.xlabel("Mean Predicted Value", fontsize=11)
        plt.ylabel("Mean Observed True Value", fontsize=11)
        plt.title("Regression Reliability & Calibration Curve", fontsize=13)
        plt.legend()
        plt.tight_layout()

        out_path = paths.OUTPUT_DIR / filename
        plt.savefig(out_path, dpi=300)
        plt.close()

        logger.info(f"Calibration plot saved to: {out_path}")
        return out_path

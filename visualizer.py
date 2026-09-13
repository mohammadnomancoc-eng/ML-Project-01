"""Data & Model Performance Visualization Utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from config import paths
from logger import logger


class MLVisualizer:
    """Generates charts and diagnostics plots saved to output directory."""

    def __init__(self, style: str = "darkgrid"):
        sns.set_theme(style=style)

    def plot_correlation_heatmap(self, corr_df: pd.DataFrame, filename: str = "correlation_heatmap.png") -> Path:
        """Renders and saves correlation matrix heatmap."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_df, annot=True, cmap="Blues", fmt=".2f", square=True)
        plt.title("Feature Correlation Matrix", fontsize=14, pad=15)
        plt.tight_layout()

        output_path = paths.OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved correlation heatmap to: {output_path}")
        return output_path

    def plot_predictions_vs_actual(
        self, y_true: np.ndarray, y_pred: np.ndarray, filename: str = "pred_vs_actual.png"
    ) -> Path:
        """Plots scatter plot of predicted vs ground-truth values."""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color="#3E71C0", edgecolors="w", s=40)
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal (y=x)")
        plt.xlabel("Actual Ground Truth", fontsize=12)
        plt.ylabel("Model Predictions", fontsize=12)
        plt.title("Actual vs Predicted Scatter", fontsize=14)
        plt.legend()
        plt.tight_layout()

        output_path = paths.OUTPUT_DIR / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved prediction scatter plot to: {output_path}")
        return output_path

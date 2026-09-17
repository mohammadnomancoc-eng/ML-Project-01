"""Publication-Ready Feature Importance Plotter."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from config import paths
from logger import get_logger

logger = get_logger("ImportancePlotter")


class FeatureImportancePlotter:
    """Renders customized horizontal feature importance bar charts."""

    @staticmethod
    def plot_importances(
        feature_names: list,
        importances: np.ndarray,
        filename: str = "feature_importance_ranked.png",
        top_n: int = 10,
    ) -> Path:
        """Plots top-N feature importances with percentage annotations."""
        df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        df = df.sort_values(by="Importance", ascending=True).tail(top_n)

        plt.figure(figsize=(9, 6))
        sns.set_theme(style="whitegrid")

        bars = plt.barh(df["Feature"], df["Importance"], color="#3E71C0", edgecolor="#202D43", height=0.6)
        plt.xlabel("Importance Score", fontsize=11, fontweight="bold")
        plt.title(f"Top {len(df)} Feature Importances", fontsize=13, pad=12)

        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.005, bar.get_y() + bar.get_height() / 2, f"{width:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        out_path = paths.OUTPUT_DIR / filename
        plt.savefig(out_path, dpi=300)
        plt.close()

        logger.info(f"Feature importance chart saved to: {out_path}")
        return out_path

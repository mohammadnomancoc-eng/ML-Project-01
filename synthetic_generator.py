"""Data Augmentation and Synthetic Sample Generator."""

import numpy as np
import pandas as pd
from typing import Optional
from logger import get_logger

logger = get_logger("SyntheticGenerator")


class SyntheticAugmenter:
    """Augments training datasets via linear interpolation and Gaussian jitter."""

    def __init__(self, noise_scale: float = 0.05, random_state: int = 42):
        self.noise_scale = noise_scale
        self.rng = np.random.RandomState(random_state)

    def add_gaussian_jitter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Injects small Gaussian noise to continuous numerical columns."""
        augmented = df.copy()
        numeric_cols = augmented.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_std = augmented[col].std()
            if col_std > 0:
                noise = self.rng.normal(0, self.noise_scale * col_std, size=len(augmented))
                augmented[col] += noise

        logger.info(f"Injected Gaussian jitter to {len(numeric_cols)} columns across {len(df)} rows.")
        return augmented

    def interpolate_samples(self, df: pd.DataFrame, n_samples: int = 200) -> pd.DataFrame:
        """Creates synthetic rows via random convex pairs interpolation."""
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df) < 2:
            return df

        idx1 = self.rng.choice(len(numeric_df), size=n_samples)
        idx2 = self.rng.choice(len(numeric_df), size=n_samples)
        alphas = self.rng.uniform(0.1, 0.9, size=(n_samples, 1))

        mat1 = numeric_df.iloc[idx1].values
        mat2 = numeric_df.iloc[idx2].values
        synth_mat = mat1 * alphas + mat2 * (1 - alphas)

        synth_df = pd.DataFrame(synth_mat, columns=numeric_df.columns)
        logger.info(f"Generated {n_samples} interpolated synthetic samples.")
        return pd.concat([df, synth_df], ignore_index=True)

    def inject_extreme_outliers(
        self, df: pd.DataFrame, target_columns: Optional[list] = None, ratio: float = 0.02, multiplier: float = 5.0
    ) -> pd.DataFrame:
        """Injects extreme statistical anomalies into dataset to stress test robust estimators."""
        perturbed = df.copy()
        numeric_cols = target_columns or list(perturbed.select_dtypes(include=[np.number]).columns)
        n_outliers = max(1, int(len(perturbed) * ratio))

        for col in numeric_cols:
            if col in perturbed.columns:
                outlier_indices = self.rng.choice(len(perturbed), size=n_outliers, replace=False)
                std_val = perturbed[col].std() or 1.0
                signs = self.rng.choice([-1, 1], size=n_outliers)
                perturbed.loc[outlier_indices, col] += signs * multiplier * std_val

        logger.info(f"Injected {n_outliers} synthetic outlier values across {len(numeric_cols)} features.")
        return perturbed


"""Data Sampling Engine for ML-Project-01.

Provides reservoir sampling, stratified quantile sampling, and class-balanced subsampling.
"""

from typing import Optional, Union, List
import numpy as np
import pandas as pd
from logger import logger


class DataSampler:
    """Utility class providing versatile data sampling strategies."""

    @staticmethod
    def reservoir_sample(
        data: Union[pd.DataFrame, np.ndarray, List],
        sample_size: int,
        random_state: Optional[int] = 42,
    ) -> Union[pd.DataFrame, np.ndarray, List]:
        """Performs reservoir sampling to draw random sample from stream/dataset."""
        if random_state is not None:
            np.random.seed(random_state)

        n = len(data)
        if sample_size >= n:
            return data

        reservoir_indices = list(range(sample_size))
        for i in range(sample_size, n):
            j = np.random.randint(0, i + 1)
            if j < sample_size:
                reservoir_indices[j] = i

        if isinstance(data, pd.DataFrame):
            return data.iloc[reservoir_indices].reset_index(drop=True)
        elif isinstance(data, np.ndarray):
            return data[reservoir_indices]
        else:
            return [data[i] for i in reservoir_indices]

    @staticmethod
    def stratified_quantile_sample(
        df: pd.DataFrame,
        target_column: str,
        n_bins: int = 5,
        fraction: float = 0.5,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Samples continuous target uniformly across distribution quantiles."""
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        temp_df = df.copy()
        quantiles = pd.qcut(temp_df[target_column], q=n_bins, labels=False, duplicates="drop")
        temp_df["_strata_bin"] = quantiles

        sampled = temp_df.groupby("_strata_bin", group_keys=False).apply(
            lambda x: x.sample(frac=fraction, random_state=random_state)
        )
        sampled = sampled.drop(columns=["_strata_bin"]).reset_index(drop=True)
        logger.info(f"Stratified sample created with {len(sampled)} rows across {n_bins} quantiles.")
        return sampled

    @staticmethod
    def balanced_categorical_sample(
        df: pd.DataFrame,
        category_column: str,
        samples_per_class: Optional[int] = None,
        random_state: Optional[int] = 42,
    ) -> pd.DataFrame:
        """Draws balanced number of rows for each categorical group."""
        if category_column not in df.columns:
            raise ValueError(f"Category column '{category_column}' not found in DataFrame.")

        class_counts = df[category_column].value_counts()
        target_n = samples_per_class if samples_per_class is not None else class_counts.min()

        balanced_groups = []
        for cat_val, group in df.groupby(category_column):
            replace = len(group) < target_n
            sampled_group = group.sample(n=target_n, replace=replace, random_state=random_state)
            balanced_groups.append(sampled_group)

        result_df = pd.concat(balanced_groups, axis=0).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        logger.info(f"Balanced categorical dataset created with shape {result_df.shape}.")
        return result_df

    @staticmethod
    def systematic_sample(df: pd.DataFrame, step_size: int = 5, start_index: int = 0) -> pd.DataFrame:
        """Samples rows at fixed periodic intervals."""
        indices = np.arange(start_index, len(df), step_size)
        sampled = df.iloc[indices].reset_index(drop=True)
        logger.info(f"Systematic sampling extracted {len(sampled)} rows (step={step_size}).")
        return sampled

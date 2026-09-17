"""Pandas DataFrame Memory Footprint Optimizer."""

import numpy as np
import pandas as pd
from typing import Tuple
from logger import get_logger

logger = get_logger("MemoryOptimizer")


class DataFrameMemoryOptimizer:
    """Downcasts numeric types and converts low-cardinality strings to categories."""

    @staticmethod
    def optimize(df: pd.DataFrame, verbose: bool = True) -> Tuple[pd.DataFrame, float]:
        """Reduces memory usage by choosing minimal valid numeric types."""
        start_mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
        optimized = df.copy()

        for col in optimized.columns:
            col_type = optimized[col].dtype

            if np.issubdtype(col_type, np.integer):
                c_min = optimized[col].min()
                c_max = optimized[col].max()
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized[col] = optimized[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized[col] = optimized[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized[col] = optimized[col].astype(np.int32)
            elif np.issubdtype(col_type, np.floating):
                optimized[col] = optimized[col].astype(np.float32)
            elif col_type == object and optimized[col].nunique() / len(optimized) < 0.3:
                optimized[col] = optimized[col].astype("category")

        end_mem = optimized.memory_usage(deep=True).sum() / (1024 * 1024)
        reduction_pct = round((1 - end_mem / start_mem) * 100, 2) if start_mem > 0 else 0

        if verbose:
            logger.info(f"Memory reduction: {start_mem:.2f} MB -> {end_mem:.2f} MB ({reduction_pct}% reduction)")

        return optimized, reduction_pct

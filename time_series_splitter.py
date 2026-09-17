"""Temporal and Time-Series Cross-Validation Splitters."""

import numpy as np
import pandas as pd
from typing import Generator, Tuple
from logger import get_logger

logger = get_logger("TimeSeriesSplitter")


class TemporalSplitter:
    """Generates expanding and sliding temporal train/validation indices."""

    def __init__(self, n_splits: int = 4, max_train_size: int = None, test_size: int = None):
        self.n_splits = n_splits
        self.max_train_size = max_train_size
        self.test_size = test_size

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Yields train and test index arrays for temporal evaluation."""
        n_samples = len(df)
        test_size = self.test_size or (n_samples // (self.n_splits + 1))

        logger.info(f"Generating {self.n_splits} temporal splits (test window size: {test_size})...")

        for i in range(self.n_splits):
            test_end = n_samples - (self.n_splits - 1 - i) * test_size
            test_start = test_end - test_size
            train_end = test_start

            train_start = 0 if not self.max_train_size else max(0, train_end - self.max_train_size)
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            yield train_idx, test_idx

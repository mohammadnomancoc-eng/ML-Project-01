"""Data Ingestion and Synthetic Dataset Generation Module."""

import numpy as np
import pandas as pd
from typing import Tuple, Generator, Optional
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from config import config, paths
from logger import logger


class DataLoader:
    """Handles dataset generation, disk loading, and data splitting."""

    @staticmethod
    def generate_synthetic_dataset(
        n_samples: int = config.N_SAMPLES,
        n_features: int = config.N_FEATURES,
        noise: float = 12.5,
        save_to_disk: bool = True,
    ) -> pd.DataFrame:
        """Generates realistic synthetic regression dataset."""
        logger.info(f"Generating synthetic dataset with {n_samples} samples and {n_features} features...")
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=int(n_features * 0.75),
            noise=noise,
            random_state=config.RANDOM_STATE,
        )

        feature_names = [f"feature_{i+1}" for i in range(n_features)]
        df = pd.DataFrame(X, columns=feature_names)
        df[config.TARGET_COLUMN] = y

        # Inject realistic categorical column
        regions = ["North", "South", "East", "West"]
        df["region"] = np.random.choice(regions, size=n_samples)

        if save_to_disk:
            raw_path = paths.DATA_DIR / "raw_data.csv"
            df.to_csv(raw_path, index=False)
            logger.info(f"Dataset successfully saved to {raw_path}")

        return df

    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Loads data from a CSV file."""
        logger.info(f"Loading dataset from {file_path}")
        return pd.read_csv(file_path)

    @staticmethod
    def stream_csv_chunks(file_path: str, chunksize: int = 500) -> Generator[pd.DataFrame, None, None]:
        """Streams large CSV files in manageable chunks."""
        logger.info(f"Streaming CSV from {file_path} in chunks of {chunksize}...")
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            yield chunk

    @staticmethod
    def split_dataset(
        df: pd.DataFrame,
        target_col: str = config.TARGET_COLUMN,
        test_size: float = config.TEST_SIZE,
        val_size: float = config.VAL_SIZE,
        stratify_quantiles: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Performs 3-way train/validation/test split with optional quantile stratification."""
        X = df.drop(columns=[target_col])
        y = df[target_col]

        strat_bins = None
        if stratify_quantiles is not None:
            strat_bins = pd.qcut(y, q=stratify_quantiles, labels=False, duplicates="drop")

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=strat_bins
        )

        adjusted_val_size = val_size / (1.0 - test_size)
        val_strat = None
        if stratify_quantiles is not None:
            val_strat = pd.qcut(y_train_val, q=stratify_quantiles, labels=False, duplicates="drop")

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=adjusted_val_size, random_state=config.RANDOM_STATE, stratify=val_strat
        )

        logger.info(f"Data split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

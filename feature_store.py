"""Lightweight Feature Store for Offline & Online Feature Retrieval."""

import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from config import paths
from logger import get_logger

logger = get_logger("FeatureStore")


class FeatureStore:
    """Manages offline tabular feature views and low-latency entity lookups."""

    def __init__(self, store_dir: str = "data/features"):
        self.store_dir = paths.DATA_DIR / "features"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.memory_store: Dict[str, pd.DataFrame] = {}

    def register_feature_view(self, view_name: str, df: pd.DataFrame, entity_key: str) -> None:
        """Saves feature view to disk and in-memory cache."""
        logger.info(f"Registering feature view '{view_name}' with {len(df)} entities on key '{entity_key}'...")
        file_path = self.store_dir / f"{view_name}.parquet"
        df.to_parquet(file_path, index=False)
        self.memory_store[view_name] = df.set_index(entity_key)
        logger.info(f"Feature view '{view_name}' successfully persisted to {file_path}")

    def get_online_features(self, view_name: str, entity_ids: List[Any]) -> pd.DataFrame:
        """Retrieves low-latency feature vectors for specific entity IDs."""
        if view_name not in self.memory_store:
            file_path = self.store_dir / f"{view_name}.parquet"
            if not file_path.exists():
                raise KeyError(f"Feature view '{view_name}' does not exist.")
            self.memory_store[view_name] = pd.read_parquet(file_path)

        cached_view = self.memory_store[view_name]
        return cached_view.loc[cached_view.index.isin(entity_ids)]

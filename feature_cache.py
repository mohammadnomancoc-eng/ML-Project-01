"""Offline-to-Online Feature Cache and Key-Value Store for ML-Project-01.

Provides fast in-memory, disk-backed, and TTL (Time-to-Live) key-value caching for feature vectors
to minimize redundant feature transformations during real-time online inference.
"""

import time
from typing import Dict, List, Optional, Any
import pandas as pd
from logger import logger


class FeatureStoreCache:
    """Key-value cache for precomputed entity feature vectors with expiration support."""

    def __init__(self, default_ttl_seconds: int = 3600):
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}

    def put(self, entity_id: str, features: Dict[str, Any], ttl_seconds: Optional[int] = None) -> None:
        """Stores feature dictionary for a given entity key with expiration timestamp."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        expire_at = time.time() + ttl
        self._cache[str(entity_id)] = {
            "features": features,
            "expires_at": expire_at,
        }

    def get(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached features if not expired."""
        key = str(entity_id)
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if time.time() > entry["expires_at"]:
            del self._cache[key]  # Expire entry
            return None

        return entry["features"]

    def put_dataframe(self, df: pd.DataFrame, key_column: str, ttl_seconds: Optional[int] = None) -> int:
        """Batch inserts entire DataFrame indexed by key_column into the feature cache."""
        if key_column not in df.columns:
            raise ValueError(f"Key column '{key_column}' not found in DataFrame.")

        records = df.to_dict(orient="records")
        for rec in records:
            key_val = str(rec[key_column])
            feat_dict = {k: v for k, v in rec.items() if k != key_column}
            self.put(key_val, feat_dict, ttl_seconds=ttl_seconds)

        logger.info(f"Cached {len(records)} entity feature vectors in FeatureStoreCache.")
        return len(records)

    def size(self) -> int:
        """Returns current count of non-expired cached entities."""
        now = time.time()
        active_keys = [k for k, v in self._cache.items() if v["expires_at"] > now]
        return len(active_keys)

"""Dataset Schema Evolution and Migration Utility."""

import pandas as pd
from typing import Dict, Any, List
from logger import get_logger

logger = get_logger("SchemaMigrator")


class SchemaMigrator:
    """Manages schema version migrations, column renames, and type casting for datasets."""

    def __init__(self, target_schema: Dict[str, str]):
        self.target_schema = target_schema

    def migrate(self, df: pd.DataFrame, rename_map: Dict[str, str] = None, defaults: Dict[str, Any] = None) -> pd.DataFrame:
        """Applies column renames, fills missing expected columns with defaults, and enforces dtypes."""
        migrated = df.copy()

        # Apply renames
        if rename_map:
            migrated = migrated.rename(columns=rename_map)

        # Inject missing target columns
        if defaults:
            for col, val in defaults.items():
                if col not in migrated.columns:
                    migrated[col] = val

        # Enforce target dtypes
        for col, expected_dtype in self.target_schema.items():
            if col in migrated.columns:
                try:
                    migrated[col] = migrated[col].astype(expected_dtype)
                except Exception as e:
                    logger.warning(f"Failed to cast column '{col}' to {expected_dtype}: {e}")

        logger.info(f"Successfully migrated dataframe to target schema with {len(migrated.columns)} columns.")
        return migrated

"""Automated Text & Raw Data Cleaning Utilities."""

import re
import pandas as pd
from typing import List, Optional
from logger import get_logger

logger = get_logger("DataCleaner")


class DataCleaner:
    """Sanitizes raw text strings, strips non-ASCII noise, and cleans dirty dataframes."""

    @staticmethod
    def clean_text_column(series: pd.Series) -> pd.Series:
        """Trims whitespace, converts to lowercase, and strips non-alphanumeric noise."""
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .apply(lambda x: re.sub(r"[^\w\s-]", "", x))
            .apply(lambda x: re.sub(r"\s+", " ", x))
        )

    @classmethod
    def sanitize_dataframe(cls, df: pd.DataFrame, text_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Cleans headers and selected text columns across dataframe."""
        cleaned = df.copy()

        # Clean column names (lowercase and snake_case)
        cleaned.columns = [
            re.sub(r"\W+", "_", col.strip().lower()).strip("_")
            for col in cleaned.columns
        ]

        if text_columns:
            for col in text_columns:
                if col in cleaned.columns:
                    cleaned[col] = cls.clean_text_column(cleaned[col])

        logger.info(f"Sanitized dataframe with {len(cleaned.columns)} columns.")
        return cleaned

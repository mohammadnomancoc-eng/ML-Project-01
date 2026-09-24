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

    @staticmethod
    def drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Removes columns with zero variance or all identical values."""
        constant_cols = [col for col in df.columns if df[col].nunique(dropna=False) <= 1]
        if constant_cols:
            logger.info(f"Dropping {len(constant_cols)} constant columns: {constant_cols}")
            return df.drop(columns=constant_cols)
        return df

    @staticmethod
    def impute_missing_categories(df: pd.DataFrame, fill_value: str = "Unknown") -> pd.DataFrame:
        """Fills null values in string/categorical columns with a default token."""
        df_imputed = df.copy()
        cat_cols = df_imputed.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols:
            df_imputed[col] = df_imputed[col].fillna(fill_value)
        return df_imputed

    @staticmethod
    def drop_low_variance_columns(df: pd.DataFrame, variance_threshold: float = 1e-5) -> pd.DataFrame:
        """Removes numeric columns with variance below specified threshold."""
        num_cols = df.select_dtypes(include=["number"]).columns
        low_var_cols = [col for col in num_cols if df[col].var() < variance_threshold]
        if low_var_cols:
            logger.info(f"Dropping {len(low_var_cols)} low-variance columns: {low_var_cols}")
            return df.drop(columns=low_var_cols)
        return df


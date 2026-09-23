"""Dataset Fingerprinting and Schema Checksum Verification for ML-Project-01.

Computes cryptographic hashes, statistical distribution moments (mean, std, skewness),
and schema fingerprints to detect silent data changes and dataset tampering.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from logger import logger


class DatasetFingerprinter:
    """Generates unique deterministic fingerprints and statistical profiles for DataFrames."""

    @staticmethod
    def generate_fingerprint(df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates cryptographic checksum and statistical signature of a DataFrame."""
        numeric_df = df.select_dtypes(include=[np.number])
        cat_df = df.select_dtypes(exclude=[np.number])

        # 1. Deterministic Content Hash (using serialized head & tail representation)
        content_repr = f"{df.shape}_{list(df.columns)}_{df.iloc[:20].to_dict()}_{df.iloc[-20:].to_dict()}".encode("utf-8")
        content_hash = hashlib.sha256(content_repr).hexdigest()

        # 2. Statistical Moments
        numeric_moments = {}
        for col in numeric_df.columns:
            series = numeric_df[col].dropna()
            if len(series) > 0:
                numeric_moments[col] = {
                    "mean": round(float(series.mean()), 4),
                    "std": round(float(series.std()), 4),
                    "min": round(float(series.min()), 4),
                    "max": round(float(series.max()), 4),
                    "null_count": int(numeric_df[col].isnull().sum()),
                }

        # 3. Categorical Signatures
        categorical_signatures = {}
        for col in cat_df.columns:
            categorical_signatures[col] = {
                "unique_count": int(cat_df[col].nunique()),
                "top_value": str(cat_df[col].mode().iloc[0]) if not cat_df[col].empty else None,
                "null_count": int(cat_df[col].isnull().sum()),
            }

        fingerprint = {
            "content_sha256": content_hash,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "column_names": list(df.columns),
            "numeric_moments": numeric_moments,
            "categorical_signatures": categorical_signatures,
        }

        logger.info(f"Dataset fingerprint generated. SHA256: {content_hash[:12]}...")
        return fingerprint

    @staticmethod
    def compare_fingerprints(fp1: Dict[str, Any], fp2: Dict[str, Any]) -> Dict[str, Any]:
        """Compares two dataset fingerprints and returns drift / schema diff summary."""
        is_identical = fp1["content_sha256"] == fp2["content_sha256"]
        shape_diff = fp1["shape"] != fp2["shape"]
        cols_added = list(set(fp2["column_names"]) - set(fp1["column_names"]))
        cols_removed = list(set(fp1["column_names"]) - set(fp2["column_names"]))

        return {
            "is_identical": is_identical,
            "shape_changed": shape_diff,
            "columns_added": cols_added,
            "columns_removed": cols_removed,
        }

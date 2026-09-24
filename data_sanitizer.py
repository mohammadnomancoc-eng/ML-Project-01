"""Data Sanitizer and PII Masking Engine for ML-Project-01.

Provides automated detection and masking of personally identifiable information (PII)
such as emails, credit cards, IP addresses, SSNs, and phone numbers in tabular data.
"""

import re
from typing import Dict, List, Optional, Union
import pandas as pd
from logger import logger


class DataSanitizer:
    """Detects and redacts sensitive personally identifiable information (PII) from DataFrames."""

    PATTERNS = {
        "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "ip_address": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "phone_number": r"\b(?:\+?\d{1,3}[-\s]?)?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}\b",
    }

    def __init__(self, mask_token: str = "[REDACTED]"):
        self.mask_token = mask_token
        self.compiled_regex = {k: re.compile(v) for k, v in self.PATTERNS.items()}

    def sanitize_text(self, text: str) -> str:
        """Sanitizes a single text string by replacing recognized PII with mask tokens."""
        if not isinstance(text, str):
            return text
        sanitized = text
        for pii_type, regex in self.compiled_regex.items():
            sanitized = regex.sub(f"[{pii_type.upper()}_{self.mask_token}]", sanitized)
        return sanitized

    def sanitize_dataframe(
        self, df: pd.DataFrame, text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Sanitizes all or selected text columns in a DataFrame."""
        df_clean = df.copy()
        if text_columns is None:
            text_columns = df_clean.select_dtypes(include=["object", "string"]).columns.tolist()

        redaction_count = 0
        for col in text_columns:
            if col in df_clean.columns:
                original = df_clean[col].astype(str)
                df_clean[col] = df_clean[col].apply(lambda val: self.sanitize_text(str(val)) if pd.notnull(val) else val)
                redaction_count += (original != df_clean[col]).sum()

        logger.info(f"Sanitized DataFrame: {redaction_count} PII occurrences redacted across {len(text_columns)} columns.")
        return df_clean

    def audit_pii_density(self, df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        """Audits PII risk and returns detection counts per column and pattern."""
        report = {}
        text_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in text_cols:
            col_report = {}
            for pii_type, regex in self.compiled_regex.items():
                matches = df[col].astype(str).apply(lambda x: len(regex.findall(x))).sum()
                if matches > 0:
                    col_report[pii_type] = int(matches)
            if col_report:
                report[col] = col_report

        return report

    def register_custom_pattern(self, name: str, regex_pattern: str) -> None:
        """Registers a custom PII pattern or sensitive token for masking."""
        self.PATTERNS[name] = regex_pattern
        self.compiled_regex[name] = re.compile(regex_pattern)
        logger.info(f"Registered custom PII pattern: '{name}'")

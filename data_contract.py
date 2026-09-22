"""Data Contract Enforcement and Schema Specification for ML-Project-01.

Validates DataFrame schemas, allowed value sets, nullability bounds, and numeric constraints against a defined Data Contract.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
from logger import logger


@dataclass
class ColumnContract:
    """Defines strict validation rules for an individual column."""

    name: str
    dtype: str  # 'numeric', 'string', 'category'
    nullable: bool = True
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None


class DataContract:
    """Enforces schema consistency, constraints, and data contracts on input datasets."""

    def __init__(self, contracts: List[ColumnContract]):
        self.contracts = {c.name: c for c in contracts}

    def validate(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validates a DataFrame against the data contract definitions."""
        violations = []

        for col_name, rule in self.contracts.items():
            if col_name not in df.columns:
                violations.append(f"Missing required column: '{col_name}'")
                continue

            col = df[col_name]

            # Nullable check
            if not rule.nullable and col.isnull().any():
                violations.append(f"Column '{col_name}' contains nulls but is marked non-nullable.")

            # Dtype validation
            if rule.dtype == "numeric":
                if not pd.api.types.is_numeric_dtype(col):
                    violations.append(f"Column '{col_name}' is expected to be numeric.")
                else:
                    if rule.min_value is not None and (col.min() < rule.min_value):
                        violations.append(
                            f"Column '{col_name}' min value ({col.min()}) violates contract min ({rule.min_value})"
                        )
                    if rule.max_value is not None and (col.max() > rule.max_value):
                        violations.append(
                            f"Column '{col_name}' max value ({col.max()}) violates contract max ({rule.max_value})"
                        )

            # Allowed categories check
            if rule.allowed_values is not None:
                invalid_cats = set(col.dropna().unique()) - set(rule.allowed_values)
                if invalid_cats:
                    violations.append(
                        f"Column '{col_name}' contains disallowed values: {invalid_cats}"
                    )

        is_valid = len(violations) == 0
        if is_valid:
            logger.info("Data Contract Validation: PASS (0 violations).")
        else:
            logger.warning(f"Data Contract Validation: FAIL ({len(violations)} violations).")

        return {
            "is_valid": is_valid,
            "violations_count": len(violations),
            "violations": violations,
        }

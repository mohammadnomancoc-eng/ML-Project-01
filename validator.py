"""Input Schema Validation Module for ML Pipeline & REST API."""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class SinglePredictionRequest(BaseModel):
    """Schema for single sample prediction request."""

    feature_1: float = Field(..., description="Numerical feature 1")
    feature_2: float = Field(..., description="Numerical feature 2")
    feature_3: float = Field(..., description="Numerical feature 3")
    feature_4: float = Field(..., description="Numerical feature 4")
    feature_5: float = Field(..., description="Numerical feature 5")
    feature_6: float = Field(..., description="Numerical feature 6")
    feature_7: float = Field(..., description="Numerical feature 7")
    feature_8: float = Field(..., description="Numerical feature 8")
    region: str = Field(default="North", description="Categorical Region: North, South, East, West")

    @field_validator("region")
    def validate_region(cls, v: str) -> str:
        valid_regions = {"North", "South", "East", "West"}
        if v not in valid_regions:
            raise ValueError(f"Region must be one of {valid_regions}, got '{v}'")
        return v


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction request."""

    instances: List[SinglePredictionRequest]


class PredictionResponse(BaseModel):
    """Response schema for predictions."""

    prediction: Optional[float] = None
    predictions: Optional[List[float]] = None
    status: str = "success"


class DataFrameValidator:
    """Validates Pandas DataFrames against expectations before ingestion."""

    @staticmethod
    def validate_numeric_ranges(
        df,
        column_bounds: dict,
    ) -> List[str]:
        """Checks if dataframe columns fall within expected [min, max] bounds."""
        violations = []
        for col, (min_val, max_val) in column_bounds.items():
            if col in df.columns:
                actual_min = df[col].min()
                actual_max = df[col].max()
                if actual_min < min_val:
                    violations.append(f"Column '{col}' minimum {actual_min} below expected {min_val}")
                if actual_max > max_val:
                    violations.append(f"Column '{col}' maximum {actual_max} above expected {max_val}")
        return violations

    @staticmethod
    def validate_null_ratios(df, max_allowed_ratio: float = 0.2) -> List[str]:
        """Detects columns with missing value percentage higher than threshold."""
        violations = []
        for col in df.columns:
            null_ratio = df[col].isnull().mean()
            if null_ratio > max_allowed_ratio:
                violations.append(f"Column '{col}' has null ratio {null_ratio:.2%} > allowed {max_allowed_ratio:.2%}")
        return violations


"""Fluent Machine Learning Pipeline Builder."""

from typing import List, Tuple, Any
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from logger import get_logger

logger = get_logger("PipelineBuilder")


class PipelineBuilder:
    """Constructs Scikit-Learn Pipeline objects using a fluent chainable API."""

    def __init__(self):
        self.steps: List[Tuple[str, Any]] = []

    def add_step(self, name: str, transformer_or_estimator: Any) -> "PipelineBuilder":
        """Appends a named step to the execution pipeline."""
        self.steps.append((name, transformer_or_estimator))
        return self

    def with_standard_scaler(self) -> "PipelineBuilder":
        """Adds standard normalization scaler."""
        return self.add_step("scaler", StandardScaler())

    def with_regressor(self, regressor: Any = None) -> "PipelineBuilder":
        """Adds the final estimating regressor."""
        estimator = regressor or Ridge()
        return self.add_step("model", estimator)

    def build(self) -> Pipeline:
        """Assembles and returns the immutable Scikit-Learn Pipeline instance."""
        if not self.steps:
            raise ValueError("Pipeline must contain at least one step.")
        logger.info(f"Assembled Pipeline with {len(self.steps)} steps: {[s[0] for s in self.steps]}")
        return Pipeline(steps=self.steps)

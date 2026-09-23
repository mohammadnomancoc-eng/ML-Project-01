"""Streaming Online Learner and Incremental Model for ML-Project-01.

Implements online regression with SGDRegressor, exponential sample weighting for temporal adaptation,
and streaming evaluation metrics.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from logger import logger


class StreamingOnlineLearner:
    """Incremental online learning model supporting partial_fit and streaming data updates."""

    def __init__(self, learning_rate: str = "adaptive", eta0: float = 0.01, decay_factor: float = 0.995):
        self.learning_rate = learning_rate
        self.eta0 = eta0
        self.decay_factor = decay_factor
        self.model = SGDRegressor(
            loss="squared_error",
            penalty="l2",
            learning_rate=learning_rate,
            eta0=eta0,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.total_samples_seen: int = 0
        self.running_loss_: float = 0.0
        self.is_initialized: bool = False

    def partial_fit_batch(
        self, X_chunk: pd.DataFrame, y_chunk: pd.Series, sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Incrementally updates the model with a new batch/chunk of streaming data."""
        numeric_X = X_chunk.select_dtypes(include=[np.number]).fillna(0)
        
        if not self.is_initialized:
            self.scaler.partial_fit(numeric_X)
            scaled_X = self.scaler.transform(numeric_X)
            self.model.partial_fit(scaled_X, y_chunk, sample_weight=sample_weight)
            self.is_initialized = True
        else:
            self.scaler.partial_fit(numeric_X)
            scaled_X = self.scaler.transform(numeric_X)
            self.model.partial_fit(scaled_X, y_chunk, sample_weight=sample_weight)

        preds = self.model.predict(scaled_X)
        batch_loss = float(np.mean((y_chunk.values - preds) ** 2))

        # Update running exponential loss
        if self.total_samples_seen == 0:
            self.running_loss_ = batch_loss
        else:
            self.running_loss_ = self.decay_factor * self.running_loss_ + (1 - self.decay_factor) * batch_loss

        self.total_samples_seen += len(X_chunk)
        logger.info(
            f"Streaming update complete (+{len(X_chunk)} samples, total={self.total_samples_seen}). Running MSE: {self.running_loss_:.4f}"
        )
        return batch_loss

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generates predictions for incoming data using current online model state."""
        numeric_X = X.select_dtypes(include=[np.number]).fillna(0)
        scaled_X = self.scaler.transform(numeric_X)
        return self.model.predict(scaled_X)

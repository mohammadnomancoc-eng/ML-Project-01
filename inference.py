"""Inference and Prediction Pipeline Module."""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple
from model_serializer import ModelSerializer
from logger import logger


class ModelPredictor:
    """Loads saved pipeline artifacts to perform real-time and batch predictions."""

    def __init__(
        self,
        model_filename: str = "best_model.joblib",
        preprocessor_filename: str = "preprocessor.joblib",
        feature_engineer_filename: str = "feature_engineer.joblib",
    ):
        self.model = ModelSerializer.load_artifact(model_filename)
        self.preprocessor = ModelSerializer.load_artifact(preprocessor_filename)
        self.feature_engineer = ModelSerializer.load_artifact(feature_engineer_filename)
        logger.info("ModelPredictor successfully initialized with loaded artifacts.")

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Runs full inference transformation and model prediction on a batch."""
        cleaned_df = self.preprocessor.transform(df)
        engineered_df = self.feature_engineer.transform(cleaned_df)
        predictions = self.model.predict(engineered_df)
        return np.round(predictions, 4)

    def predict_single(self, sample_dict: Dict[str, Any]) -> float:
        """Predicts target for a single input record."""
        df = pd.DataFrame([sample_dict])
        preds = self.predict_batch(df)
        return float(preds[0])

    def predict_with_timing(self, sample_dict: Dict[str, Any]) -> Tuple[float, float]:
        """Returns prediction along with latency in milliseconds."""
        start = time.perf_counter()
        pred = self.predict_single(sample_dict)
        latency_ms = round((time.perf_counter() - start) * 1000, 3)
        return pred, latency_ms

    def benchmark_latency_percentiles(self, df_sample: pd.DataFrame, n_runs: int = 100) -> Dict[str, float]:
        """Measures p50, p95, and p99 inference latency percentiles over n_runs."""
        latencies = []
        sample_dict = df_sample.iloc[0].to_dict()

        for _ in range(n_runs):
            start = time.perf_counter()
            _ = self.predict_single(sample_dict)
            latencies.append((time.perf_counter() - start) * 1000)

        return {
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
            "mean_latency_ms": round(float(np.mean(latencies)), 3),
        }

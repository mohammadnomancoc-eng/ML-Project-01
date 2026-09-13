"""Inference and Prediction Pipeline Module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union
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

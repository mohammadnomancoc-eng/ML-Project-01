"""Model Training Suite supporting multiple algorithms."""

import time
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from config import config
from logger import logger


class ModelTrainer:
    """Trains and maintains machine learning model algorithms."""

    def __init__(self, random_state: int = config.RANDOM_STATE):
        self.random_state = random_state
        self.models: Dict[str, Any] = {
            "linear_regression": LinearRegression(),
            "ridge": Ridge(alpha=1.0, random_state=random_state),
            "elastic_net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state),
            "huber": HuberRegressor(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        }
        self.trained_models_: Dict[str, Any] = {}
        self.training_times_: Dict[str, float] = {}

    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Trains all registered model architectures."""
        logger.info(f"Starting training run for {len(self.models)} models on {len(X_train)} samples...")

        for name, model in self.models.items():
            start_time = time.time()
            logger.info(f"Training [{name}]...")
            model.fit(X_train, y_train)
            elapsed = time.time() - start_time

            self.trained_models_[name] = model
            self.training_times_[name] = round(elapsed, 4)
            logger.info(f"Finished [{name}] in {self.training_times_[name]}s")

        return self.trained_models_

    def get_model(self, name: str):
        """Retrieves a trained model instance by name."""
        if name not in self.trained_models_:
            raise KeyError(f"Model '{name}' has not been trained yet.")
        return self.trained_models_[name]

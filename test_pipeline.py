"""Unit and Integration Tests for ML Pipeline Components."""

import pytest
import numpy as np
import pandas as pd
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator


def test_data_loader_synthetic():
    """Verify synthetic dataset generation shapes and schema."""
    df = DataLoader.generate_synthetic_dataset(n_samples=100, n_features=4, save_to_disk=False)
    assert len(df) == 100
    assert "target" in df.columns
    assert "region" in df.columns


def test_preprocessor_imputation():
    """Verify missing values and outlier handling."""
    df = pd.DataFrame({
        "num1": [1.0, 2.0, np.nan, 1000.0],
        "cat1": ["A", "A", np.nan, "B"],
    })
    prep = DataPreprocessor()
    transformed = prep.fit_transform(df)

    assert transformed["num1"].isnull().sum() == 0
    assert transformed["cat1"].isnull().sum() == 0


def test_model_training_and_eval():
    """Verify model trainer and evaluation metric outputs."""
    df = DataLoader.generate_synthetic_dataset(n_samples=150, n_features=4, save_to_disk=False)
    X = df.drop(columns=["target", "region"])
    y = df["target"]

    trainer = ModelTrainer()
    trained_models = trainer.train_all(X, y)
    assert "ridge" in trained_models

    preds = trained_models["ridge"].predict(X)
    metrics = ModelEvaluator.evaluate_model(y, preds, "ridge")
    assert "r2_score" in metrics
    assert "rmse" in metrics

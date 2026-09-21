"""Unit and Integration Tests for ML Pipeline Components."""

import pytest
import numpy as np
import pandas as pd
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from model_serializer import ModelSerializer


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
    metrics = ModelEvaluator.evaluate_model(y, preds, "ridge", n_features=4)
    assert "r2_score" in metrics
    assert "adjusted_r2" in metrics
    assert "rmse" in metrics


def test_artifact_serialization(tmp_path):
    """Verify model saving and loading consistency."""
    dummy_obj = {"key": "test_value", "weights": [1.0, 2.0, 3.0]}
    saved_path = ModelSerializer.save_artifact(dummy_obj, "test_artifact.joblib")
    loaded_obj = ModelSerializer.load_artifact("test_artifact.joblib")

    assert loaded_obj["key"] == "test_value"
    assert loaded_obj["weights"] == [1.0, 2.0, 3.0]


def test_loss_functions():
    """Verify custom loss function calculations."""
    from loss_functions import huber_loss, quantile_loss, smape_loss, wape_loss

    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])

    assert huber_loss(y_true, y_pred) >= 0.0
    assert quantile_loss(y_true, y_pred, 0.5) >= 0.0
    assert smape_loss(y_true, y_pred) >= 0.0
    assert wape_loss(y_true, y_pred) >= 0.0


def test_learning_rate_scheduler():
    """Verify learning rate schedule trajectory generation."""
    from learning_rate_scheduler import LRScheduler

    schedule = LRScheduler.generate_schedule(
        scheduler_type="warm_restart", initial_lr=0.01, total_epochs=20, t_0=5, t_mult=2
    )
    assert len(schedule) == 20
    assert all(lr >= 0.0 for lr in schedule)


def test_data_cleaner_utilities():
    """Verify constant column dropping and missing category imputation."""
    from data_cleaner import DataCleaner

    df = pd.DataFrame({
        "constant_col": [1, 1, 1],
        "var_col": [1, 2, 3],
        "cat_col": ["a", None, "c"],
    })
    dropped = DataCleaner.drop_constant_columns(df)
    assert "constant_col" not in dropped.columns
    assert "var_col" in dropped.columns

    imputed = DataCleaner.impute_missing_categories(df)
    assert imputed["cat_col"].isnull().sum() == 0


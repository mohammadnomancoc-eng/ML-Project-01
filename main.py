"""Master CLI Entrypoint for ML-Project-01."""

import argparse
import time
import sys
import uvicorn
from data_loader import DataLoader
from preprocessor import DataPreprocessor
from feature_engineer import FeatureEngineer
from eda_analyzer import EDAAnalyzer
from model_trainer import ModelTrainer
from evaluator import ModelEvaluator
from hyperparameter_tuner import HyperparameterTuner
from visualizer import MLVisualizer
from model_serializer import ModelSerializer
from model_registry import ModelRegistry
from pipeline_profiler import PipelineProfiler
from config import config, paths
from logger import logger


def run_pipeline(enable_profiler: bool = False):
    """Executes the full machine learning training & evaluation pipeline."""
    start_total = time.time()
    logger.info("=== Starting End-to-End ML Pipeline ===")
    profiler = PipelineProfiler() if enable_profiler else None

    # 1. Generate / Load Data
    if profiler:
        with profiler.track_stage("1_Data_Generation_and_EDA"):
            df = DataLoader.generate_synthetic_dataset()
            EDAAnalyzer.generate_summary(df)
    else:
        df = DataLoader.generate_synthetic_dataset()
        EDAAnalyzer.generate_summary(df)

    # 2. Split Data
    X_train, X_val, X_test, y_train, y_val, y_test = DataLoader.split_dataset(df)

    # 3. Preprocessing (Imputation & Outlier Clipping)
    if profiler:
        with profiler.track_stage("2_Preprocessing"):
            preprocessor = DataPreprocessor()
            X_train_clean = preprocessor.fit_transform(X_train)
            X_val_clean = preprocessor.transform(X_val)
            X_test_clean = preprocessor.transform(X_test)
            ModelSerializer.save_artifact(preprocessor, "preprocessor.joblib")
    else:
        preprocessor = DataPreprocessor()
        X_train_clean = preprocessor.fit_transform(X_train)
        X_val_clean = preprocessor.transform(X_val)
        X_test_clean = preprocessor.transform(X_test)
        ModelSerializer.save_artifact(preprocessor, "preprocessor.joblib")

    # 4. Feature Engineering (Scaling & Encoders)
    if profiler:
        with profiler.track_stage("3_Feature_Engineering"):
            feature_engineer = FeatureEngineer(scaling_method="standard")
            X_train_eng = feature_engineer.fit_transform(X_train_clean)
            X_val_eng = feature_engineer.transform(X_val_clean)
            X_test_eng = feature_engineer.transform(X_test_clean)
            ModelSerializer.save_artifact(feature_engineer, "feature_engineer.joblib")
    else:
        feature_engineer = FeatureEngineer(scaling_method="standard")
        X_train_eng = feature_engineer.fit_transform(X_train_clean)
        X_val_eng = feature_engineer.transform(X_val_clean)
        X_test_eng = feature_engineer.transform(X_test_clean)
        ModelSerializer.save_artifact(feature_engineer, "feature_engineer.joblib")

    # 5. Visualizer
    viz = MLVisualizer()
    corr = EDAAnalyzer.compute_correlation_matrix(X_train_eng)
    viz.plot_correlation_heatmap(corr)

    # 6. Train Models
    if profiler:
        with profiler.track_stage("4_Model_Training"):
            trainer = ModelTrainer()
            trained_models = trainer.train_all(X_train_eng, y_train)
    else:
        trainer = ModelTrainer()
        trained_models = trainer.train_all(X_train_eng, y_train)

    # 7. Evaluate Models
    if profiler:
        with profiler.track_stage("5_Model_Evaluation"):
            eval_results = {}
            best_model_name = None
            best_r2 = -float("inf")

            for name, model in trained_models.items():
                y_pred = model.predict(X_val_eng)
                metrics = ModelEvaluator.evaluate_model(y_val, y_pred, model_name=name, n_features=X_val_eng.shape[1])
                eval_results[name] = metrics

                if metrics["r2_score"] > best_r2:
                    best_r2 = metrics["r2_score"]
                    best_model_name = name
    else:
        eval_results = {}
        best_model_name = None
        best_r2 = -float("inf")

        for name, model in trained_models.items():
            y_pred = model.predict(X_val_eng)
            metrics = ModelEvaluator.evaluate_model(y_val, y_pred, model_name=name, n_features=X_val_eng.shape[1])
            eval_results[name] = metrics

            if metrics["r2_score"] > best_r2:
                best_r2 = metrics["r2_score"]
                best_model_name = name

    leaderboard = ModelEvaluator.compare_models(eval_results)
    logger.info(f"\nModel Leaderboard:\n{leaderboard}")

    # 8. Save Best Model & Register in Registry
    best_model = trainer.get_model(best_model_name)
    ModelSerializer.save_artifact(
        best_model,
        "best_model.joblib",
        metadata={"algorithm": best_model_name, "best_r2_val": best_r2},
    )

    registry = ModelRegistry()
    registry.register_model(
        model_name=best_model_name,
        artifact_filename="best_model.joblib",
        version="v1.1.0",
        stage="Production",
        metrics=eval_results.get(best_model_name, {}),
        description=f"Best evaluated model ({best_model_name}) from pipeline run.",
    )

    # 9. Test set prediction visualization & residual analysis
    y_test_pred = best_model.predict(X_test_eng)
    viz.plot_predictions_vs_actual(y_test.values, y_test_pred)
    viz.plot_residual_distribution(y_test.values, y_test_pred)

    if profiler:
        print("\n--- Pipeline Profiler Report ---")
        print(profiler.get_summary_report())

    elapsed_total = round(time.time() - start_total, 2)
    logger.info(f"=== Pipeline Completed Successfully in {elapsed_total}s ===")


def main():
    parser = argparse.ArgumentParser(description="ML-Project-01 CLI")
    parser.add_argument(
        "--mode",
        choices=["train", "serve", "profile", "registry"],
        default="train",
        help="Execute training pipeline, launch REST API server, profile latency, or inspect registry",
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for FastAPI server")
    args = parser.parse_args()

    if args.mode in ["train", "profile"]:
        run_pipeline(enable_profiler=(args.mode == "profile"))
    elif args.mode == "serve":
        logger.info(f"Starting FastAPI server on http://0.0.0.0:{args.port}")
        uvicorn.run("api:app", host="0.0.0.0", port=args.port, reload=True)
    elif args.mode == "registry":
        registry = ModelRegistry()
        print("Registry Manifest:", registry._load_manifest())


if __name__ == "__main__":
    main()

# ML-Project-01 — API Reference 📚

Complete technical documentation for the machine learning modules, core classes, and REST serving endpoints.

---

## 🔌 REST API Endpoints

### `GET /health`
Returns system health and predictor status.
- **Response:**
  ```json
  {
    "status": "healthy",
    "model_loaded": true
  }
  ```

### `POST /predict`
Performs real-time single-sample regression inference.
- **Request Body:**
  ```json
  {
    "feature_1": 0.45,
    "feature_2": -1.2,
    "feature_3": 3.14,
    "feature_4": 0.88,
    "feature_5": -0.15,
    "feature_6": 1.02,
    "feature_7": -0.77,
    "feature_8": 0.33,
    "region": "North"
  }
  ```
- **Response:**
  ```json
  {
    "prediction": 128.45,
    "status": "success"
  }
  ```

---

## 🐍 Python Core Modules

| Module | Primary Class | Purpose |
|---|---|---|
| `data_loader.py` | `DataLoader` | Synthetic dataset generation & 3-way splits |
| `preprocessor.py` | `DataPreprocessor` | Missing value imputation & IQR outlier clipping |
| `feature_engineer.py` | `FeatureEngineer` | Dynamic scaling (Standard/MinMax/Robust) & One-Hot Encoding |
| `feature_selector.py` | `FeatureSelector` | VarianceThreshold & Mutual Information selection |
| `cross_validator.py` | `CrossValidator` | K-Fold cross-validation score & variance metrics |
| `ensemble_model.py` | `EnsembleBuilder` | Voting & Stacking Regressor meta-models |
| `outlier_detector.py` | `OutlierDetector` | Isolation Forest & LOF anomaly detection |
| `data_drift_detector.py` | `DataDriftDetector` | Kolmogorov-Smirnov & PSI statistical drift checks |
| `model_trainer.py` | `ModelTrainer` | Multi-algorithm estimator fitting & time tracking |
| `evaluator.py` | `ModelEvaluator` | RMSE, MAE, R², and leaderboard ranking |
| `inference.py` | `ModelPredictor` | Real-time & batch prediction pipeline |
| `batch_processor.py` | `BatchProcessor` | Streamed chunked CSV batch inference |
| `alert_notifier.py` | `AlertNotifier` | Webhook alert dispatcher |
| `database_connector.py` | `DatabaseConnector` | SQLite & SQL database connector |

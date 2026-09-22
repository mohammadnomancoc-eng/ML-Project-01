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
| `data_sampler.py` | `DataSampler` | Reservoir sampling, stratified quantile sampling, and balanced subsets |
| `preprocessor.py` | `DataPreprocessor` | Missing value imputation & IQR outlier clipping |
| `missing_value_handler.py` | `MissingValueHandler` | KNN imputation, simple median, and missingness indicators |
| `feature_engineer.py` | `FeatureEngineer` | Dynamic scaling (Standard/MinMax/Robust) & One-Hot Encoding |
| `interaction_generator.py` | `FeatureInteractionGenerator` | Pairwise product, ratio, and difference feature synthesis |
| `leakage_detector.py` | `DataLeakageDetector` | Target correlation leak and split contamination auditing |
| `concept_drift_detector.py` | `ConceptDriftDetector` | Page-Hinkley streaming error drift and degradation detector |
| `feature_selector.py` | `FeatureSelector` | VarianceThreshold & Mutual Information selection |
| `cross_validator.py` | `CrossValidator` | K-Fold cross-validation score & variance metrics |
| `ensemble_model.py` | `EnsembleBuilder` | Voting & Stacking Regressor meta-models |
| `anomaly_scorer.py` | `EnsembleAnomalyScorer` | Multi-method consensus Isolation Forest & LOF anomaly scorer |
| `outlier_detector.py` | `OutlierDetector` | Isolation Forest & LOF anomaly detection |
| `data_drift_detector.py` | `DataDriftDetector` | Kolmogorov-Smirnov & PSI statistical drift checks |
| `threshold_optimizer.py` | `ThresholdOptimizer` | Decision threshold tuning for F-beta, Youden's J, and business cost matrices |
| `adversarial_tester.py` | `AdversarialRobustnessTester` | Gaussian noise injection and feature dropout stress testing |
| `model_registry.py` | `ModelRegistry` | Semantic artifact versioning and staging lifecycle manager |
| `counterfactual_explainer.py` | `CounterfactualExplainer` | Minimal perturbation search and actionable counterfactuals |
| `pipeline_profiler.py` | `PipelineProfiler` | Stage-by-stage latency profiler and Markdown waterfall reports |
| `data_contract.py` | `DataContract` | Strict schema validation, value bounds, and allowed category constraints |
| `model_trainer.py` | `ModelTrainer` | Multi-algorithm estimator fitting & time tracking |
| `evaluator.py` | `ModelEvaluator` | RMSE, MAE, R², and leaderboard ranking |
| `loss_functions.py` | `huber_loss`, `wape_loss` | Robust and percentage-based regression loss functions |
| `learning_rate_scheduler.py` | `LRScheduler` | Step decay, exponential, and SGDR warm restart schedulers |
| `early_stopping.py` | `EarlyStopping` | Loss plateau detector with warmup epochs and relative tolerance |
| `data_cleaner.py` | `DataCleaner` | String normalization, constant dropping & category imputation |
| `validator.py` | `DataFrameValidator` | Pydantic API schemas and DataFrame bounds validator |
| `system_health.py` | `SystemHealthMonitor` | Hardware telemetry, disk space and memory monitoring |
| `metric_tracker.py` | `MetricTracker` | Real-time moving averages and Markdown summary tables |
| `cost_analyzer.py` | `BusinessCostAnalyzer` | Asymmetric business loss and cloud hosting cost estimator |
| `model_card.py` | `ModelCardGenerator` | Automated Markdown model cards and JSON metadata manifests |
| `synthetic_generator.py` | `SyntheticAugmenter` | Gaussian jitter, interpolation & synthetic outlier injection |
| `inference.py` | `ModelPredictor` | Real-time & batch prediction pipeline |
| `batch_processor.py` | `BatchProcessor` | Streamed chunked CSV batch inference |
| `alert_notifier.py` | `AlertNotifier` | Webhook alert dispatcher |
| `database_connector.py` | `DatabaseConnector` | SQLite & SQL database connector |

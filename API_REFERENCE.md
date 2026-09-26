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

## 🐍 Python Core & Extended Modules

| Module | Primary Class | Purpose |
|---|---|---|
| `data_loader.py` | `DataLoader` | Synthetic dataset generation & 3-way splits |
| `data_sampler.py` | `DataSampler` | Reservoir sampling, stratified quantile sampling, and balanced subsets |
| `data_sanitizer.py` | `DataSanitizer` | Automated PII masking, email/phone scrubbing, and data anonymization |
| `dataset_fingerprint.py` | `DatasetFingerprinter` | SHA-256 cryptographic dataset hashing and statistical moment signatures |
| `preprocessor.py` | `DataPreprocessor` | Missing value imputation & IQR outlier clipping |
| `missing_value_handler.py` | `MissingValueHandler` | KNN imputation, simple median, and missingness indicators |
| `feature_engineer.py` | `FeatureEngineer` | Dynamic scaling (Standard/MinMax/Robust) & One-Hot Encoding |
| `feature_clusterer.py` | `FeatureClusterer` | Hierarchical correlation clustering and redundancy reduction |
| `target_encoder.py` | `OutOfFoldTargetEncoder` | Out-of-fold target encoding with Bayesian smoothing |
| `feature_hash_encoder.py` | `FeatureHashEncoder` | Fixed-memory hashing trick for high-cardinality categoricals |
| `interaction_generator.py` | `FeatureInteractionGenerator` | Pairwise product, ratio, and difference feature synthesis |
| `feature_cache.py` | `FeatureStoreCache` | Offline-to-online TTL feature store cache |
| `leakage_detector.py` | `DataLeakageDetector` | Target correlation leak and split contamination auditing |
| `concept_drift_detector.py` | `ConceptDriftDetector` | Page-Hinkley streaming error drift and degradation detector |
| `drift_alerter.py` | `DriftAlerter` | Multi-channel Slack, Discord, and webhook alert dispatcher |
| `threshold_optimizer.py` | `ThresholdOptimizer` | Decision threshold tuning for F-beta, Youden's J, and business cost matrices |
| `model_calibrator.py` | `ModelCalibrator` | Platt scaling, Isotonic regression, and ECE metric calibrator |
| `conformal_predictor.py` | `ConformalPredictor` | Split conformal prediction intervals with coverage guarantees |
| `subgroup_auditor.py` | `SubgroupAuditor` | Disparity ratio, bias, and underperforming cohort slice auditor |
| `adversarial_tester.py` | `AdversarialRobustnessTester` | Gaussian noise injection and feature dropout stress testing |
| `model_registry.py` | `ModelRegistry` | Semantic artifact versioning and staging lifecycle manager |
| `model_lineage.py` | `LineageTracker` | DAG artifact lineage provenance and checksum tracker |
| `counterfactual_explainer.py` | `CounterfactualExplainer` | Minimal perturbation search and actionable counterfactuals |
| `shap_approximator.py` | `ShapleyApproximator` | Fast Monte-Carlo Shapley feature attribution |
| `bayesian_optimizer.py` | `BayesianHyperparameterOptimizer`| Gaussian Process SMBO with Expected Improvement acquisition |
| `streaming_learner.py` | `StreamingOnlineLearner` | Online SGD regressor with incremental partial_fit updates |
| `tabular_synthesizer.py` | `TabularCopulaSynthesizer` | Gaussian Copula multivariate tabular data synthesizer |
| `sample_weigher.py` | `SampleWeigher` | Temporal exponential decay, inverse class balance, and density weights |
| `multiclass_evaluator.py` | `MulticlassEvaluator` | Multiclass F1-macro/weighted, Cohen's Kappa, and MCC reports |
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
| `database_connector.py` | `DatabaseConnector` | SQLite & SQL database connector |
| `data_quality_auditor.py` | `DataQualityAuditor` | Automated dataset health and completeness auditing |
| `distribution_drift.py` | `DistributionDriftEstimator` | Wasserstein Earth Mover's Distance distribution drift detection |
| `data_slice_miner.py` | `DataSliceMiner` | High-error subpopulation cohort discovery |
| `uncertainty_estimator.py` | `EnsembleUncertaintyEstimator` | Tree variance and epistemic uncertainty bounds |
| `pipeline_checkpoint.py` | `PipelineCheckpointManager` | Intermediate pipeline state snapshot caching |
| `residual_analyzer.py` | `ResidualDiagnosticAnalyzer` | Durbin-Watson and heteroscedasticity diagnostic checks |
| `cloud_cost_optimizer.py` | `CloudCostOptimizer` | Latency SLA instance sizing and cost optimizer |
| `feature_selector_v2.py` | `RecursiveFeatureSelector` | RFECV recursive feature elimination with cross-validation |
| `model_rule_exporter.py` | `ModelRuleExporter` | Portable model formula and rule extraction |


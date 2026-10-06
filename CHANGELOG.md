# Changelog 📋

All notable changes to the **ML-Project-01** repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.7.0] — 2026-10-06

### Added
- **Temporal Drift Synthesis**: `add_temporal_trend` method in `data_loader.py`.
- **Quantile Fingerprinting**: `compute_quantile_fingerprint` in `dataset_fingerprint.py`.
- **Target Duplicate Auditor**: `detect_duplicate_target_leakage` in `leakage_detector.py`.
- **Streaming Drift Reporter**: `get_drift_status_summary` in `concept_drift_detector.py`.
- **Equity Disparity Metric**: `compute_equity_disparity_score` in `subgroup_auditor.py`.
- **Noise Sensitivity Profiler**: `evaluate_noise_sensitivity_curve` in `adversarial_tester.py`.
- **Stage Version Filtering**: `list_registered_versions_by_stage` in `model_registry.py`.
- **Lineage Provenance Summary**: `get_provenance_graph_summary` in `model_lineage.py`.
- **Counterfactual Delta Metrics**: `get_counterfactual_delta_summary` in `counterfactual_explainer.py`.
- **Top-K Shapley Filter**: `get_top_k_attributions` in `shap_approximator.py`.
- **Bayesian Trajectory Tracker**: `get_convergence_trajectory` in `bayesian_optimizer.py`.
- **Copula Correlation Score**: `get_correlation_similarity_score` in `tabular_synthesizer.py`.
- **Weight Clamping Utility**: `clip_extreme_weights` in `sample_weigher.py`.
- **Extended Test Suite V7**: Comprehensive unit tests for analytical and diagnostic helpers.

---

## [2.6.0] — 2026-10-04

### Added
- **Environment Telemetry**: `get_environment_info` helper in `config.py`.
- **Sampling Stats Reporter**: `get_sampling_stats` ratio helper in `data_sampler.py`.
- **Imputation Pipeline Summary**: `get_imputation_summary` in `missing_value_handler.py`.
- **Hash Collision Estimator**: `estimate_collision_rate` helper in `feature_hash_encoder.py`.
- **Target Encoding Map Accessor**: `get_feature_encoding_map` in `target_encoder.py`.
- **Interaction Column Inspector**: `get_generated_feature_names` in `interaction_generator.py`.
- **Brier Score Calculation**: `brier_score` evaluation metric in `model_calibrator.py`.
- **Cost Curve Trajectory Generator**: `generate_cost_curve` in `threshold_optimizer.py`.
- **Extended Test Suite V6**: Unit tests for sampling, hashing, calibration, and encoder helpers.

---

## [2.5.0] — 2026-10-03

### Added
- **Quantile Clipping**: `clip_extreme_quantiles` static method in `preprocessor.py`.
- **Quantile Transformation**: Added `"quantile"` uniform scaling method in `feature_engineer.py`.
- **SMAPE Regression Metric**: Symmetric Mean Absolute Percentage Error in `evaluator.py`.
- **Relative Squared Error (RSE)**: Loss calculation helper in `loss_functions.py`.
- **Skewness Quality Auditing**: Added automatic skewness threshold warning checks in `data_quality_auditor.py`.
- **IPv6 Redaction**: Added IPv6 address regex pattern masking in `data_sanitizer.py`.
- **Exponential Variance**: `compute_exponential_variance` calculation in `metric_tracker.py`.
- **Patience Remaining Property**: `patience_remaining` property helper in `early_stopping.py`.
- **Correlation Ranking Plot**: `plot_feature_correlation_ranking` horizontal bar chart in `visualizer.py`.
- **KS Residual Normality Test**: Kolmogorov-Smirnov test in `residual_analyzer.py`.
- **Conformal Interval Width Ratio**: `compute_relative_interval_width` in `conformal_predictor.py`.
- **Cluster Centroids**: `get_cluster_centroids` calculation in `feature_clusterer.py`.
- **Alert Breakdown by Severity**: `get_alert_counts_by_severity` in `drift_alerter.py`.
- **Artifact Metadata Helper**: `get_artifact_metadata` in `model_serializer.py`.
- **Thread & Process Telemetry**: `get_process_thread_info` helper in `system_health.py`.
- **Annual Cloud Cost Projections**: `project_annual_cloud_cost` in `cost_analyzer.py`.
- **HTML Sanitization**: `strip_html_tags` text sanitization in `data_cleaner.py`.
- **Population Stability Index (PSI)**: `calculate_population_stability_index` in `distribution_drift.py`.
- **Slice Query Filtering**: `filter_by_slice` helper in `data_slice_miner.py`.
- **Uncertainty Normalization**: `normalize_uncertainty_scores` in `uncertainty_estimator.py`.
- **Checkpoint Cache Management**: `clear_checkpoints` in `pipeline_checkpoint.py`.
- **Multi-Region Cost Sizing**: Multi-region cost latency multipliers in `cloud_cost_optimizer.py`.
- **Decision Tree Rule Export**: `export_decision_tree_rules` extractor in `model_rule_exporter.py`.
- **Column Type Validation**: `validate_column_types` schema checker in `validator.py`.
- **1-Cycle Learning Rate**: `one_cycle_lr` schedule in `learning_rate_scheduler.py`.

---

## [2.4.0] — 2026-10-01

### Added
- **Sample Weights Generator**: `generate_sample_weights` decay method in `data_loader.py`.
- **Relative RMSE Metric**: `rrmse` error normalization metric in `evaluator.py`.
- **ML Environment Inspector**: `get_ml_env_vars` helper in `system_health.py`.
- **Callable Execution Benchmark**: `measure_execution_overhead` timing helper in `benchmark.py`.

---

## [2.3.0] — 2026-09-30

### Added
- **JSON Configuration Loader**: `load_from_json` classmethod in `config.py`.
- **Mean Bias Deviation**: `mean_bias_deviation` regression metric in `evaluator.py`.
- **Smooth L1 Loss**: `smooth_l1_loss` loss computation in `loss_functions.py`.
- **Duplicate Row Ratio**: Automatic duplicate row audit metric in `data_quality_auditor.py`.
- **Passport PII Pattern**: Regex pattern masking for passport numbers in `data_sanitizer.py`.
- **Metric History Reset**: `reset()` method in `metric_tracker.py`.
- **Early Stopping Reset**: `reset()` method in `early_stopping.py`.
- **Learning Curves Plotter**: `plot_learning_curves` in `visualizer.py`.
- **Leverage Outlier Detector**: `detect_influential_outliers` in `residual_analyzer.py`.
- **Conformal Coverage Summary**: `get_coverage_summary` in `conformal_predictor.py`.
- **Cluster Count Property**: `n_clusters` property in `feature_clusterer.py`.
- **Teams Webhook Formatter**: `build_teams_alert` in `drift_alerter.py`.
- **Artifact Existence Check**: `artifact_exists` method in `model_serializer.py`.

---

## [2.1.0] — 2026-09-24

### Added
- **Seed Reproducibility**: `set_seed` helper for deterministic random number generation in `config.py`.
- **Quantile Stratified Splits**: Option to stratify continuous target values across bins during 3-way dataset splits in `data_loader.py`.
- **Robust MAD Outlier Clipping**: Median Absolute Deviation (MAD) alternative to IQR in `preprocessor.py`.
- **Power Transformations**: Box-Cox / Yeo-Johnson variance stabilizing transforms in `feature_engineer.py`.
- **MSLE Evaluation Metric**: Mean Squared Logarithmic Error tracking in `evaluator.py`.
- **Training Timing Metadata**: Structured dictionary breakdown of algorithm training durations in `model_trainer.py`.
- **Randomized Parameter Search**: RandomizedSearchCV sweep method in `hyperparameter_tuner.py`.
- **Quantile-Quantile Normality Plots**: Residual Q-Q plot diagnostic generation in `visualizer.py`.
- **Artifact SHA-256 Checksums**: Automated checksum generation and integrity verification in `model_serializer.py`.
- **Latency Percentile Profiling**: Real-time measurement of p50, p95, and p99 inference latency in `inference.py`.
- **Prometheus Telemetry Endpoint**: Operational `/metrics` route in `api.py`.
- **Schema Column Validation**: Pre-ingestion column existence checks in `validator.py`.
- **Low-Variance Feature Filtering**: Automated removal of near-constant numeric columns in `data_cleaner.py`.
- **Carbon Footprint Estimator**: Compute energy consumption and CO2 emission modeling in `cost_analyzer.py`.
- **Hardware Acceleration Telemetry**: GPU and CUDA driver detection in `system_health.py`.
- **CSV Metric History Export**: Exporting live training moving averages to CSV in `metric_tracker.py`.
- **BibTeX Citation Generator**: Automated citation block generation for model cards in `model_card.py`.
- **Early Stopping Weights Tracker**: Best weights storage and restore state mechanism in `early_stopping.py`.
- **Polynomial LR Decay**: Polynomial decay schedule generator in `learning_rate_scheduler.py`.
- **MASE Regression Loss**: Mean Absolute Scaled Error for time-series forecasting in `loss_functions.py`.
- **Systematic Interval Sampling**: Periodic deterministic sampling in `data_sampler.py`.
- **Custom PII Pattern Registry**: Runtime registration of custom sensitive regex tokens in `data_sanitizer.py`.
- **Cluster Group Mapping**: Key-value cluster grouping dictionary export in `feature_clusterer.py`.
- **Conformal Margin Getter**: Direct access to calibrated uncertainty bounds in `conformal_predictor.py`.
- **Alert History Management**: In-memory history clearing utility in `drift_alerter.py`.
- **Comprehensive Unit Testing**: Expanded unit tests for estimators and loss functions in `test_extended_suite.py`.

---

## [2.0.0] — 2026-09-23

### Added
- Split conformal prediction intervals with distribution-free coverage guarantees (`conformal_predictor.py`).
- End-to-end DAG lineage provenance graph and checksum tracking (`model_lineage.py`).
- Regex-based PII masking and data sanitization engine (`data_sanitizer.py`).
- Gaussian Process surrogate Bayesian hyperparameter optimization (`bayesian_optimizer.py`).
- Streaming SGD online learner with partial_fit updates (`streaming_learner.py`).
- Gaussian Copula high-fidelity tabular data synthesizer (`tabular_synthesizer.py`).
- Multi-method ensemble anomaly scorer combining Isolation Forest and LOF (`anomaly_scorer.py`).
- Counterfactual minimal perturbation search explainer (`counterfactual_explainer.py`).
- Pipeline stage latency and memory profiler with markdown waterfall reports (`pipeline_profiler.py`).
- Local model registry with semantic versioning and lifecycle stage promotion (`model_registry.py`).
- Strict data contract and schema validation engine (`data_contract.py`).

---

## [1.0.0] — 2026-09-22

### Added
- Initial modular ML pipeline architecture with multi-algorithm training, evaluation, and FastAPI serving.

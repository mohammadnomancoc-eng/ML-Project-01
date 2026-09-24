# Changelog 📋

All notable changes to the **ML-Project-01** repository are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

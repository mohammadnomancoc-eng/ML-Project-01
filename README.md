# ML-Project-01 — End-to-End Machine Learning Pipeline 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, enterprise-ready machine learning framework and predictive analytics platform built in Python. Designed with a modular architecture covering synthetic data generation, PII sanitization, advanced preprocessing, feature engineering, surrogate Bayesian optimization, online streaming learning, model training, conformal prediction intervals, DAG lineage tracking, and REST API deployment.

---

## 📌 Project Architecture

```text
ML-Project-01/
├── config.py                 # Centralized configuration & hyperparameters
├── data_loader.py            # Synthetic dataset generator & CSV loaders
├── data_sampler.py           # Reservoir, stratified, and balanced data sampler
├── data_sanitizer.py         # Regex PII scrubber (emails, cards, SSNs, IPs)
├── dataset_fingerprint.py    # SHA-256 data hashing & statistical moments
├── preprocessor.py           # Missing value imputation & outlier clipping
├── missing_value_handler.py  # KNN and multivariate missing value imputation
├── feature_engineer.py       # Scalers, encoders, and feature transformers
├── feature_clusterer.py      # Correlation clustering & multicollinearity reduction
├── target_encoder.py         # Out-of-fold target encoding with smoothing
├── feature_hash_encoder.py   # Fixed-memory hashing trick for high-cardinality
├── interaction_generator.py  # Pairwise feature interaction synthesizer
├── feature_cache.py          # Offline-to-online TTL feature store cache
├── leakage_detector.py       # Target leakage and split contamination auditing
├── concept_drift_detector.py # Page-Hinkley streaming error drift tracker
├── drift_alerter.py          # Slack/Discord webhook alert dispatcher
├── threshold_optimizer.py    # Cost-benefit and F-beta threshold tuner
├── model_calibrator.py       # Platt scaling & Isotonic probability calibration
├── conformal_predictor.py    # Split conformal intervals & coverage guarantees
├── subgroup_auditor.py       # Demographic slice discovery & disparity auditor
├── adversarial_tester.py     # Noise injection and perturbation stress tester
├── model_registry.py         # Versioning and lifecycle staging registry
├── model_lineage.py          # DAG provenance graph & artifact checksums
├── anomaly_scorer.py         # Multi-method ensemble anomaly scorer
├── counterfactual_explainer.py# Minimal edit counterfactual explainer
├── shap_approximator.py      # Monte Carlo Shapley attribution engine
├── bayesian_optimizer.py     # Gaussian Process SMBO hyperparameter tuner
├── streaming_learner.py      # Online SGD regressor with partial_fit updates
├── tabular_synthesizer.py    # Gaussian Copula tabular data generator
├── sample_weigher.py         # Temporal decay & inverse frequency weigher
├── multiclass_evaluator.py   # Multiclass F1, Kappa, and MCC evaluator
├── pipeline_profiler.py      # Execution latency and waterfall profiler
├── data_contract.py          # Strict data contract and schema validator
├── eda_analyzer.py           # Exploratory data analysis & statistical summaries
├── model_trainer.py          # Multi-algorithm model training suite
├── hyperparameter_tuner.py   # Grid & random search optimization engine
├── evaluator.py              # Regression & classification metric reports
├── visualizer.py             # Performance curves & feature importance plots
├── model_serializer.py       # Model & pipeline artifact persistence
├── inference.py              # Real-time batch & single-sample inference engine
├── api.py                    # FastAPI REST serving endpoints
├── validator.py              # Pydantic input schemas & validation
├── logger.py                 # Structured experiment logging
├── benchmark.py              # Latency profiling & throughput benchmark
└── main.py                   # Master CLI entrypoint (train/serve/profile/registry/lineage)
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/mohammadnomancoc-eng/ML-Project-01.git
cd ML-Project-01

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
python main.py --mode train
```

### 3. Profile Execution Latency

```bash
python main.py --mode profile
```

### 4. Inspect Model Registry & Lineage

```bash
python main.py --mode registry
python main.py --mode lineage
```

### 5. Launch Inference REST API

```bash
python main.py --mode serve --port 8000
```

### 6. Run Full Test Suite

```bash
python -m unittest discover -p "test_*.py"
```

---

## 🛠️ Pipeline Highlights & Key Features

- **Conformal Prediction Guarantees**: Non-parametric split conformal intervals ($1 - \alpha$ confidence) via [conformal_predictor.py](file:///f:/Projects/ML%20project%201/conformal_predictor.py).
- **DAG Lineage & Checksums**: End-to-end provenance graphs and SHA-256 artifact auditing via [model_lineage.py](file:///f:/Projects/ML%20project%201/model_lineage.py).
- **Data Sanitization & PII Scrubbing**: Automated redacting of emails, phone numbers, and SSNs via [data_sanitizer.py](file:///f:/Projects/ML%20project%201/data_sanitizer.py).
- **Surrogate Bayesian Optimization**: Gaussian Process SMBO with Expected Improvement via [bayesian_optimizer.py](file:///f:/Projects/ML%20project%201/bayesian_optimizer.py).
- **Online Streaming Learning**: Incremental SGD updates with time-decay sample weighting via [streaming_learner.py](file:///f:/Projects/ML%20project%201/streaming_learner.py).
- **Tabular Data Synthesis**: High-fidelity Gaussian Copula synthetic record generation via [tabular_synthesizer.py](file:///f:/Projects/ML%20project%201/tabular_synthesizer.py).
- **Model Governance & Registry**: Semantic versioning and lifecycle stage transitions via [model_registry.py](file:///f:/Projects/ML%20project%201/model_registry.py).

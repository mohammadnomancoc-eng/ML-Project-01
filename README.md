# ML-Project-01 — End-to-End Machine Learning Pipeline 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, production-ready machine learning framework and predictive analytics pipeline built in Python. Designed with modular architecture covering data generation, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, artifact serialization, model registry, and REST API deployment.

---

## 📌 Project Architecture

```text
ML-Project-01/
├── config.py                 # Centralized configuration & hyperparameters
├── data_loader.py            # Synthetic dataset generator & CSV loaders
├── data_sampler.py           # Reservoir, stratified, and balanced data sampler
├── preprocessor.py           # Missing value imputation & outlier handling
├── missing_value_handler.py  # KNN and multivariate missing value imputation
├── feature_engineer.py       # Scalers, encoders, and feature transformers
├── interaction_generator.py  # Pairwise feature interaction synthesizer
├── leakage_detector.py       # Target leakage and split contamination auditing
├── concept_drift_detector.py # Page-Hinkley streaming error drift tracker
├── threshold_optimizer.py    # Cost-benefit and F-beta threshold tuner
├── adversarial_tester.py     # Noise injection and feature perturbation stress tester
├── model_registry.py         # Versioning and lifecycle staging registry
├── anomaly_scorer.py         # Multi-method ensemble anomaly scorer
├── counterfactual_explainer.py# Minimal edit counterfactual explainer
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
└── main.py                   # Master CLI entrypoint (train/serve/profile/registry)
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

### 4. Launch Inference REST API

```bash
python main.py --mode serve --port 8000
```

### 5. Run Test Suite

```bash
python test_pipeline.py
# or with pytest:
pytest test_pipeline.py -v
```

---

## 🛠️ Pipeline Highlights & Key Features

- **Model Registry & Staging**: Versioned artifacts and lifecycle promotion (`model_registry.py`).
- **Data Contract Validation**: Strict column type, bound, and category constraints (`data_contract.py`).
- **Concept & Data Drift**: Real-time Page-Hinkley error drift and KS-test detectors (`concept_drift_detector.py`).
- **Adversarial Robustness**: Perturbation stress testing and feature dropout impact (`adversarial_tester.py`).
- **Decision Threshold Optimization**: Business cost matrix and F-beta optimization (`threshold_optimizer.py`).
- **Counterfactual Explanations**: Actionable minimal feature edit recommendations (`counterfactual_explainer.py`).
- **Pipeline Profiling**: Stage latency waterfall markdown report generation (`pipeline_profiler.py`).

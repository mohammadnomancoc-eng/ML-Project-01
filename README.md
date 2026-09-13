# ML-Project-01 — End-to-End Machine Learning Pipeline 🧠⚡

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A robust, production-ready machine learning framework and predictive analytics pipeline built in Python. Designed with modular architecture covering data generation, preprocessing, feature engineering, model training, hyperparameter tuning, evaluation, artifact serialization, and REST API deployment.

---

## 📌 Project Architecture

```text
ML-Project-01/
├── config.py                 # Centralized configuration & hyperparameters
├── data_loader.py            # Synthetic dataset generator & CSV loaders
├── preprocessor.py           # Missing value imputation & outlier handling
├── feature_engineer.py       # Scalers, encoders, and feature transformers
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
└── main.py                   # CLI entrypoint for running the pipeline
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

### 3. Launch Inference REST API

```bash
python main.py --mode serve --port 8000
```

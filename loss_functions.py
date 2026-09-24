"""Custom Regression Loss Functions and Evaluation Metrics."""

import numpy as np


def huber_loss(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.35) -> float:
    """Computes robust Huber loss (less sensitive to outliers than MSE)."""
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * (error**2)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return float(np.mean(np.where(is_small_error, squared_loss, linear_loss)))


def quantile_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.5) -> float:
    """Computes pinball/quantile loss for asymmetric risk optimization."""
    error = y_true - y_pred
    return float(np.mean(np.maximum(quantile * error, (quantile - 1) * error)))


def log_cosh_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Log-Cosh loss (smooth approximation of MAE)."""
    error = y_pred - y_true
    return float(np.mean(np.log(np.cosh(error + 1e-12))))


def mape_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Mean Absolute Percentage Error (MAPE)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def smape_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Symmetric Mean Absolute Percentage Error (SMAPE)."""
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    return float(np.mean(np.abs(y_pred[mask] - y_true[mask]) / denominator[mask]) * 100)


def wape_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Weighted Absolute Percentage Error (WAPE)."""
    total_actual = np.sum(np.abs(y_true))
    if total_actual == 0:
        return 0.0
    return float((np.sum(np.abs(y_true - y_pred)) / total_actual) * 100)


def mase_loss(y_true: np.ndarray, y_pred: np.ndarray, y_train_baseline: np.ndarray = None) -> float:
    """Computes Mean Absolute Scaled Error (MASE)."""
    mae = np.mean(np.abs(y_true - y_pred))
    if y_train_baseline is not None and len(y_train_baseline) > 1:
        scale = np.mean(np.abs(np.diff(y_train_baseline)))
    else:
        scale = np.mean(np.abs(np.diff(y_true))) if len(y_true) > 1 else 1.0

    return float(mae / (scale + 1e-8))



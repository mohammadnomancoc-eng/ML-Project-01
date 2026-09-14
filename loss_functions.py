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

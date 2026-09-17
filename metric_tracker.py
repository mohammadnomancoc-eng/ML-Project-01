"""Real-time Training Metric Accumulator & Moving Averages."""

from collections import defaultdict
from typing import Dict, List, Any
import numpy as np


class MetricTracker:
    """Accumulates step/epoch metrics and calculates exponential moving averages."""

    def __init__(self, ema_alpha: float = 0.1):
        self.ema_alpha = ema_alpha
        self.history: Dict[str, List[float]] = defaultdict(list)
        self.ema_values: Dict[str, float] = {}

    def update(self, metric_name: str, value: float) -> float:
        """Appends value and updates Exponential Moving Average."""
        self.history[metric_name].append(value)
        if metric_name not in self.ema_values:
            self.ema_values[metric_name] = value
        else:
            self.ema_values[metric_name] = (
                self.ema_alpha * value + (1 - self.ema_alpha) * self.ema_values[metric_name]
            )
        return self.ema_values[metric_name]

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Returns statistical summary of all tracked metrics."""
        summary = {}
        for name, values in self.history.items():
            arr = np.array(values)
            summary[name] = {
                "latest": round(float(arr[-1]), 4) if len(arr) > 0 else 0.0,
                "ema": round(float(self.ema_values.get(name, 0.0)), 4),
                "mean": round(float(np.mean(arr)), 4),
                "min": round(float(np.min(arr)), 4),
                "max": round(float(np.max(arr)), 4),
            }
        return summary

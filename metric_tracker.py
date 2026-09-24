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

    def export_markdown_table(self) -> str:
        """Formats the metric summary into a GitHub-flavored Markdown table."""
        summary = self.get_summary()
        headers = ["Metric", "Latest", "EMA", "Mean", "Min", "Max"]
        lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        for metric, stats in summary.items():
            lines.append(
                f"| {metric} | {stats['latest']} | {stats['ema']} | {stats['mean']} | {stats['min']} | {stats['max']} |"
            )
        return "\n".join(lines)

    def export_csv(self, file_path: str) -> None:
        """Exports tracked metric history to CSV."""
        import pandas as pd

        df = pd.DataFrame(dict(self.history))
        df.to_csv(file_path, index_label="step")


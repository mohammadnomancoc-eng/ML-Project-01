"""Business Cost Matrix and ROI Impact Analyzer."""

import numpy as np
from typing import Dict, Any
from logger import get_logger

logger = get_logger("CostAnalyzer")


class BusinessCostAnalyzer:
    """Translates regression prediction error distributions into business dollar cost impacts."""

    def __init__(self, cost_underestimate_per_unit: float = 25.0, cost_overestimate_per_unit: float = 10.0):
        self.cost_under = cost_underestimate_per_unit
        self.cost_over = cost_overestimate_per_unit

    def calculate_monetary_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculates asymmetric financial loss associated with prediction residuals."""
        errors = y_pred - y_true
        overestimates = np.maximum(0, errors)
        underestimates = np.maximum(0, -errors)

        cost_over_total = float(np.sum(overestimates * self.cost_over))
        cost_under_total = float(np.sum(underestimates * self.cost_under))
        total_loss = cost_over_total + cost_under_total
        avg_loss_per_case = total_loss / len(y_true) if len(y_true) > 0 else 0.0

        report = {
            "total_financial_loss": round(total_loss, 2),
            "average_loss_per_prediction": round(avg_loss_per_case, 2),
            "overestimate_cost": round(cost_over_total, 2),
            "underestimate_cost": round(cost_under_total, 2),
            "total_cases_evaluated": len(y_true),
        }

        logger.info(f"Financial Impact: ${total_loss:,.2f} total loss (${avg_loss_per_case:.2f}/case)")
        return report

    @staticmethod
    def estimate_cloud_inference_cost(
        n_requests_per_day: int,
        avg_latency_ms: float = 25.0,
        hourly_instance_rate: float = 0.08,
        concurrency: int = 1,
    ) -> Dict[str, Any]:
        """Calculates monthly cloud compute hosting costs based on inference throughput."""
        monthly_hours = 24 * 30.5
        total_monthly_compute = monthly_hours * hourly_instance_rate * concurrency
        total_monthly_requests = n_requests_per_day * 30.5
        cost_per_million_requests = (
            (total_monthly_compute / total_monthly_requests) * 1_000_000 if total_monthly_requests > 0 else 0.0
        )
        return {
            "monthly_infrastructure_cost_usd": round(total_monthly_compute, 2),
            "estimated_monthly_requests": int(total_monthly_requests),
            "cost_per_1m_requests_usd": round(cost_per_million_requests, 4),
            "instance_concurrency": concurrency,
        }


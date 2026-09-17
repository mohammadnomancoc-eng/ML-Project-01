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

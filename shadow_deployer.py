"""Shadow Deployment and Challenger Comparison Module."""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from logger import get_logger

logger = get_logger("ShadowDeployer")


class ShadowDeployer:
    """Routes live traffic to champion model while scoring asynchronously on challenger."""

    def __init__(self, champion_predictor, challenger_predictor):
        self.champion = champion_predictor
        self.challenger = challenger_predictor
        self.prediction_logs = []

    def predict(self, sample: Dict[str, Any]) -> Tuple[float, float]:
        """Returns champion prediction for user while recording challenger divergence."""
        champ_pred = self.champion.predict_single(sample)

        try:
            challenger_pred = self.challenger.predict_single(sample)
        except Exception as e:
            logger.warning(f"Challenger failed to predict: {e}")
            challenger_pred = None

        divergence = abs(champ_pred - challenger_pred) if challenger_pred is not None else None
        self.prediction_logs.append({
            "champion": champ_pred,
            "challenger": challenger_pred,
            "divergence": divergence,
        })

        return champ_pred, challenger_pred

    def get_shadow_report(self) -> Dict[str, Any]:
        """Calculates divergence statistics between champion and shadow model."""
        valid_divs = [log["divergence"] for log in self.prediction_logs if log["divergence"] is not None]
        return {
            "total_requests": len(self.prediction_logs),
            "mean_divergence": round(float(np.mean(valid_divs)), 4) if valid_divs else 0.0,
            "max_divergence": round(float(np.max(valid_divs)), 4) if valid_divs else 0.0,
        }

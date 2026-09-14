"""Ensemble Modeling: Voting & Stacking Regressors."""

import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from logger import get_logger

logger = get_logger("Ensemble")


class EnsembleBuilder:
    """Builds advanced ensemble meta-models (Voting and Stacking)."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.base_estimators = [
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=random_state)),
            ("gb", GradientBoostingRegressor(n_estimators=100, random_state=random_state)),
        ]

    def build_voting_regressor(self, weights: List[float] = [1.0, 2.0, 2.0]) -> VotingRegressor:
        """Constructs a weighted VotingRegressor ensemble."""
        logger.info(f"Building VotingRegressor with weights: {weights}")
        return VotingRegressor(estimators=self.base_estimators, weights=weights)

    def build_stacking_regressor(self) -> StackingRegressor:
        """Constructs a StackingRegressor using Ridge as final meta-estimator."""
        logger.info("Building StackingRegressor with Ridge final estimator...")
        final_estimator = Ridge(alpha=0.5, random_state=self.random_state)
        return StackingRegressor(
            estimators=self.base_estimators,
            final_estimator=final_estimator,
            cv=5,
            n_jobs=-1,
        )

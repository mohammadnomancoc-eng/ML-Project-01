"""Sequential Model-Based Optimization (SMBO) and Bayesian Hyperparameter Tuner for ML-Project-01.

Implements Gaussian Process surrogate model with Expected Improvement (EI) acquisition function
for efficient black-box hyperparameter optimization.
"""

from typing import Callable, Dict, List, Tuple, Any
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from logger import logger


class BayesianHyperparameterOptimizer:
    """Surrogate-based Bayesian optimization for continuous and discrete hyperparameter spaces."""

    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial_points: int = 5,
        n_iterations: int = 15,
        xi: float = 0.01,
        random_state: int = 42,
    ):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_initial_points = n_initial_points
        self.n_iterations = n_iterations
        self.xi = xi
        self.random_state = random_state
        self.gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5), alpha=1e-6, normalize_y=True, random_state=random_state
        )
        self.X_history: List[List[float]] = []
        self.y_history: List[float] = []

    def _expected_improvement(self, X_candidates: np.ndarray) -> np.ndarray:
        """Calculates Expected Improvement (EI) over current best value."""
        mu, sigma = self.gpr.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)
        current_best = np.max(self.y_history)

        improvement = mu - current_best - self.xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        return ei

    def optimize(self, objective_func: Callable[[Dict[str, float]], float]) -> Dict[str, Any]:
        """Runs the Bayesian optimization loop to maximize objective_func."""
        np.random.seed(self.random_state)
        dim = len(self.param_names)

        # 1. Sample initial random points (Latin Hypercube / Uniform)
        for _ in range(self.n_initial_points):
            pt = [
                np.random.uniform(self.param_bounds[name][0], self.param_bounds[name][1])
                for name in self.param_names
            ]
            score = objective_func(dict(zip(self.param_names, pt)))
            self.X_history.append(pt)
            self.y_history.append(score)

        # 2. Sequential SMBO loop
        for iter_idx in range(self.n_iterations):
            X_arr = np.array(self.X_history)
            y_arr = np.array(self.y_history)
            self.gpr.fit(X_arr, y_arr)

            # Sample candidate points to evaluate acquisition
            candidates = np.column_stack([
                np.random.uniform(self.param_bounds[name][0], self.param_bounds[name][1], 1000)
                for name in self.param_names
            ])
            ei = self._expected_improvement(candidates)
            next_pt = candidates[np.argmax(ei)]

            next_score = objective_func(dict(zip(self.param_names, next_pt)))
            self.X_history.append(next_pt.tolist())
            self.y_history.append(next_score)

        best_idx = np.argmax(self.y_history)
        best_params = dict(zip(self.param_names, self.X_history[best_idx]))
        best_score = self.y_history[best_idx]

        logger.info(f"Bayesian optimization completed. Best Score: {best_score:.4f} with params {best_params}")
        return {
            "best_params": best_params,
            "best_score": round(float(best_score), 4),
            "n_iterations": len(self.y_history),
        }

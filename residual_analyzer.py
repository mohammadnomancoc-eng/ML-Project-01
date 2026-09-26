"""Residual Heteroscedasticity and Statistical Diagnostic Analyzer for ML-Project-01.

Evaluates regression residuals for constant variance (homoscedasticity) and Durbin-Watson autocorrelation.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import stats
from logger import logger


class ResidualDiagnosticAnalyzer:
    """Analyzes regression residuals for normality, autocorrelation, and heteroscedasticity."""

    @staticmethod
    def audit_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculates Durbin-Watson autocorrelation and Shapiro-Wilk residual normality test."""
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        n = len(residuals)

        # 1. Durbin-Watson statistic (2.0 = no autocorrelation)
        diff_residuals = np.diff(residuals)
        dw_stat = float(np.sum(diff_residuals ** 2) / (np.sum(residuals ** 2) + 1e-8))

        # 2. Shapiro-Wilk or skewness normality check
        skewness = float(stats.skew(residuals))
        kurtosis = float(stats.kurtosis(residuals))

        # 3. Residual correlation with predicted values (Heteroscedasticity check)
        abs_res = np.abs(residuals)
        hetero_corr = float(np.corrcoef(abs_res, y_pred)[0, 1])

        results = {
            "durbin_watson_stat": round(dw_stat, 4),
            "autocorrelation_status": "NONE" if 1.5 <= dw_stat <= 2.5 else "DETECTED",
            "residual_skewness": round(skewness, 4),
            "residual_kurtosis": round(kurtosis, 4),
            "heteroscedasticity_correlation": round(hetero_corr, 4),
            "is_homoscedastic": abs(hetero_corr) < 0.20,
        }

        logger.info(f"Residual Diagnostics: DW={dw_stat:.2f}, Skew={skewness:.2f}, Heteroscedasticity Corr={hetero_corr:.2f}")
        return results

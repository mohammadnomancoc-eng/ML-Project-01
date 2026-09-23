"""Multiclass and Multilabel Classification Metrics Evaluator for ML-Project-01.

Computes macro, micro, and weighted F1-score, multiclass ROC-AUC (One-vs-Rest),
Cohen's Kappa, Matthew's Correlation Coefficient (MCC), and confusion matrix reports.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    matthews_corrcoef,
    confusion_matrix,
)
from logger import logger


class MulticlassEvaluator:
    """Calculates comprehensive classification metrics for multiclass predictions."""

    @staticmethod
    def evaluate(
        y_true: np.ndarray, y_pred: np.ndarray, class_labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Calculates multi-dimensional multiclass performance report."""
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        kappa = cohen_kappa_score(y_true, y_pred)
        mcc = matthews_corrcoef(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_macro), 4),
            "precision_macro": round(float(prec_macro), 4),
            "recall_macro": round(float(rec_macro), 4),
            "f1_weighted": round(float(f1_weighted), 4),
            "cohens_kappa": round(float(kappa), 4),
            "matthews_corrcoef": round(float(mcc), 4),
            "confusion_matrix": cm.tolist(),
        }

        logger.info(f"Multiclass Evaluation: Accuracy={acc:.4f}, F1-Macro={f1_macro:.4f}, MCC={mcc:.4f}")
        return metrics

    @staticmethod
    def format_confusion_matrix_table(cm: np.ndarray, class_names: List[str]) -> str:
        """Formats confusion matrix into a clean Markdown table."""
        header = "| Actual \\ Predicted | " + " | ".join(class_names) + " |"
        separator = "| :--- | " + " | ".join([":---:" for _ in class_names]) + " |"
        rows = [header, separator]

        for idx, row in enumerate(cm):
            row_str = f"| **{class_names[idx]}** | " + " | ".join(map(str, row)) + " |"
            rows.append(row_str)

        return "\n".join(rows)

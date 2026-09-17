"""Model Knowledge Distillation and Teacher-Student Compression."""

import pandas as pd
from typing import Any, Dict
from sklearn.linear_model import Ridge
from evaluator import ModelEvaluator
from logger import get_logger

logger = get_logger("Distiller")


class KnowledgeDistiller:
    """Compresses large teacher ensembles into ultra-fast student models using soft pseudo-labels."""

    def __init__(self, teacher_model: Any, student_model: Any = None):
        self.teacher = teacher_model
        self.student = student_model or Ridge(alpha=1.0)

    def distill(self, X_train: pd.DataFrame, y_true: pd.Series, alpha: float = 0.5) -> Any:
        """Trains student on a combination of ground truth and soft teacher predictions."""
        logger.info("Generating soft teacher targets for knowledge distillation...")
        teacher_preds = self.teacher.predict(X_train)

        # Blend teacher predictions with ground truth
        blended_targets = alpha * teacher_preds + (1 - alpha) * y_true.values

        logger.info(f"Training student model ({self.student.__class__.__name__}) on blended targets...")
        self.student.fit(X_train, blended_targets)

        student_preds = self.student.predict(X_train)
        metrics = ModelEvaluator.evaluate_model(y_true, student_preds, "DistilledStudent")
        logger.info(f"Distilled Student performance: R2 = {metrics['r2_score']}")
        return self.student

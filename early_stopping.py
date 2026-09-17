"""Early Stopping Mechanism for Iterative Training."""

from typing import Optional
from logger import get_logger

logger = get_logger("EarlyStopping")


class EarlyStopping:
    """Monitors validation loss/metric to halt training when performance plateaus."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False

    def step(self, current_score: float) -> bool:
        """Evaluates current epoch score and returns True if training should stop."""
        if self.best_score is None:
            self.best_score = current_score
            return False

        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
            logger.info(f"Score improved to {current_score:.4f}. Resetting patience counter.")
        else:
            self.counter += 1
            logger.info(f"Patience counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f})")
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered. Halting training.")

        return self.early_stop

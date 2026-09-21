"""Early Stopping Mechanism for Iterative Training."""

from typing import Optional
from logger import get_logger

logger = get_logger("EarlyStopping")


class EarlyStopping:
    """Monitors validation loss/metric to halt training when performance plateaus."""

    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 1e-4,
        mode: str = "min",
        min_epochs: int = 0,
        relative_delta: bool = False,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.min_epochs = min_epochs
        self.relative_delta = relative_delta
        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.current_epoch: int = 0
        self.early_stop: bool = False

    def step(self, current_score: float) -> bool:
        """Evaluates current epoch score and returns True if training should stop."""
        self.current_epoch += 1

        if self.best_score is None:
            self.best_score = current_score
            return False

        threshold = self.min_delta
        if self.relative_delta and abs(self.best_score) > 1e-8:
            threshold = abs(self.best_score) * self.min_delta

        if self.mode == "min":
            improved = current_score < (self.best_score - threshold)
        else:
            improved = current_score > (self.best_score + threshold)

        if improved:
            self.best_score = current_score
            self.counter = 0
            logger.info(f"Score improved to {current_score:.4f}. Resetting patience counter.")
        else:
            if self.current_epoch >= self.min_epochs:
                self.counter += 1
                logger.info(f"Patience counter: {self.counter}/{self.patience} (Best: {self.best_score:.4f})")
                if self.counter >= self.patience:
                    self.early_stop = True
                    logger.info("Early stopping triggered. Halting training.")
            else:
                logger.info(f"Warmup epoch {self.current_epoch}/{self.min_epochs}: skipping patience increment.")

        return self.early_stop

    def get_summary(self) -> dict:
        """Returns current early stopping state."""
        return {
            "best_score": self.best_score,
            "current_epoch": self.current_epoch,
            "counter": self.counter,
            "early_stop_triggered": self.early_stop,
        }


"""Learning Rate Schedulers for Optimization Algorithms."""

import math
from typing import List


class LRScheduler:
    """Calculates learning rates across training steps and epochs."""

    @staticmethod
    def step_decay(initial_lr: float, epoch: int, drop_rate: float = 0.5, epochs_drop: int = 10) -> float:
        """Step decay: drops learning rate by drop_rate every epochs_drop."""
        return initial_lr * math.pow(drop_rate, math.floor((1 + epoch) / epochs_drop))

    @staticmethod
    def exponential_decay(initial_lr: float, epoch: int, decay_rate: float = 0.05) -> float:
        """Exponential decay: smoothly decreases lr over epochs."""
        return initial_lr * math.exp(-decay_rate * epoch)

    @staticmethod
    def cosine_annealing(initial_lr: float, epoch: int, total_epochs: int, min_lr: float = 1e-6) -> float:
        """Cosine annealing: follows a cosine curve down to min_lr."""
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * epoch / total_epochs))

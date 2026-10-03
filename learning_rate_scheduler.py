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

    @staticmethod
    def cosine_annealing_warm_restarts(
        initial_lr: float, epoch: int, t_0: int = 10, t_mult: int = 2, min_lr: float = 1e-6
    ) -> float:
        """Cosine annealing with stochastic warm restarts (SGDR)."""
        t_curr = epoch
        t_i = t_0
        while t_curr >= t_i:
            t_curr -= t_i
            t_i *= t_mult
        return min_lr + 0.5 * (initial_lr - min_lr) * (1 + math.cos(math.pi * t_curr / t_i))

    @staticmethod
    def polynomial_decay(initial_lr: float, epoch: int, total_epochs: int, power: float = 1.0, min_lr: float = 1e-6) -> float:
        """Polynomial decay: decays learning rate with power polynomial curve."""
        decay_factor = (1.0 - (epoch / max(1, total_epochs))) ** power
        return (initial_lr - min_lr) * max(0.0, decay_factor) + min_lr

    @staticmethod
    def one_cycle_lr(max_lr: float, epoch: int, total_epochs: int, pct_start: float = 0.3, div_factor: float = 25.0) -> float:
        """One-cycle policy: warms up to max_lr and cosine anneals to min_lr."""
        initial_lr = max_lr / div_factor
        min_lr = initial_lr / 1000.0
        warmup_epochs = int(total_epochs * pct_start)
        if epoch <= warmup_epochs:
            # Linear warmup
            pct = epoch / max(1, warmup_epochs)
            return initial_lr + (max_lr - initial_lr) * pct
        else:
            # Cosine decay
            decay_epochs = total_epochs - warmup_epochs
            curr_epoch = epoch - warmup_epochs
            return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * curr_epoch / max(1, decay_epochs)))

    @classmethod
    def generate_schedule(cls, scheduler_type: str, initial_lr: float, total_epochs: int, **kwargs) -> List[float]:
        """Generate full learning rate trajectory list for the entire training cycle."""
        schedule = []
        for epoch in range(total_epochs):
            if scheduler_type == "step":
                lr = cls.step_decay(initial_lr, epoch, **kwargs)
            elif scheduler_type == "exponential":
                lr = cls.exponential_decay(initial_lr, epoch, **kwargs)
            elif scheduler_type == "cosine":
                lr = cls.cosine_annealing(initial_lr, epoch, total_epochs, **kwargs)
            elif scheduler_type == "warm_restart":
                lr = cls.cosine_annealing_warm_restarts(initial_lr, epoch, **kwargs)
            elif scheduler_type == "polynomial":
                lr = cls.polynomial_decay(initial_lr, epoch, total_epochs, **kwargs)
            elif scheduler_type == "one_cycle":
                lr = cls.one_cycle_lr(initial_lr, epoch, total_epochs, **kwargs)
            else:
                lr = initial_lr
            schedule.append(lr)
        return schedule



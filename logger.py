"""Structured Logging for Model Training and Pipeline Execution."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config import paths


def get_logger(name: str = "ML_Pipeline") -> logging.Logger:
    """Configures and returns a structured logger instance."""
    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    log_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    # File Handler
    today_str = datetime.now().strftime("%Y-%m-%d")
    log_file = paths.LOGS_DIR / f"run_{today_str}.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    return logger


logger = get_logger()


def log_execution_time(func):
    """Decorator to measure and log function execution duration."""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"Function '{func.__name__}' executed in {elapsed:.4f}s")
        return res

    return wrapper

"""Centralized Configuration for ML-Project-01."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


@dataclass
class PathConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    OUTPUT_DIR: Path = BASE_DIR / "outputs"
    LOGS_DIR: Path = BASE_DIR / "logs"

    def __post_init__(self):
        for directory in [self.DATA_DIR, self.MODELS_DIR, self.OUTPUT_DIR, self.LOGS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    RANDOM_STATE: int = int(os.getenv("ML_RANDOM_STATE", "42"))
    TEST_SIZE: float = float(os.getenv("ML_TEST_SIZE", "0.2"))
    VAL_SIZE: float = float(os.getenv("ML_VAL_SIZE", "0.1"))
    N_SAMPLES: int = int(os.getenv("ML_N_SAMPLES", "2500"))
    N_FEATURES: int = int(os.getenv("ML_N_FEATURES", "8"))
    TARGET_COLUMN: str = os.getenv("ML_TARGET_COL", "target")
    SCALING_METHOD: str = os.getenv("ML_SCALING_METHOD", "standard")  # 'standard', 'minmax', 'robust'

    ALGORITHMS: List[str] = field(
        default_factory=lambda: [
            "linear_regression",
            "ridge",
            "random_forest",
            "gradient_boosting",
        ]
    )

    PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = field(
        default_factory=lambda: {
            "ridge": {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]},
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
            },
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "RANDOM_STATE": self.RANDOM_STATE,
            "TEST_SIZE": self.TEST_SIZE,
            "VAL_SIZE": self.VAL_SIZE,
            "N_SAMPLES": self.N_SAMPLES,
            "N_FEATURES": self.N_FEATURES,
            "TARGET_COLUMN": self.TARGET_COLUMN,
            "SCALING_METHOD": self.SCALING_METHOD,
            "ALGORITHMS": self.ALGORITHMS,
        }

    def save_json(self, file_path: Path) -> None:
        """Export configuration parameters to a JSON file."""
        import json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)


paths = PathConfig()
config = ModelConfig()


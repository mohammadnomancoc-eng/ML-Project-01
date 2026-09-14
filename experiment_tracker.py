"""Lightweight Local Experiment Tracking & Model Registry."""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from config import paths
from logger import get_logger

logger = get_logger("ExperimentTracker")


class ExperimentTracker:
    """Logs training runs, parameters, metrics, and artifact references."""

    def __init__(self, registry_file: str = "experiment_runs.json"):
        self.registry_path = paths.OUTPUT_DIR / registry_file
        self.runs: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return []
        return []

    def _save(self) -> None:
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(self.runs, f, indent=2)

    def log_run(
        self,
        model_name: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Dict[str, str],
    ) -> str:
        """Logs a single experiment execution run."""
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_record = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "metrics": metrics,
            "artifacts": artifacts,
        }

        self.runs.append(run_record)
        self._save()
        logger.info(f"Experiment logged successfully: [{run_id}] | R2: {metrics.get('r2_score')}")
        return run_id

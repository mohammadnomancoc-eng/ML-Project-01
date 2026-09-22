"""Model Registry and Lifecycle Stage Management for ML-Project-01.

Provides semantic versioning, stage promotion (Development, Staging, Production, Archived),
lineage metadata tracking, and artifact loading.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from config import paths
from logger import logger
from model_serializer import ModelSerializer


class ModelRegistry:
    """Manages model artifacts, semantic versions, and lifecycle staging states."""

    REGISTRY_FILE = paths.MODELS_DIR / "registry_manifest.json"

    def __init__(self):
        self._ensure_manifest()

    def _ensure_manifest(self) -> None:
        """Initializes empty registry manifest file if not existing."""
        if not self.REGISTRY_FILE.exists():
            initial_data = {"models": {}, "created_at": datetime.now().isoformat()}
            with open(self.REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=4)

    def _load_manifest(self) -> Dict[str, Any]:
        """Loads the registry JSON metadata manifest."""
        with open(self.REGISTRY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        """Saves updated registry manifest."""
        manifest["updated_at"] = datetime.now().isoformat()
        with open(self.REGISTRY_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)

    def register_model(
        self,
        model_name: str,
        artifact_filename: str,
        version: str = "v1.0.0",
        stage: str = "Development",
        metrics: Optional[Dict[str, float]] = None,
        description: str = "",
    ) -> Dict[str, Any]:
        """Registers a new model version in the registry."""
        manifest = self._load_manifest()

        if model_name not in manifest["models"]:
            manifest["models"][model_name] = []

        entry = {
            "version": version,
            "stage": stage,
            "artifact_filename": artifact_filename,
            "metrics": metrics or {},
            "description": description,
            "registered_at": datetime.now().isoformat(),
        }

        manifest["models"][model_name].append(entry)
        self._save_manifest(manifest)
        logger.info(f"Model '{model_name}' version {version} registered with stage '{stage}'.")
        return entry

    def promote_stage(self, model_name: str, version: str, new_stage: str) -> bool:
        """Promotes a registered model version to a new lifecycle stage (e.g. Production)."""
        valid_stages = ["Development", "Staging", "Production", "Archived"]
        if new_stage not in valid_stages:
            raise ValueError(f"Invalid stage '{new_stage}'. Must be one of {valid_stages}")

        manifest = self._load_manifest()
        if model_name not in manifest["models"]:
            return False

        updated = False
        for entry in manifest["models"][model_name]:
            if new_stage == "Production" and entry.get("stage") == "Production":
                entry["stage"] = "Archived"  # Demote previous prod model

            if entry["version"] == version:
                entry["stage"] = new_stage
                entry["stage_updated_at"] = datetime.now().isoformat()
                updated = True

        if updated:
            self._save_manifest(manifest)
            logger.info(f"Promoted {model_name}:{version} to stage '{new_stage}'.")
        return updated

    def get_production_model(self, model_name: str) -> Optional[Any]:
        """Loads and returns the current Production model object for the specified model name."""
        manifest = self._load_manifest()
        if model_name not in manifest["models"]:
            return None

        for entry in manifest["models"][model_name]:
            if entry.get("stage") == "Production":
                return ModelSerializer.load_artifact(entry["artifact_filename"])
        return None

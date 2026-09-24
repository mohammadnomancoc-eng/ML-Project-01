"""Model Artifact Persistence and Deserialization Module."""

import json
import joblib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
from config import paths
from logger import logger


class ModelSerializer:
    """Saves and loads trained models, transformers, and training metadata."""

    @staticmethod
    def save_artifact(obj: Any, filename: str, metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Saves a Python object using joblib alongside metadata JSON."""
        file_path = paths.MODELS_DIR / filename
        joblib.dump(obj, file_path)
        logger.info(f"Saved binary artifact to: {file_path}")

        if metadata:
            meta_path = file_path.with_suffix(".json")
            metadata["saved_at"] = datetime.now().isoformat()
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info(f"Saved artifact metadata to: {meta_path}")

        return file_path

    @staticmethod
    def load_artifact(filename: str) -> Any:
        """Loads a persisted binary artifact."""
        file_path = paths.MODELS_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")
        logger.info(f"Loading artifact from: {file_path}")
        return joblib.load(file_path)

    @staticmethod
    def compute_checksum(filename: str) -> str:
        """Computes SHA-256 hash of a saved artifact."""
        import hashlib

        file_path = paths.MODELS_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Artifact not found at {file_path}")

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    @staticmethod
    def verify_checksum(filename: str, expected_hash: str) -> bool:
        """Verifies integrity of artifact against expected SHA-256 hash."""
        actual_hash = ModelSerializer.compute_checksum(filename)
        is_match = actual_hash.lower() == expected_hash.lower()
        if not is_match:
            logger.warning(f"Integrity check failed for {filename}! Expected: {expected_hash}, Got: {actual_hash}")
        return is_match

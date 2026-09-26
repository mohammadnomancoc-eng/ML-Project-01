"""Pipeline Intermediate State Checkpointing for ML-Project-01.

Saves and resumes intermediate transformed DataFrames and states to avoid recomputing expensive pipeline stages.
"""

from pathlib import Path
from typing import Optional, Any
import pandas as pd
from config import paths
from logger import logger


class PipelineCheckpointManager:
    """Manages serializing intermediate feature DataFrames and stages to disk."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or (paths.DATA_DIR / "checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(self, df: pd.DataFrame, stage_name: str) -> Path:
        """Saves a DataFrame snapshot for a specific pipeline stage."""
        file_path = self.checkpoint_dir / f"checkpoint_{stage_name}.parquet"
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved stage checkpoint '{stage_name}' ({len(df)} rows) to: {file_path}")
        return file_path

    def load_checkpoint(self, stage_name: str) -> Optional[pd.DataFrame]:
        """Loads a cached stage DataFrame if it exists."""
        file_path = self.checkpoint_dir / f"checkpoint_{stage_name}.parquet"
        if file_path.exists():
            logger.info(f"Loaded checkpoint for '{stage_name}' from: {file_path}")
            return pd.read_parquet(file_path)
        return None

    def has_checkpoint(self, stage_name: str) -> bool:
        """Returns True if stage checkpoint exists on disk."""
        return (self.checkpoint_dir / f"checkpoint_{stage_name}.parquet").exists()

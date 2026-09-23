"""DAG-Based Artifact Lineage and Data Provenance Tracker for ML-Project-01.

Captures end-to-end cryptographic checksums, pipeline graph dependencies, and training provenance metadata.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from config import paths
from logger import logger


class LineageTracker:
    """Records directed acyclic graph (DAG) metadata of pipeline executions."""

    LINEAGE_FILE = paths.MODELS_DIR / "lineage_provenance.json"

    def __init__(self):
        self.nodes: List[Dict[str, Any]] = []
        self.edges: List[Dict[str, str]] = []
        self.run_id: str = datetime.now().strftime("run_%Y%m%d_%H%M%S")

    @staticmethod
    def compute_sha256(file_path: Path) -> str:
        """Computes SHA-256 hash of a file for cryptographic integrity."""
        if not file_path.exists():
            return "FILE_NOT_FOUND"
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def record_node(
        self,
        node_id: str,
        node_type: str,  # 'dataset', 'transform', 'model', 'metric'
        metadata: Optional[Dict[str, Any]] = None,
        artifact_path: Optional[Path] = None,
    ) -> None:
        """Adds a pipeline artifact node to the lineage graph."""
        node = {
            "id": node_id,
            "type": node_type,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "sha256": self.compute_sha256(artifact_path) if artifact_path else None,
        }
        self.nodes.append(node)

    def record_edge(self, source_id: str, target_id: str, relationship: str = "produced_by") -> None:
        """Adds a directional dependency edge between two pipeline nodes."""
        self.edges.append({
            "from": source_id,
            "to": target_id,
            "relation": relationship,
        })

    def export_lineage_manifest(self) -> Dict[str, Any]:
        """Saves lineage DAG provenance graph to JSON."""
        manifest = {
            "run_id": self.run_id,
            "generated_at": datetime.now().isoformat(),
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": self.nodes,
            "edges": self.edges,
        }
        with open(self.LINEAGE_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Lineage provenance graph saved with {len(self.nodes)} nodes to {self.LINEAGE_FILE}.")
        return manifest

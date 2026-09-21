"""Automated Model Card Generator (Markdown Documentation)."""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from config import paths
from logger import get_logger

logger = get_logger("ModelCard")


class ModelCardGenerator:
    """Generates standardized Model Cards in Markdown format."""

    @staticmethod
    def generate_card(
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        features: List[str],
        output_filename: str = "MODEL_CARD.md",
    ) -> Path:
        """Constructs and saves a model card markdown document."""
        logger.info(f"Generating Model Card for [{model_name}] v{version}...")

        metrics_table = "\n".join([f"| **{k.upper()}** | `{v}` |" for k, v in metrics.items()])
        features_list = "\n".join([f"- `{feat}`" for feat in features])

        content = f"""# Model Card: {model_name}

**Version:** {version}  
**Date Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**License:** MIT  

---

## 📌 Model Overview
- **Model Architecture:** `{model_name}`
- **Intended Use:** High-throughput numerical regression & predictive analytics.
- **Input Features ({len(features)}):**
{features_list}

---

## 📊 Performance Benchmarks
| Metric | Value |
|---|---|
{metrics_table}

---

## ⚙️ Hyperparameters
```json
{hyperparameters}
```

---

## ⚠️ Limitations & Ethical Considerations
- Designed for continuous numerical target variables.
- Models should be retrained periodically if statistical data drift occurs.
"""
        output_path = paths.OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        logger.info(f"Model Card written to: {output_path}")
        return output_path

    @staticmethod
    def generate_json_manifest(
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        output_filename: str = "model_manifest.json",
    ) -> Path:
        """Exports model metadata manifest in structured JSON format."""
        import json

        manifest = {
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "metrics": metrics,
            "hyperparameters": hyperparameters,
            "status": "validated",
        }
        output_path = paths.OUTPUT_DIR / output_filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=4)
        logger.info(f"Model manifest saved to {output_path}")
        return output_path


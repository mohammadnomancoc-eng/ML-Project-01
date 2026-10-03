"""Lightweight JSON and Pure Python Model Rule Exporter for ML-Project-01.

Extracts linear coefficients, intercepts, and decision tree thresholds into dependency-free JSON and Python code.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from logger import logger


class ModelRuleExporter:
    """Exports model equations and weights to plain JSON and portable Python snippets."""

    @staticmethod
    def export_linear_model(model: Any, feature_names: List[str], model_name: str = "LinearModel") -> Dict[str, Any]:
        """Extracts linear equation weights and intercept into structured JSON representation."""
        if not hasattr(model, "coef_"):
            raise ValueError("Model does not contain linear coefficients ('coef_').")

        coefs = {feat: round(float(c), 6) for feat, c in zip(feature_names, model.coef_)}
        intercept = round(float(model.intercept_), 6)

        formula_terms = [f"({val} * {feat})" for feat, val in coefs.items()]
        equation_str = f"prediction = {intercept} + " + " + ".join(formula_terms)

        export_data = {
            "model_name": model_name,
            "intercept": intercept,
            "coefficients": coefs,
            "equation_formula": equation_str,
        }

        logger.info(f"Exported {model_name} linear rules with {len(coefs)} features.")
        return export_data

    @staticmethod
    def export_decision_tree_rules(tree_model: Any, feature_names: List[str]) -> List[Dict[str, Any]]:
        """Extracts decision tree node split rules and threshold values into structured list."""
        if not hasattr(tree_model, "tree_"):
            raise ValueError("Model does not contain decision tree structure ('tree_').")
        
        tree = tree_model.tree_
        nodes = []
        for i in range(tree.node_count):
            if tree.feature[i] >= 0:
                nodes.append({
                    "node_id": i,
                    "type": "split",
                    "feature": feature_names[tree.feature[i]] if tree.feature[i] < len(feature_names) else f"feature_{tree.feature[i]}",
                    "threshold": round(float(tree.threshold[i]), 6),
                    "left_child": int(tree.children_left[i]),
                    "right_child": int(tree.children_right[i]),
                })
            else:
                nodes.append({
                    "node_id": i,
                    "type": "leaf",
                    "value": round(float(tree.value[i][0][0]), 6),
                })
        return nodes


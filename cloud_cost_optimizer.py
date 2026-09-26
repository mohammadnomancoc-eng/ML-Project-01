"""Cloud Instance Sizing and Inference Cost Optimizer for ML-Project-01.

Evaluates latency requirements and throughput targets to select the most cost-efficient cloud hosting instance.
"""

from typing import Dict, List, Any
from logger import logger


class CloudCostOptimizer:
    """Selects optimal cloud instance type based on latency SLAs and monthly request volume."""

    INSTANCE_CATALOG = [
        {"name": "t4g.small", "vcpus": 2, "ram_gb": 2.0, "hourly_cost_usd": 0.0168, "max_qps": 25},
        {"name": "c6g.medium", "vcpus": 1, "ram_gb": 2.0, "hourly_cost_usd": 0.0340, "max_qps": 60},
        {"name": "c6g.xlarge", "vcpus": 4, "ram_gb": 8.0, "hourly_cost_usd": 0.1360, "max_qps": 240},
        {"name": "g4dn.xlarge (GPU)", "vcpus": 4, "ram_gb": 16.0, "hourly_cost_usd": 0.5260, "max_qps": 850},
    ]

    @classmethod
    def recommend_instance(cls, peak_qps_target: float) -> Dict[str, Any]:
        """Recommends minimal cost instance capable of sustaining peak_qps_target."""
        for inst in cls.INSTANCE_CATALOG:
            if inst["max_qps"] >= peak_qps_target:
                monthly_cost = round(inst["hourly_cost_usd"] * 24 * 30.5, 2)
                logger.info(f"Recommended cloud instance: {inst['name']} (${monthly_cost}/month for {peak_qps_target} QPS)")
                return {
                    "recommended_instance": inst["name"],
                    "vcpus": inst["vcpus"],
                    "ram_gb": inst["ram_gb"],
                    "hourly_cost_usd": inst["hourly_cost_usd"],
                    "estimated_monthly_cost_usd": monthly_cost,
                    "max_sustainable_qps": inst["max_qps"],
                    "headroom_percent": round(((inst["max_qps"] - peak_qps_target) / inst["max_qps"]) * 100, 1),
                }

        # Highest tier fallback
        inst = cls.INSTANCE_CATALOG[-1]
        monthly_cost = round(inst["hourly_cost_usd"] * 24 * 30.5, 2)
        return {
            "recommended_instance": inst["name"],
            "vcpus": inst["vcpus"],
            "ram_gb": inst["ram_gb"],
            "hourly_cost_usd": inst["hourly_cost_usd"],
            "estimated_monthly_cost_usd": monthly_cost,
            "max_sustainable_qps": inst["max_qps"],
            "headroom_percent": 0.0,
        }

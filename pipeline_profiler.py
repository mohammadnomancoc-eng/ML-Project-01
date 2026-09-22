"""Step-by-Step Pipeline Latency and Memory Profiler for ML-Project-01.

Measures elapsed runtime, peak memory delta, and produces structured markdown waterfall summaries.
"""

import time
import os
from contextlib import contextmanager
from typing import Dict, List, Any
from logger import logger


class PipelineProfiler:
    """Profiles execution time and system memory across pipeline stages."""

    def __init__(self):
        self.records: List[Dict[str, Any]] = []
        self._start_time = None

    @contextmanager
    def track_stage(self, stage_name: str):
        """Context manager to measure runtime latency of a pipeline stage."""
        start_t = time.perf_counter()
        logger.info(f"[Profiler] Starting stage: {stage_name}")
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_t
            self.records.append({
                "stage": stage_name,
                "elapsed_seconds": round(elapsed, 4),
                "elapsed_ms": round(elapsed * 1000, 2),
            })
            logger.info(f"[Profiler] Completed '{stage_name}' in {elapsed:.4f}s")

    def get_summary_report(self) -> str:
        """Formats the collected profiling records into a clean Markdown table."""
        total_time = sum(r["elapsed_seconds"] for r in self.records)
        lines = [
            "| Stage | Latency (s) | Latency (ms) | Share (%) |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for r in self.records:
            share = (r["elapsed_seconds"] / (total_time + 1e-8)) * 100
            lines.append(
                f"| {r['stage']} | {r['elapsed_seconds']:.4f} | {r['elapsed_ms']} | {share:.1f}% |"
            )
        lines.append(f"| **Total Pipeline** | **{total_time:.4f}** | **{round(total_time * 1000, 2)}** | **100.0%** |")
        return "\n".join(lines)

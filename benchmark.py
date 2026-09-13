"""Model Latency Profiling and Inference Benchmark Tool."""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any
from logger import logger


class ModelBenchmark:
    """Profiles inference execution speed, latency percentiles, and throughput."""

    @staticmethod
    def benchmark_latency(predictor_fn, sample_data, n_iterations: int = 500) -> Dict[str, Any]:
        """Calculates latency statistics across iterations."""
        logger.info(f"Running latency benchmark over {n_iterations} iterations...")
        latencies = []

        # Warmup
        for _ in range(25):
            predictor_fn(sample_data)

        # Measurement
        for _ in range(n_iterations):
            start = time.perf_counter()
            predictor_fn(sample_data)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)

        latencies = np.array(latencies)
        results = {
            "iterations": n_iterations,
            "mean_latency_ms": round(float(np.mean(latencies)), 3),
            "median_latency_ms": round(float(np.median(latencies)), 3),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 3),
            "p99_latency_ms": round(float(np.percentile(latencies, 99)), 3),
            "throughput_req_per_sec": round(float(1000.0 / np.mean(latencies)), 1),
        }

        logger.info(f"Benchmark: Mean {results['mean_latency_ms']}ms | P95 {results['p95_latency_ms']}ms | {results['throughput_req_per_sec']} req/sec")
        return results

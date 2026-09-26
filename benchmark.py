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

    @staticmethod
    def benchmark_batch_scalability(predictor_fn, sample_df: pd.DataFrame, batch_sizes: list = [1, 10, 50, 100, 500]) -> Dict[int, float]:
        """Measures throughput across varying inference batch sizes."""
        scalability_report = {}
        for b_size in batch_sizes:
            batch = pd.concat([sample_df] * int(np.ceil(b_size / len(sample_df))), ignore_index=True).iloc[:b_size]
            start = time.perf_counter()
            for _ in range(20):
                predictor_fn(batch)
            avg_batch_time_ms = ((time.perf_counter() - start) / 20.0) * 1000.0
            throughput = round((b_size / (avg_batch_time_ms / 1000.0)), 1) if avg_batch_time_ms > 0 else 0.0
            scalability_report[b_size] = throughput

        logger.info(f"Batch scalability throughput (req/s): {scalability_report}")
        return scalability_report

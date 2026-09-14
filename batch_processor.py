"""Large-Scale Batch Inference with Chunked Stream Processing."""

from pathlib import Path
from typing import Generator
import pandas as pd
from logger import get_logger

logger = get_logger("BatchProcessor")


class BatchProcessor:
    """Processes large CSV datasets in streaming chunks for inference and scoring."""

    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def process_file(
        self, input_file: Path, output_file: Path, predict_fn
    ) -> int:
        """Reads input CSV in chunks, applies predictor, and writes to output CSV."""
        logger.info(f"Starting batch stream processing of {input_file} (chunk size: {self.chunk_size})...")

        total_rows = 0
        first_chunk = True

        for chunk_idx, chunk_df in enumerate(pd.read_csv(input_file, chunksize=self.chunk_size)):
            predictions = predict_fn(chunk_df)
            chunk_df["prediction"] = predictions

            mode = "w" if first_chunk else "a"
            header = first_chunk
            chunk_df.to_csv(output_file, mode=mode, header=header, index=False)

            first_chunk = False
            total_rows += len(chunk_df)
            logger.info(f"Processed chunk {chunk_idx + 1} ({len(chunk_df)} rows) — Total: {total_rows}")

        logger.info(f"Batch processing completed: {total_rows} rows written to {output_file}")
        return total_rows

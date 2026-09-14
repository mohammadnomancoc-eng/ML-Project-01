"""Database Connection and SQL Data Ingestion Module."""

import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd
from logger import get_logger

logger = get_logger("DBConnector")


class DatabaseConnector:
    """Manages SQLite database connections and table query extraction."""

    def __init__(self, db_path: str = "data/ml_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def save_dataframe_to_table(self, df: pd.DataFrame, table_name: str, if_exists: str = "replace") -> int:
        """Saves a Pandas DataFrame into an SQL table."""
        logger.info(f"Writing {len(df)} rows into SQL table '{table_name}'...")
        with self._get_connection() as conn:
            df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        return len(df)

    def load_table_as_dataframe(self, table_name: str, limit: Optional[int] = None) -> pd.DataFrame:
        """Loads an SQL table as a Pandas DataFrame."""
        query = f"SELECT * FROM {table_name}"
        if limit:
            query += f" LIMIT {limit}"

        logger.info(f"Executing SQL query: {query}")
        with self._get_connection() as conn:
            return pd.read_sql_query(query, conn)

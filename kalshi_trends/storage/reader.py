"""Read raw and processed data from local storage."""

from pathlib import Path
from typing import Optional, Union

import pandas as pd


class StorageReader:
    """Read cached and processed data."""

    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.processed_dir = self.base_dir / "processed"
        self.cache_dir = self.base_dir / "cache"

    def list_raw_files(self, pattern: str = "*") -> list[Path]:
        """List raw data files matching pattern."""
        if not self.raw_dir.exists():
            return []
        return sorted(self.raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)

    def read_latest_markets(self) -> Optional[pd.DataFrame]:
        """Read most recent markets CSV/Parquet."""
        for ext in ["parquet", "csv"]:
            files = list(self.raw_dir.glob(f"markets_*.{ext}"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                if ext == "parquet":
                    return pd.read_parquet(latest)
                return pd.read_csv(latest)
        return None

    def read_latest_events(self) -> Optional[pd.DataFrame]:
        """Read most recent events CSV/Parquet."""
        for ext in ["parquet", "csv"]:
            files = list(self.raw_dir.glob(f"events_*.{ext}"))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                if ext == "parquet":
                    return pd.read_parquet(latest)
                return pd.read_csv(latest)
        return None

    def read_candlesticks(self, ticker: str) -> Optional[pd.DataFrame]:
        """Read candlestick file for a ticker."""
        safe = ticker.replace("/", "_").replace(":", "_")
        path = self.raw_dir / f"candlesticks_{safe}.csv"
        if path.exists():
            return pd.read_csv(path)
        return None

    def read_processed(self, name: str) -> Optional[pd.DataFrame]:
        """Read processed dataset by name."""
        for ext in ["parquet", "csv"]:
            path = self.processed_dir / f"{name}.{ext}"
            if path.exists():
                if ext == "parquet":
                    return pd.read_parquet(path)
                return pd.read_csv(path)
        return None

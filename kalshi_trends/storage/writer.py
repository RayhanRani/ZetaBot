"""Write raw and processed data to local storage."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class StorageWriter:
    """Write fetched and processed data to CSV, Parquet, and cache."""

    def __init__(self, base_dir: Union[str, Path] = "data"):
        self.base_dir = Path(base_dir)
        self.raw_dir = self.base_dir / "raw"
        self.cache_dir = self.base_dir / "cache"
        self.processed_dir = self.base_dir / "processed"
        _ensure_dir(self.raw_dir)
        _ensure_dir(self.cache_dir)
        _ensure_dir(self.processed_dir)

    def _cache_path(self, key: str, ext: str = "json") -> Path:
        return self.cache_dir / f"{key}.{ext}"

    def cache_response(self, key: str, data: Union[dict, list]) -> Path:
        """Cache API response as JSON for reuse."""
        path = self._cache_path(key)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.debug("Cached %s to %s", key, path)
        return path

    def load_cached(self, key: str) -> Optional[Union[dict, list]]:
        """Load cached response if exists."""
        path = self._cache_path(key)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    def write_markets(self, markets: list[dict], suffix: str = "") -> Path:
        """Write markets to CSV and Parquet."""
        df = pd.DataFrame(_flatten_markets(markets))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if suffix == "" else suffix
        csv_path = self.raw_dir / f"markets_{stamp}.csv"
        pq_path = self.raw_dir / f"markets_{stamp}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, index=False)
        logger.info("Wrote %d markets to %s", len(df), csv_path)
        return csv_path

    def write_events(self, events: list[dict], suffix: str = "") -> Path:
        """Write events (with flattened markets) to CSV and Parquet."""
        rows = _flatten_events(events)
        df = pd.DataFrame(rows)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S") if suffix == "" else suffix
        csv_path = self.raw_dir / f"events_{stamp}.csv"
        pq_path = self.raw_dir / f"events_{stamp}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, index=False)
        logger.info("Wrote %d event-markets to %s", len(df), csv_path)
        return csv_path

    def write_candlesticks(self, ticker: str, candlesticks: list[dict]) -> Path:
        """Write candlestick history for a market."""
        df = pd.DataFrame(_flatten_candlesticks(candlesticks))
        safe_ticker = ticker.replace("/", "_").replace(":", "_")
        csv_path = self.raw_dir / f"candlesticks_{safe_ticker}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path

    def write_processed(self, name: str, df: pd.DataFrame) -> Path:
        """Write processed/analysis output to data/processed/."""
        _ensure_dir(self.processed_dir)
        csv_path = self.processed_dir / f"{name}.csv"
        pq_path = self.processed_dir / f"{name}.parquet"
        df.to_csv(csv_path, index=False)
        df.to_parquet(pq_path, index=False)
        return csv_path


def _flatten_markets(markets: list[dict]) -> list[dict]:
    out = []
    for m in markets:
        row = {
            "ticker": m.get("ticker"),
            "event_ticker": m.get("event_ticker"),
            "title": m.get("title"),
            "subtitle": m.get("subtitle"),
            "status": m.get("status"),
            "last_price": m.get("last_price"),
            "last_price_dollars": m.get("last_price_dollars"),
            "yes_bid": m.get("yes_bid"),
            "yes_ask": m.get("yes_ask"),
            "no_bid": m.get("no_bid"),
            "no_ask": m.get("no_ask"),
            "volume": m.get("volume"),
            "volume_24h": m.get("volume_24h"),
            "open_interest": m.get("open_interest"),
            "liquidity": m.get("liquidity"),
            "open_time": m.get("open_time"),
            "close_time": m.get("close_time"),
            "expiration_time": m.get("expiration_time"),
            "created_time": m.get("created_time"),
            "updated_time": m.get("updated_time"),
        }
        out.append(row)
    return out


def _flatten_events(events: list[dict]) -> list[dict]:
    rows = []
    for ev in events:
        base = {
            "event_ticker": ev.get("event_ticker"),
            "event_title": ev.get("title"),
            "series_ticker": ev.get("series_ticker"),
            "sub_title": ev.get("sub_title"),
            "category": ev.get("category"),
            "mutually_exclusive": ev.get("mutually_exclusive"),
        }
        for m in ev.get("markets", []):
            flat = _flatten_markets([m])[0]
            if not flat.get("title"):
                flat["title"] = base["event_title"]
            row = {**base, **flat}
            rows.append(row)
    return rows


def _flatten_candlesticks(candlesticks: list[dict]) -> list[dict]:
    out = []
    for c in candlesticks:
        end_ts = c.get("end_period_ts")
        price = c.get("price") or {}
        vol = c.get("volume", 0)
        oi = c.get("open_interest", 0)
        out.append({
            "end_period_ts": end_ts,
            "open": price.get("open"),
            "high": price.get("high"),
            "low": price.get("low"),
            "close": price.get("close"),
            "mean": price.get("mean"),
            "volume": vol,
            "open_interest": oi,
        })
    return out

"""
KalshiData.com data adapter.

KalshiData.com displays Kalshi prediction market data. It uses Kalshi's public API
(api.elections.kalshi.com) as its backend. This adapter fetches from that same API
to obtain markets, events, and price history - the same data shown on kalshidata.com.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Any, Iterator, Optional

import httpx

logger = logging.getLogger(__name__)

# Kalshi public API - same backend used by kalshidata.com
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_RATE_LIMIT_DELAY = 1.0  # seconds between requests


def _retry_with_backoff(
    fn,
    max_retries: int = 3,
    base_delay: float = 2.0,
    backoff_factor: float = 2.0,
):
    """Execute fn with exponential backoff on failure."""
    last_exc = None
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return fn()
        except (httpx.HTTPError, httpx.TimeoutException) as e:
            last_exc = e
            if attempt < max_retries - 1:
                logger.warning(
                    "Request failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    max_retries,
                    e,
                    delay,
                )
                time.sleep(delay)
                delay *= backoff_factor
    raise last_exc


class KalshiDataClient:
    """
    Client for fetching Kalshi market data (same source as kalshidata.com).
    Uses Kalshi's public API - no authentication required for market data.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        rate_limit_delay: float = DEFAULT_RATE_LIMIT_DELAY,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self._last_request_time: float = 0

    def _throttle(self) -> None:
        """Enforce rate limit between requests."""
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET request with retry and rate limiting."""
        url = f"{self.base_url}{path}"

        def _do():
            self._throttle()
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
                return resp.json()

        return _retry_with_backoff(_do)

    def get_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
    ) -> dict:
        """Fetch markets list. Returns {markets, cursor}."""
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        return self._get("/markets", params)

    def get_events(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        with_nested_markets: bool = True,
        status: Optional[str] = None,
        series_ticker: Optional[str] = None,
    ) -> dict:
        """Fetch events (with nested markets). Returns {events, cursor, milestones}."""
        params: dict[str, Any] = {
            "limit": limit,
            "with_nested_markets": str(with_nested_markets).lower(),
        }
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if series_ticker:
            params["series_ticker"] = series_ticker
        return self._get("/events", params)

    def get_series_list(self, limit: int = 200, cursor: Optional[str] = None) -> dict:
        """Fetch series list. Returns {series, cursor}."""
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/series", params)

    def get_market_candlesticks(
        self,
        series_ticker: str,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict:
        """
        Fetch candlestick (OHLC) history for a market.
        period_interval: 1 (1min), 60 (1hr), 1440 (1day)
        """
        path = f"/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        return self._get(path, params)

    def get_event_candlesticks(
        self,
        event_ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 60,
    ) -> dict:
        """Fetch event-level candlesticks (aggregate across markets)."""
        path = f"/events/{event_ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        return self._get(path, params)

    def iter_all_events(
        self,
        status: Optional[str] = None,
        with_nested_markets: bool = True,
    ) -> Iterator[dict]:
        """Iterate over all events with pagination."""
        cursor = None
        while True:
            data = self.get_events(
                limit=200,
                cursor=cursor,
                with_nested_markets=with_nested_markets,
                status=status,
            )
            for ev in data.get("events", []):
                yield ev
            cursor = data.get("cursor")
            if not cursor:
                break

    def iter_all_markets(
        self,
        status: Optional[str] = None,
    ) -> Iterator[dict]:
        """Iterate over all markets with pagination."""
        cursor = None
        while True:
            data = self.get_markets(limit=200, cursor=cursor, status=status)
            for m in data.get("markets", []):
                yield m
            cursor = data.get("cursor")
            if not cursor:
                break

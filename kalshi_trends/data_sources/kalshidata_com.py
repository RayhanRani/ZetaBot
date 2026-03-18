"""
Kalshi API client for market data.

Fetches from api.elections.kalshi.com. Uses API key from .env when set.
"""

import base64
import os

from dotenv import load_dotenv

load_dotenv()

import time
import logging
from typing import Any, Iterator, Optional

import httpx

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.primitives.hashes import SHA256
    from cryptography.hazmat.backends import default_backend
    _CRYPTO_AVAILABLE = True
except ImportError:
    _CRYPTO_AVAILABLE = False

logger = logging.getLogger(__name__)

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
DEFAULT_RATE_LIMIT_DELAY = 1.0


def _load_private_key() -> Optional[Any]:
    """Load RSA private key from KALSHI_PRIVATE_KEY env."""
    if not _CRYPTO_AVAILABLE:
        return None
    pem = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
    if not pem or "-----BEGIN" not in pem:
        return None
    return serialization.load_pem_private_key(
        pem.encode(), password=None, backend=default_backend()
    )


def _sign_request(private_key: Any, method: str, path: str) -> dict:
    """Return Kalshi auth headers. Path must exclude query string."""
    if not _CRYPTO_AVAILABLE:
        return {}
    ts_ms = int(time.time() * 1000)
    msg = f"{ts_ms}{method}{path}"
    sig = private_key.sign(
        msg.encode(),
        padding.PSS(mgf=padding.MGF1(SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
        SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": os.environ.get("KALSHI_API_KEY_ID", ""),
        "KALSHI-ACCESS-TIMESTAMP": str(ts_ms),
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(sig).decode(),
    }


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
    """Client for Kalshi market data. Uses API key from .env when set."""

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
        self._private_key = _load_private_key()
        self._http_client: Optional[httpx.Client] = None

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.monotonic()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        """GET request with retry, rate limiting, optional auth. Reuses HTTP connection."""
        path_only = path.split("?")[0] if "?" in path else path
        url = f"{self.base_url}{path}"

        def _do():
            self._throttle()
            headers = {}
            if self._private_key and os.environ.get("KALSHI_API_KEY_ID"):
                headers = _sign_request(self._private_key, "GET", path_only)
            
            # Reuse HTTP client connection for better performance
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout)
            
            resp = self._http_client.get(url, params=params, headers=headers or None)
            resp.raise_for_status()
            return resp.json()

        return _retry_with_backoff(_do)
    
    def __del__(self):
        """Clean up HTTP client on deletion."""
        if self._http_client is not None:
            try:
                self._http_client.close()
            except:
                pass

    def _post(self, path: str, body: dict) -> dict:
        """POST request with retry, rate limiting, auth."""
        def _do():
            self._throttle()
            headers = {"Content-Type": "application/json"}
            if self._private_key and os.environ.get("KALSHI_API_KEY_ID"):
                headers.update(_sign_request(self._private_key, "POST", path))
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout)
            resp = self._http_client.post(
                f"{self.base_url}{path}", json=body, headers=headers
            )
            resp.raise_for_status()
            return resp.json()
        return _retry_with_backoff(_do)

    def _delete(self, path: str) -> dict:
        """DELETE request with retry, rate limiting, auth."""
        def _do():
            self._throttle()
            headers = {}
            if self._private_key and os.environ.get("KALSHI_API_KEY_ID"):
                headers = _sign_request(self._private_key, "DELETE", path)
            if self._http_client is None:
                self._http_client = httpx.Client(timeout=self.timeout)
            resp = self._http_client.delete(
                f"{self.base_url}{path}", headers=headers or None
            )
            resp.raise_for_status()
            return resp.json()
        return _retry_with_backoff(_do)

    def get_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
        status: Optional[str] = None,
        tickers: Optional[str] = None,
    ) -> dict:
        """Fetch markets list. Returns {markets, cursor}. tickers: comma-separated list (max 10)."""
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if status:
            params["status"] = status
        if tickers:
            params["tickers"] = tickers
        return self._get("/markets", params)

    def get_market_titles(self, ticker_list: list[str]) -> dict[str, str]:
        """Fetch market titles for given tickers. Returns {ticker: title}. Batches of 10."""
        result: dict[str, str] = {}
        for i in range(0, len(ticker_list), 10):
            batch = ticker_list[i : i + 10]
            tickers_param = ",".join(batch)
            data = self.get_markets(limit=10, tickers=tickers_param)
            for m in data.get("markets", []):
                t = m.get("ticker")
                if t:
                    result[t] = m.get("title") or m.get("subtitle") or t
        return result

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

    def get_trades(
        self,
        limit: int = 500,
        cursor: Optional[str] = None,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
    ) -> dict:
        """Fetch trades. Returns {trades, cursor}."""
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        if ticker:
            params["ticker"] = ticker
        if min_ts:
            params["min_ts"] = min_ts
        if max_ts:
            params["max_ts"] = max_ts
        return self._get("/markets/trades", params)

    def iter_all_trades(
        self,
        ticker: Optional[str] = None,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        max_trades: Optional[int] = None,
    ) -> Iterator[dict]:
        """Iterate over all trades with pagination. Uses larger batch size (1000) for efficiency."""
        cursor = None
        count = 0
        while True:
            data = self.get_trades(
                limit=1000,  # Increased from 500 to reduce API calls
                cursor=cursor,
                ticker=ticker,
                min_ts=min_ts,
                max_ts=max_ts,
            )
            for t in data.get("trades", []):
                yield t
                count += 1
                if max_trades and count >= max_trades:
                    return
            cursor = data.get("cursor")
            if not cursor:
                break

    def fetch_trades_chunk(
        self,
        min_ts: Optional[int] = None,
        max_ts: Optional[int] = None,
        cursor: Optional[str] = None,
        max_trades: int = 100000,
    ) -> tuple[list[dict], Optional[str], bool]:
        """
        Fetch up to max_trades trades. Returns (list of trades, next_cursor, has_more).
        Use next_cursor on the next call to continue. has_more is False when no more data.
        """
        trades: list[dict] = []
        current_cursor = cursor
        while len(trades) < max_trades:
            data = self.get_trades(
                limit=1000,
                cursor=current_cursor,
                min_ts=min_ts,
                max_ts=max_ts,
            )
            batch = data.get("trades", [])
            if not batch:
                return trades, None, False
            trades.extend(batch)
            current_cursor = data.get("cursor")
            if not current_cursor:
                return trades, None, False
        return trades, current_cursor, True

    # ---- Historical API (for archived data; same base URL) ----
    def get_historical_cutoff(self) -> dict:
        """Returns cutoff timestamps for live vs historical data."""
        return self._get("/historical/cutoff")

    def get_historical_markets(
        self,
        limit: int = 200,
        cursor: Optional[str] = None,
    ) -> dict:
        """Settled markets older than cutoff. Returns {markets, cursor}."""
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/historical/markets", params)

    def iter_historical_markets(self) -> Iterator[dict]:
        """Iterate over all historical (archived) markets."""
        cursor = None
        while True:
            data = self.get_historical_markets(limit=200, cursor=cursor)
            for m in data.get("markets", []):
                yield m
            cursor = data.get("cursor")
            if not cursor:
                break

    def get_historical_market_candlesticks(
        self,
        ticker: str,
        start_ts: int,
        end_ts: int,
        period_interval: int = 1440,
    ) -> dict:
        """Candlestick (OHLC + volume) for an archived market. period_interval: 1, 60, or 1440 (min)."""
        path = f"/historical/markets/{ticker}/candlesticks"
        params = {
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        }
        return self._get(path, params)

    # ---- Order book ----

    def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Fetch order book for a market. Returns {orderbook: {yes: [...], no: [...]}}."""
        return self._get(f"/markets/{ticker}/orderbook", {"depth": depth})

    # ---- Portfolio (auth required) ----

    def get_portfolio_orders(
        self,
        status: Optional[str] = None,
        ticker: Optional[str] = None,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict:
        """Fetch portfolio orders. status: 'resting', 'executed', 'cancelled', 'all'."""
        params: dict[str, Any] = {"limit": limit}
        if status:
            params["status"] = status
        if ticker:
            params["ticker"] = ticker
        if cursor:
            params["cursor"] = cursor
        return self._get("/portfolio/orders", params)

    def get_portfolio_positions(
        self,
        limit: int = 100,
        cursor: Optional[str] = None,
    ) -> dict:
        """Fetch portfolio positions. Returns {market_positions, event_positions, cursor}."""
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/portfolio/positions", params)

    def create_order(
        self,
        ticker: str,
        action: str,
        side: str,
        count: int,
        order_type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> dict:
        """
        Place an order. Returns the created order object.
        action: 'buy' or 'sell'
        side: 'yes' or 'no'
        count: number of contracts
        order_type: 'limit' or 'market'
        yes_price: price in cents (1–99) when buying/selling YES
        no_price: price in cents (1–99) when buying/selling NO
        """
        body: dict[str, Any] = {
            "ticker": ticker,
            "action": action,
            "side": side,
            "count": count,
            "type": order_type,
        }
        if yes_price is not None:
            body["yes_price"] = yes_price
        if no_price is not None:
            body["no_price"] = no_price
        if client_order_id:
            body["client_order_id"] = client_order_id
        return self._post("/portfolio/orders", body)

    def cancel_order(self, order_id: str) -> dict:
        """Cancel a resting order by ID."""
        return self._delete(f"/portfolio/orders/{order_id}")

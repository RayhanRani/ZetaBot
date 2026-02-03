"""
Backtest bid/ask edge strategy on settled markets.

Uses historical candlestick close as entry price proxy (we don't have historical
bid/ask). Strategy: buy YES when close suggests undervaluation, sell when overvalued.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_sources.kalshidata_com import KalshiDataClient
from ..storage.writer import StorageWriter
from ..storage.reader import StorageReader

logger = logging.getLogger(__name__)


def fetch_settled_for_backtest(limit_events: int = 100, data_dir: str = "data", use_cache: bool = True) -> pd.DataFrame:
    """Fetch settled events with result for backtest."""
    writer = StorageWriter(data_dir)
    if use_cache:
        cached = writer.load_cached("settled_markets_backtest")
        if cached and isinstance(cached, list) and len(cached) > 0:
            return pd.DataFrame(cached)
    client = KalshiDataClient()
    rows = []
    count = 0
    for ev in client.iter_all_events(status="settled", with_nested_markets=True):
        for m in ev.get("markets", []):
            if m.get("result") not in ("yes", "no"):
                continue
            r = {
                "ticker": m.get("ticker"),
                "event_ticker": ev.get("event_ticker"),
                "series_ticker": (m.get("event_ticker") or "").split("-")[0] or ev.get("series_ticker"),
                "title": m.get("title"),
                "result": m.get("result"),
                "last_price": m.get("last_price"),
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "volume": m.get("volume"),
                "category": ev.get("category"),
            }
            rows.append(r)
        count += 1
        if count >= limit_events:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        writer.cache_response("settled_markets_backtest", df.to_dict(orient="records"))
    return df


def run_backtest_simple(
    settled_df: pd.DataFrame,
    entry_price_col: str = "last_price",
) -> dict:
    """
    Simple backtest: use last_price as entry (proxy for pre-close price).
    Buy YES when entry < 35 (undervalued), sell YES when entry > 65 (overvalued).
    Returns metrics.
    """
    if settled_df.empty or "result" not in settled_df.columns:
        return {"trades": 0, "win_rate": 0, "total_pnl_cents": 0, "sharpe": 0}

    df = settled_df.copy()
    df[entry_price_col] = pd.to_numeric(df[entry_price_col], errors="coerce").fillna(50)
    df = df[df[entry_price_col] > 0]
    df = df[df[entry_price_col] < 100]

    results = []
    # Strategy 1: Buy YES when price < 35 (expecting YES)
    buy_yes = df[df[entry_price_col] < 35].copy()
    for _, row in buy_yes.iterrows():
        entry = row[entry_price_col]
        outcome = row["result"]
        if outcome == "yes":
            pnl = 100 - entry  # Paid entry, got 100
        else:
            pnl = -entry  # Paid entry, got 0
        results.append({"side": "buy_yes", "entry": entry, "outcome": outcome, "pnl": pnl})

    # Strategy 2: Sell YES (buy NO) when price > 65 (expecting NO)
    sell_yes = df[df[entry_price_col] > 65].copy()
    for _, row in sell_yes.iterrows():
        entry = row[entry_price_col]
        outcome = row["result"]
        if outcome == "no":
            pnl = entry  # Received entry, paid 0
        else:
            pnl = entry - 100  # Received entry, paid 100
        results.append({"side": "sell_yes", "entry": entry, "outcome": outcome, "pnl": pnl})

    if not results:
        return {"trades": 0, "win_rate": 0, "total_pnl_cents": 0, "sharpe": 0}

    res_df = pd.DataFrame(results)
    wins = (res_df["pnl"] > 0).sum()
    total = len(res_df)
    total_pnl = res_df["pnl"].sum()
    sharpe = res_df["pnl"].mean() / (res_df["pnl"].std() + 1e-8) * (total ** 0.5) if total > 1 else 0

    return {
        "trades": total,
        "wins": int(wins),
        "losses": total - wins,
        "win_rate": wins / total if total else 0,
        "total_pnl_cents": total_pnl,
        "avg_pnl_cents": res_df["pnl"].mean(),
        "sharpe": float(sharpe),
        "buy_yes_trades": len(buy_yes),
        "sell_yes_trades": len(sell_yes),
    }


def run_backtest_with_candlesticks(
    settled_df: pd.DataFrame,
    data_dir: str = "data",
    days_before: int = 1,
) -> dict:
    """
    Backtest using candlestick close from N days before resolution.
    Requires pre-fetched candlesticks for settled markets.
    """
    reader = StorageReader(data_dir)
    rows = []
    for _, row in settled_df.iterrows():
        ticker = row.get("ticker")
        series = row.get("series_ticker") or str(row.get("event_ticker", "")).split("-")[0]
        if not series:
            continue
        ck = reader.read_candlesticks(ticker)
        if ck is None or ck.empty:
            continue
        # Use last close as entry
        close = ck["close"].iloc[-1] if "close" in ck.columns else ck.iloc[-1, 2]
        rows.append({
            "ticker": ticker,
            "result": row["result"],
            "last_price": close,
        })
    if not rows:
        return run_backtest_simple(settled_df)
    sub = pd.DataFrame(rows)
    return run_backtest_simple(sub, entry_price_col="last_price")

"""Live web dashboard for Kalshi trends."""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def _get_data_dir() -> Path:
    """Resolve data directory relative to project root."""
    env_dir = os.environ.get("KALSHI_TRENDS_DATA_DIR")
    if env_dir:
        return Path(env_dir)
    pkg_dir = Path(__file__).resolve().parent.parent
    return pkg_dir / "data"


def fetch_fresh_events(data_dir: Path) -> pd.DataFrame:
    """Fetch all open events from Kalshi API (no cache)."""
    from .data_sources.kalshidata_com import KalshiDataClient
    from .storage.writer import StorageWriter
    from .storage.reader import StorageReader

    client = KalshiDataClient()
    writer = StorageWriter(str(data_dir))
    events = []
    for ev in client.iter_all_events(status="open", with_nested_markets=True):
        events.append(ev)
    writer.write_events(events)
    writer.cache_response("events_with_markets", {"events": events})
    reader = StorageReader(str(data_dir))
    return reader.read_latest_events()


def load_dashboard_data(data_dir: Path = None, use_cache: bool = True) -> dict:
    """Load all data needed for the dashboard."""
    from .storage.reader import StorageReader

    data_dir = data_dir or _get_data_dir()
    reader = StorageReader(str(data_dir))
    events_df = reader.read_latest_events() if use_cache else None
    bid_ask_edge = reader.read_processed("bid_ask_edge")
    top_movers = reader.read_processed("top_movers")
    highest_volume = reader.read_processed("highest_volume")
    category_agg = reader.read_processed("category_aggregates")
    clustering = reader.read_processed("clustering")
    pattern_ranked = reader.read_processed("pattern_ranked_markets")

    return {
        "events": events_df,
        "bid_ask_edge": bid_ask_edge,
        "top_movers": top_movers,
        "highest_volume": highest_volume,
        "category_agg": category_agg,
        "clustering": clustering,
        "pattern_ranked": pattern_ranked,
    }


def _build_unified_table(events_df: pd.DataFrame, data: dict) -> pd.DataFrame:
    """Merge all data sources into a single table for display."""
    df = events_df.copy()
    for col in ["last_price", "volume", "volume_24h", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Sub-category: event-level grouping (e.g. "Who will the next Pope be?")
    if "event_title" not in df.columns and "event_ticker" in df.columns:
        df["sub_category"] = df["event_ticker"]
    elif "event_title" in df.columns:
        df["sub_category"] = df["event_title"]
    else:
        df["sub_category"] = df.get("event_ticker", "Unknown")

    # Merge bid/ask edge
    edge = data.get("bid_ask_edge")
    if edge is not None and not edge.empty and "ticker" in edge.columns:
        edge_cols = [c for c in ["mid", "spread", "edge_score", "buy_yes_edge", "sell_yes_edge"] if c in edge.columns]
        merge_cols = ["ticker"] + edge_cols
        edge_sub = edge[merge_cols].drop_duplicates(subset=["ticker"])
        df = df.merge(edge_sub, on="ticker", how="left", suffixes=("", "_edge"))

    # Merge pattern signal strength
    pr = data.get("pattern_ranked")
    if pr is not None and not pr.empty and "ticker" in pr.columns and "signal_strength" in pr.columns:
        pr_sub = pr[["ticker", "signal_strength"]].drop_duplicates(subset=["ticker"])
        df = df.merge(pr_sub, on="ticker", how="left")

    # Merge cluster
    cl = data.get("clustering")
    if cl is not None and not cl.empty and "ticker" in cl.columns and "cluster" in cl.columns:
        cl_sub = cl[["ticker", "cluster"]].drop_duplicates(subset=["ticker"])
        df = df.merge(cl_sub, on="ticker", how="left")

    return df


def build_dashboard(data_dir: Path = None):
    """Build Streamlit dashboard."""
    import streamlit as st

    st.set_page_config(
        page_title="Kalshi Trends",
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Kalshi Trends Dashboard")

    # Column glossary - explain each term
    glossary = {
        "sub_category": "Event or question. Markets in the same event are grouped together.",
        "ticker": "Unique ID for this market contract.",
        "title": "The exact bet question.",
        "category": "Broad topic (World, Science, Climate, etc.).",
        "implied_prob_cents": "Market price in cents (0-100). 50¢ = 50% probability. What you pay per share.",
        "volume": "Total contracts traded since market opened.",
        "volume_24h": "Contracts traded in last 24 hours.",
        "open_interest": "Contracts currently held (not closed).",
        "mid": "Midpoint between bid and ask. Fair value estimate.",
        "spread": "Ask minus bid. Lower = cheaper to trade.",
        "edge_score": "Narrow spread + liquidity. Higher = better conditions.",
        "buy_yes_edge": "Edge for buying YES when price below fair value.",
        "sell_yes_edge": "Edge for selling YES when price above fair value.",
        "signal_strength": "Distance from 50¢ times volume. High = conviction + activity.",
        "cluster": "Group of similar markets by behavior.",
    }
    st.header("Column Glossary")
    st.caption("What each column means:")
    g1, g2 = st.columns(2)
    items = list(glossary.items())
    mid = (len(items) + 1) // 2
    with g1:
        for col, desc in items[:mid]:
            st.markdown(f"**{col}** — {desc}")
    with g2:
        for col, desc in items[mid:]:
            st.markdown(f"**{col}** — {desc}")


def run_server(host: str = "0.0.0.0", port: int = 8501, data_dir: str = "data"):
    """Run Streamlit dashboard."""
    import streamlit.web.cli as stcli
    import sys

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent.parent / "dashboard.py"),
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--server.headless",
        "true",
    ]
    stcli.main()


def main():
    """Entry point when run as script."""
    build_dashboard()

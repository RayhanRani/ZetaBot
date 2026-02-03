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
    st.caption("All ongoing Kalshi markets - live data")

    data_path = _get_data_dir()
    with st.sidebar:
        st.caption("Data path: " + str(data_path))
        if st.button("Refresh data", help="Fetch fresh data from Kalshi API"):
            with st.spinner("Fetching all open markets..."):
                try:
                    fresh = fetch_fresh_events(data_path)
                    st.success(f"Loaded {len(fresh)} markets")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    # Load data - prefer fresh fetch on first load if cache empty
    try:
        data = load_dashboard_data(data_dir)
        events_df = data["events"]
        if events_df is None or events_df.empty:
            with st.spinner("Fetching data from Kalshi API..."):
                events_df = fetch_fresh_events(data_path)
                data["events"] = events_df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        import traceback
        with st.expander("Error details"):
            st.code(traceback.format_exc())
        return

    if events_df is None or events_df.empty:
        st.warning("No data found. Click 'Refresh data' in the sidebar to fetch from Kalshi.")
        return

    # Ensure event_title for sub-category
    if "event_title" not in events_df.columns and "event_ticker" in events_df.columns:
        events_df = events_df.copy()
        events_df["event_title"] = events_df["event_ticker"]
    if "event_title" not in events_df.columns:
        events_df = events_df.copy()
        events_df["event_title"] = events_df.get("title", "Unknown")

    # Compute edge/pattern on the fly if not in storage
    if data.get("bid_ask_edge") is None or data["bid_ask_edge"].empty:
        from .analysis.edge import compute_bid_ask_edge, rank_by_edge
        data["bid_ask_edge"] = rank_by_edge(events_df, min_liquidity=500)
        if data["bid_ask_edge"].empty:
            data["bid_ask_edge"] = compute_bid_ask_edge(events_df)
    if data.get("pattern_ranked") is None or data["pattern_ranked"].empty:
        pr = events_df.copy()
        pr["move_from_neutral"] = abs(pr["last_price"] - 50) if "last_price" in pr.columns else 0
        if "volume" in pr.columns:
            pr["signal_strength"] = (pr["move_from_neutral"] * pr["volume"]).fillna(0)
        else:
            pr["signal_strength"] = pr["move_from_neutral"]
        pr = pr.nlargest(min(30, len(pr)), "signal_strength")
        data["pattern_ranked"] = pr

    # Build unified table
    table_df = _build_unified_table(events_df, data)

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Markets", len(table_df))
    with col2:
        sub_cats = table_df["sub_category"].nunique() if "sub_category" in table_df.columns else 0
        st.metric("Events / Sub-categories", sub_cats)
    with col3:
        vol = table_df["volume"].sum() if "volume" in table_df.columns else 0
        st.metric("Total Volume", f"{vol:,.0f}")
    with col4:
        avg_price = table_df["last_price"].mean() if "last_price" in table_df.columns else 0
        st.metric("Avg Implied Prob", f"{avg_price:.1f}¢")

    st.divider()

    # Column glossary - explain each term at the top
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

    st.divider()

    # Data table with sub-category grouping
    st.header("All Ongoing Bets")
    display_cols = [
        "sub_category", "ticker", "title", "category",
        "last_price", "volume", "volume_24h", "open_interest",
        "mid", "spread", "edge_score", "buy_yes_edge", "sell_yes_edge",
        "signal_strength", "cluster",
    ]
    display_cols = [c for c in display_cols if c in table_df.columns]
    table_display = table_df[display_cols].copy()

    # Rename price column for clarity
    if "last_price" in table_display.columns:
        table_display = table_display.rename(columns={"last_price": "implied_prob_cents"})

    # Round numeric columns
    for c in table_display.columns:
        if table_display[c].dtype in ["float64", "float32"]:
            table_display[c] = table_display[c].round(2)

    # Sort by sub_category then by volume
    if "sub_category" in table_display.columns:
        sort_cols = ["sub_category", "volume"] if "volume" in table_display.columns else ["sub_category"]
        asc = [True, False] if len(sort_cols) == 2 else [True]
        table_display = table_display.sort_values(sort_cols, ascending=asc)

    with st.expander("View all data (click to expand/collapse)", expanded=True):
        st.dataframe(table_display, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Data from Kalshi API - Click 'Refresh data' in sidebar to update.")


def run_server(host: str = "0.0.0.0", port: int = 8501, data_dir: str = "data"):
    """Run Streamlit dashboard."""
    import streamlit.web.cli as stcli
    import sys

    sys.argv = [
        "streamlit",
        "run",
        str(Path(__file__).parent.parent / "run_dashboard.py"),
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

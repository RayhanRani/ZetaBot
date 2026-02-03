"""CLI entry point for kalshi_trends."""

import argparse
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

from .data_sources.kalshidata_com import KalshiDataClient
from .storage.writer import StorageWriter
from .storage.reader import StorageReader
from .analysis.trends import TrendAnalyzer
from .analysis.patterns import cluster_markets
from .analysis.edge import compute_bid_ask_edge, rank_by_edge
from .analysis.backtest import fetch_settled_for_backtest, run_backtest_simple
from .report import generate_report

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _parse_args():
    p = argparse.ArgumentParser(description="Kalshi Trends - fetch and analyze market data")
    sub = p.add_subparsers(dest="command", required=True)

    fetch_p = sub.add_parser("fetch", help="Fetch market data from KalshiData source")
    fetch_p.add_argument("--days", type=int, default=30, help="Days of history to fetch (for candlesticks)")
    fetch_p.add_argument("--no-cache", action="store_true", help="Ignore cache, always fetch fresh")
    fetch_p.add_argument("--limit-events", type=int, default=0, help="Limit events to fetch (0=all)")
    fetch_p.add_argument("--data-dir", default="data", help="Base data directory")

    analyze_p = sub.add_parser("analyze", help="Run analysis and generate report")
    analyze_p.add_argument("--days", type=int, default=30, help="Days of data to analyze")
    analyze_p.add_argument("--data-dir", default="data", help="Base data directory")
    analyze_p.add_argument("--output", default="data/processed", help="Output directory for report")

    serve_p = sub.add_parser("serve", help="Run live web dashboard")
    serve_p.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_p.add_argument("--port", type=int, default=8501, help="Port to bind")
    serve_p.add_argument("--data-dir", default="data", help="Data directory")

    backtest_p = sub.add_parser("backtest", help="Run bid/ask edge backtest on settled markets")
    backtest_p.add_argument("--limit-events", type=int, default=50, help="Settled events to fetch")
    backtest_p.add_argument("--data-dir", default="data", help="Data directory")

    return p.parse_args()


def cmd_fetch(args) -> None:
    """Fetch markets and events from KalshiData (Kalshi API)."""
    rate_limit = float(os.getenv("KALSHIDATA_RATE_LIMIT", "60"))
    delay = 60.0 / rate_limit if rate_limit > 0 else 1.0

    client = KalshiDataClient(rate_limit_delay=delay)
    writer = StorageWriter(args.data_dir)

    # Use cache for events if available and not --no-cache
    cache_key = "events_with_markets"
    if not args.no_cache:
        cached = writer.load_cached(cache_key)
        if cached and "events" in cached:
            events = cached["events"]
            logger.info("Using cached events (%d events)", len(events))
        else:
            cached = None
    else:
        cached = None

    if cached is None:
        events = []
        count = 0
        for ev in client.iter_all_events(status="open", with_nested_markets=True):
            events.append(ev)
            count += 1
            if args.limit_events and count >= args.limit_events:
                break
        writer.cache_response(cache_key, {"events": events})
        logger.info("Fetched %d events", len(events))

    # Flatten for candlestick sampling
    rows = []
    for ev in events:
        base = {
            "event_ticker": ev.get("event_ticker"),
            "series_ticker": ev.get("series_ticker"),
            "title": ev.get("title"),
            "category": ev.get("category"),
        }
        for m in ev.get("markets", []):
            row = {**base, **m}
            rows.append(row)

    writer.write_events(events)
    logger.info("Wrote events with %d total markets", len(rows))

    # Optionally fetch candlesticks for a sample of active markets (expensive)
    if args.days > 0 and len(rows) > 0:
        end_ts = int(datetime.now().timestamp())
        start_ts = int((datetime.now() - timedelta(days=args.days)).timestamp())
        sample = [r for r in rows if r.get("status") == "active" and r.get("volume", 0) > 100][:5]
        for r in sample:
            ticker = r.get("ticker")
            series = r.get("series_ticker") or r.get("_event_ticker", "").split("-")[0]
            if not series:
                continue
            try:
                data = client.get_market_candlesticks(
                    series_ticker=series,
                    ticker=ticker,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    period_interval=1440,
                )
                candlesticks = data.get("candlesticks", [])
                if candlesticks:
                    writer.write_candlesticks(ticker, candlesticks)
                    logger.info("Fetched %d candlesticks for %s", len(candlesticks), ticker)
            except Exception as e:
                logger.warning("Could not fetch candlesticks for %s: %s", ticker, e)


def cmd_analyze(args) -> None:
    """Run analysis and generate report."""
    reader = StorageReader(args.data_dir)
    writer = StorageWriter(args.data_dir)

    events_df = reader.read_latest_events()
    if events_df is None or events_df.empty:
        logger.error("No events data found. Run 'fetch' first.")
        return

    # Ensure numeric columns
    for col in ["last_price", "volume", "volume_24h", "open_interest"]:
        if col in events_df.columns:
            events_df[col] = events_df[col].fillna(0).astype(float)

    # Add category from event if missing
    if "category" not in events_df.columns and "event_ticker" in events_df.columns:
        events_df["category"] = "Unknown"

    # Trend analysis
    trend = TrendAnalyzer(events_df)
    top_movers = trend.top_movers()
    most_volatile = trend.most_volatile()
    highest_volume = trend.highest_volume()
    highest_oi = trend.highest_open_interest()
    category_agg = trend.category_aggregates()

    # Bid/ask edge (options-style)
    edge_df = compute_bid_ask_edge(events_df)
    edge_ranked = rank_by_edge(events_df, min_liquidity=500) if not edge_df.empty else edge_df

    # Pattern / clustering
    features = ["last_price", "volume", "open_interest"]
    features = [f for f in features if f in events_df.columns]
    clustering = cluster_markets(events_df, features, n_clusters=5)

    # Pattern-ranked: combine signal strength metrics
    pattern_ranked = events_df.copy()
    if "last_price" in pattern_ranked.columns:
        pattern_ranked["move_from_neutral"] = abs(pattern_ranked["last_price"] - 50)
    if "volume" in pattern_ranked.columns:
        pattern_ranked["signal_strength"] = (
            pattern_ranked["move_from_neutral"] * pattern_ranked["volume"]
        ).fillna(0)
        pattern_ranked = pattern_ranked.nlargest(30, "signal_strength")

    # Save processed outputs
    if not edge_ranked.empty:
        writer.write_processed("bid_ask_edge", edge_ranked)
    elif not edge_df.empty:
        writer.write_processed("bid_ask_edge", edge_df)
    writer.write_processed("top_movers", top_movers)
    writer.write_processed("most_volatile", most_volatile)
    writer.write_processed("highest_volume", highest_volume)
    writer.write_processed("category_aggregates", category_agg)
    writer.write_processed("clustering", clustering)
    writer.write_processed("pattern_ranked_markets", pattern_ranked)

    # Report
    output_dir = Path(args.output)
    report_path = generate_report(
        events_df=events_df,
        top_movers=top_movers,
        most_volatile=most_volatile,
        highest_volume=highest_volume,
        category_agg=category_agg,
        clustering=clustering,
        pattern_ranked=pattern_ranked,
        output_dir=output_dir,
    )
    logger.info("Report saved to %s", report_path)


def cmd_backtest(args) -> None:
    """Run bid/ask edge backtest on settled markets."""
    settled = fetch_settled_for_backtest(limit_events=args.limit_events, data_dir=args.data_dir)
    if settled.empty:
        logger.error("No settled markets found.")
        return
    metrics = run_backtest_simple(settled)
    logger.info("Backtest results: %s", metrics)
    print("\n--- Bid/Ask Edge Backtest ---")
    print(f"Trades: {metrics['trades']}")
    print(f"Wins: {metrics['wins']} | Losses: {metrics['losses']}")
    print(f"Win rate: {metrics['win_rate']:.1%}")
    print(f"Total PnL (cents): {metrics['total_pnl_cents']:.1f}")
    print(f"Avg PnL (cents): {metrics['avg_pnl_cents']:.2f}")
    print(f"Sharpe (approx): {metrics['sharpe']:.2f}")


def cmd_serve(args) -> None:
    """Run Streamlit live dashboard."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    dashboard_path = project_root / "run_dashboard.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_path.resolve()),
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]
    env = os.environ.copy()
    data_dir_abs = str((project_root / args.data_dir).resolve())
    env["KALSHI_TRENDS_DATA_DIR"] = data_dir_abs
    subprocess.run(cmd, env=env, cwd=str(project_root))


def main() -> None:
    args = _parse_args()
    if args.command == "fetch":
        cmd_fetch(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "serve":
        cmd_serve(args)
    elif args.command == "backtest":
        cmd_backtest(args)


if __name__ == "__main__":
    main()

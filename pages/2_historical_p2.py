"""Historical data P2 - candlestick extraction (OHLC + volume)."""

import csv
import os
import tempfile
import time
from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Kalshi Historical Data P2", layout="wide")
st.title("Kalshi Historical Data P2")
st.caption("Candlestick extraction: OHLC + volume per market per period")

# API Limits info
with st.expander("API Rate Limits & Trade Limits", expanded=False):
    st.markdown("""
    **Rate Limits (based on your API tier):**
    - Basic: 20 requests/second
    - Advanced: 30 requests/second  
    - Premier: 100 requests/second
    - Prime: 400 requests/second
    
    **Trade Data Limits:**
    - No hard limit on total trades you can fetch
    - API uses pagination (1000 trades per request)
    - Respects rate limits automatically (0.3s delay between requests)
    - Check your tier: `/account/limits` endpoint
    """)

# Candlestick Extraction Strategy
with st.expander("Candlestick Extraction Strategy", expanded=False):
    st.markdown("""
    **Candlestick Extraction (OHLC + Volume)**  
    Use when you need price/volume over time per market, not every fill. One row per market per period.

    **What you get:**

    - **Daily:** One row per market per day — open, high, low, close, volume, open interest
    - **Hourly:** One row per market per hour — same fields, finer granularity

    **How it works:**

    1. **Settled markets** — Fetches from settled events and their nested markets
    2. **Per-market candlesticks** — For each market, requests candlestick data for the selected month (daily or hourly buckets)
    3. **Single CSV** — All rows combined into one file: `kalshi_candlesticks_YYYYMM.csv`
    4. **Max markets** — You cap how many markets to include (e.g. 200) to limit API calls and file size

    **Why use Candlesticks?**

    - Much smaller: e.g. 200 markets × 30 days ≈ 6,000 rows vs millions of trade rows
    - Faster: Fewer API calls; one CSV, no chunking
    - Enough for: Backtests, charts, price/volume analysis, time-series features

    **When to use Trades instead:** When you need exact fill timestamps, sizes, or order-level detail — use the **Historical** page for chunked trades extraction.
    """)

if "csv_file_paths" not in st.session_state:
    st.session_state.csv_file_paths = []
if "csv_data_bytes" not in st.session_state:
    st.session_state.csv_data_bytes = None  # Store file bytes for reliable download
if "download_filename" not in st.session_state:
    st.session_state.download_filename = None
if "row_count" not in st.session_state:
    st.session_state.row_count = 0
if "column_count" not in st.session_state:
    st.session_state.column_count = 0


def get_client():
    try:
        from kalshi_trends.data_sources.kalshidata_com import KalshiDataClient
        rate_limit = float(os.environ.get("KALSHIDATA_RATE_LIMIT", "0.3"))
        return KalshiDataClient(rate_limit_delay=rate_limit), None
    except ImportError as e:
        if "cryptography" in str(e).lower():
            return None, "Install the cryptography package: pip install cryptography"
        return None, str(e)


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    result = {}
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, dict):
            result.update(_flatten_dict(v, prefix=key))
        elif isinstance(v, list):
            result[key] = str(v) if v else None
        else:
            result[key] = v
    return result


def _flatten_candlestick(c: dict, ticker: str, series_ticker: str = "") -> dict:
    flat = _flatten_dict(c)
    flat["ticker"] = ticker
    flat["series_ticker"] = series_ticker
    return flat


CANDLESTICK_REQUEST_DELAY = float(os.environ.get("KALSHI_CANDLESTICK_DELAY", "1.0"))


def _fetch_from_historical_api(
    client, min_ts: int, max_ts: int, period_interval: int, max_markets: int
) -> tuple[list[dict], int, list[str]]:
    """Use historical API (archived markets). Returns (rows, markets_processed, errors)."""
    rows: list[dict] = []
    market_count = 0
    errors: list[str] = []
    for m in client.iter_historical_markets():
        if market_count >= max_markets:
            break
        ticker = m.get("ticker")
        if not ticker:
            continue
        if market_count > 0:
            time.sleep(CANDLESTICK_REQUEST_DELAY)
        try:
            data = client.get_historical_market_candlesticks(
                ticker=ticker,
                start_ts=min_ts,
                end_ts=max_ts,
                period_interval=period_interval,
            )
        except Exception as e:
            if market_count < 3:  # Only log first few errors to avoid spam
                errors.append(f"{ticker}: {str(e)}")
            continue
        series_ticker = m.get("series_ticker") or ""
        candlesticks = data.get("candlesticks", [])
        for c in candlesticks:
            rows.append(_flatten_candlestick(c, ticker, series_ticker))
        market_count += 1
    return rows, market_count, errors


def _fetch_from_events_api(
    client, min_ts: int, max_ts: int, period_interval: int, max_markets: int
) -> tuple[list[dict], int, list[str]]:
    """Use live events API (settled events with nested markets). Returns (rows, markets_processed, errors)."""
    rows: list[dict] = []
    market_count = 0
    errors: list[str] = []
    for ev in client.iter_all_events(status="settled", with_nested_markets=True):
        if market_count >= max_markets:
            break
        series_ticker = ev.get("series_ticker") or ""
        for m in ev.get("markets", [])[: max_markets - market_count]:
            if market_count >= max_markets:
                break
            ticker = m.get("ticker")
            if not ticker:
                continue
            if market_count > 0:
                time.sleep(CANDLESTICK_REQUEST_DELAY)
            try:
                data = client.get_market_candlesticks(
                    series_ticker=series_ticker,
                    ticker=ticker,
                    start_ts=min_ts,
                    end_ts=max_ts,
                    period_interval=period_interval,
                )
            except Exception as e:
                if market_count < 3:  # Only log first few errors to avoid spam
                    errors.append(f"{ticker}: {str(e)}")
                continue
            candlesticks = data.get("candlesticks", [])
            for c in candlesticks:
                rows.append(_flatten_candlestick(c, ticker, series_ticker))
            market_count += 1
    return rows, market_count, errors


def fetch_candlesticks_to_csv(min_ts: int, max_ts: int, period_interval: int, max_markets: int) -> tuple[str, int, int, list[str]]:
    """
    Fetch candlesticks: try historical API first (archived), then events API (recent settled) if empty.
    Returns (csv_path, row_count, col_count, errors).
    """
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    fd, csv_path = tempfile.mkstemp(suffix=".csv", prefix="kalshi_candlesticks_")
    os.close(fd)
    all_errors = []
    try:
        rows, hist_count, hist_errors = _fetch_from_historical_api(client, min_ts, max_ts, period_interval, max_markets)
        all_errors.extend(hist_errors)
        if not rows:
            rows, events_count, events_errors = _fetch_from_events_api(client, min_ts, max_ts, period_interval, max_markets)
            all_errors.extend(events_errors)
        if not rows:
            os.unlink(csv_path)
            return "", 0, 0, all_errors
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        column_names = sorted(all_keys)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=column_names, extrasaction="ignore")
            w.writeheader()
            w.writerows([{k: r.get(k) for k in column_names} for r in rows])
        return csv_path, len(rows), len(column_names), all_errors
    except Exception as e:
        if os.path.exists(csv_path):
            os.unlink(csv_path)
        raise e


st.header("Data Extraction")

kalshi_start = datetime(2021, 7, 1)
today_month_start = datetime(datetime.now().year, datetime.now().month, 1)
months = []
cur = kalshi_start
while cur <= today_month_start:
    months.append(cur)
    if cur.month == 12:
        cur = cur.replace(year=cur.year + 1, month=1)
    else:
        cur = cur.replace(month=cur.month + 1)

years = sorted({m.year for m in months})
selected_year = st.selectbox("Year", years, index=len(years) - 1, key="candlestick_year")
available_months_for_year = [m for m in months if m.year == selected_year]
month_options = [(m.month, m.strftime("%B")) for m in available_months_for_year]
selected_month_num = st.selectbox(
    "Month",
    [m[0] for m in month_options],
    index=len(month_options) - 1,
    format_func=lambda x: next(label for num, label in month_options if num == x),
    key="candlestick_month",
)
selected_month = next(m for m in available_months_for_year if m.month == selected_month_num)
if selected_month.month == 12:
    next_month = selected_month.replace(year=selected_month.year + 1, month=1)
else:
    next_month = selected_month.replace(month=selected_month.month + 1)
min_ts = int(selected_month.timestamp())
max_ts = int(next_month.timestamp())

# Daily vs hourly
interval_choice = st.radio(
    "Granularity",
    ["Daily (one row per market per day)", "Hourly (one row per market per hour)"],
    index=0,
    key="candlestick_interval_choice",
)
candlestick_interval = 1440 if "Daily" in interval_choice else 60  # minutes

st.caption(
    f"Extracting candlesticks for {selected_month.strftime('%B %Y')} (UTC). "
    f"One row per market per {'day' if candlestick_interval == 1440 else 'hour'}."
)
max_markets = st.number_input(
    "Max markets to include (to limit API calls)",
    min_value=10,
    max_value=2000,
    value=200,
    key="max_markets_candles",
)

if st.button("Extract Candlesticks", type="primary"):
    st.session_state.csv_file_paths = []
    st.session_state.csv_data_bytes = None
    st.session_state.download_filename = None
    with st.spinner(
        f"Fetching candlesticks for {selected_month.strftime('%B %Y')} (up to {max_markets} markets)..."
    ):
        try:
            path, row_count, col_count, errors = fetch_candlesticks_to_csv(
                min_ts, max_ts, candlestick_interval, int(max_markets)
            )
            if path and row_count and os.path.exists(path):
                with open(path, "rb") as f:
                    st.session_state.csv_data_bytes = f.read()
                st.session_state.download_filename = f"kalshi_candlesticks_{selected_month.strftime('%Y%m')}.csv"
                st.session_state.csv_file_paths = [(path, row_count)]
                st.session_state.row_count = row_count
                st.session_state.column_count = col_count
                if errors:
                    st.warning(f"Some markets had errors (showing first few): {', '.join(errors[:3])}")
            else:
                error_msg = "No candlestick data returned."
                if errors:
                    error_msg += f"\n\nErrors encountered:\n" + "\n".join(errors[:5])
                error_msg += "\n\nPossible reasons:\n"
                error_msg += "- Historical API may require Premier/Prime tier for archived data\n"
                error_msg += "- Selected month may be too recent (try an older month)\n"
                error_msg += "- Markets may not have candlestick data for this date range\n"
                error_msg += "- Check your API tier at: https://kalshi.com/trade-api"
                st.warning(error_msg)
            st.rerun()
        except Exception as e:
            error_str = str(e)
            st.error(f"Error: {error_str}")
            if "401" in error_str or "403" in error_str or "Unauthorized" in error_str:
                st.info("**Authentication issue**: Make sure your KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY are correct in your .env file.")
            elif "429" in error_str or "rate limit" in error_str.lower():
                st.info("**Rate limit**: Try reducing max markets or wait a moment before retrying.")
            elif "cryptography" in error_str.lower():
                st.code("pip install cryptography", language="bash")
            else:
                st.info("Check the Kalshi API documentation: https://trade-api.readme.io/")

# Download section - show when we have extracted data
if st.session_state.csv_data_bytes and st.session_state.download_filename:
    st.divider()
    st.subheader("Download your CSV file")
    st.caption("Click the button below to save the file to your computer.")
    st.metric("Rows", f"{st.session_state.row_count:,}")
    st.metric("Columns", st.session_state.column_count)
    st.download_button(
        "Download CSV",
        data=st.session_state.csv_data_bytes,
        file_name=st.session_state.download_filename,
        mime="text/csv",
        type="primary",
        key="dl_candlestick_csv",
    )

"""
Historical data - chunked trades extraction.

Streamlit page that fetches Kalshi trade history by month using cursor-based
pagination. Writes CSVs in chunks of 1M trades; progress updates every 100k trades.
Requires KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY in .env (or env).
"""

import csv
import os
import tempfile
from datetime import datetime
from typing import Optional

import streamlit as st

# -----------------------------------------------------------------------------
# Page config and copy
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Kalshi Historical Data", layout="wide")
st.title("Kalshi Historical Data")
st.caption("Extract historical trade data from Kalshi API")

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

with st.expander("Trades Extraction Strategy", expanded=False):
    st.markdown("""
    **Chunked Fetching & Progressive Download (Trades)**  
    Use when you need every fill. This approach extracts large historical trade datasets efficiently.

    **How it works:**

    - **Progressive Fetching (100,000 trades per step)**  
      Fetches data in 100k-trade increments; progress bar and count update every 100,000 trades.
      Uses API pagination with cursor-based resumption for reliability.

    - **Automatic File Chunking (1,000,000 trades per CSV)**  
      Creates a new CSV every 1,000,000 trades; files appear below as they're created.
      Progress bar fills from 0-100% over each 1M file (10 progress updates per file).
      Prevents memory issues and keeps file sizes manageable.

    - **Continuous Processing**  
      Continues automatically after each 100k file; no manual steps.
      Handles any month size, from thousands to millions of trades.

    **Why use Trades?**
    - Memory Efficiency: Streams directly to CSV
    - Reliability: Cursor-based pagination
    - Manageability: 1M trade files; naming: `kalshi_trades_YYYYMM_partN.csv`

    **Result:** Full trade-level history with progress visibility and downloadable chunks.
    """)

# -----------------------------------------------------------------------------
# Constants and session state
# -----------------------------------------------------------------------------

PROGRESS_CHUNK = 100_000   # Progress bar and count update every N trades
FILE_CHUNK = 1_000_000     # New CSV file every N trades

def _init_session_state():
    defaults = {
        "csv_file_paths": [],
        "row_count": 0,
        "column_count": 0,
        "fetch_phase": "idle",
        "fetch_cursor": None,
        "fetch_progress": 0,
        "current_csv_path": None,
        "current_csv_row_count": 0,
        "current_csv_columns": None,
        "last_file_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def get_client():
    """Return (KalshiDataClient, None) or (None, error_message)."""
    try:
        from kalshi_trends.data_sources.kalshidata_com import KalshiDataClient
        rate_limit = float(os.environ.get("KALSHIDATA_RATE_LIMIT", "0.3"))
        return KalshiDataClient(rate_limit_delay=rate_limit), None
    except ImportError as e:
        if "cryptography" in str(e).lower():
            return None, "Install the cryptography package: pip install cryptography"
        return None, str(e)


def _flatten_dict(d: dict, prefix: str = "") -> dict:
    """Flatten nested dict for CSV row; keys get optional prefix."""
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


def fetch_one_progress_chunk(
    min_ts: int, max_ts: int, cursor: Optional[str]
) -> tuple[list[dict], Optional[str], bool]:
    """Fetch up to PROGRESS_CHUNK trades. Returns (trades, next_cursor, has_more)."""
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    use_cursor = None if cursor == "START" else cursor
    return client.fetch_trades_chunk(
        min_ts=min_ts, max_ts=max_ts, cursor=use_cursor, max_trades=PROGRESS_CHUNK
    )


def write_trades_to_csv(
    path: str, trades: list[dict], column_names: list[str], append: bool = False
) -> None:
    """Write flattened trades to CSV; if append, skip header."""
    rows = [_flatten_dict(t) for t in trades]
    if not rows:
        return
    mode = "a" if append else "w"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=column_names)
        if not append:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in column_names})


# -----------------------------------------------------------------------------
# UI: month selection
# -----------------------------------------------------------------------------

st.header("Data Extraction")

kalshi_start = datetime(2021, 7, 1)
today_month_start = datetime(datetime.now().year, datetime.now().month, 1)
months = []
cur = kalshi_start
while cur <= today_month_start:
    months.append(cur)
    cur = cur.replace(month=cur.month % 12 + 1, year=cur.year + (1 if cur.month == 12 else 0))

years = sorted({m.year for m in months})
selected_year = st.selectbox("Year", years, index=len(years) - 1, key="trade_year")
available_months_for_year = [m for m in months if m.year == selected_year]
month_options = [(m.month, m.strftime("%B")) for m in available_months_for_year]
selected_month_num = st.selectbox(
    "Month",
    [m[0] for m in month_options],
    index=len(month_options) - 1,
    format_func=lambda x: next(label for num, label in month_options if num == x),
    key="trade_month",
)
selected_month = next(m for m in available_months_for_year if m.month == selected_month_num)
if selected_month.month == 12:
    next_month = selected_month.replace(year=selected_month.year + 1, month=1)
else:
    next_month = selected_month.replace(month=selected_month.month + 1)
min_ts = int(selected_month.timestamp())
max_ts = int(next_month.timestamp())

st.caption(
    f"Extracting all trades for {selected_month.strftime('%B %Y')} (UTC). "
    f"Progress updates every {PROGRESS_CHUNK:,} trades; new CSV every {FILE_CHUNK:,} trades."
)

# -----------------------------------------------------------------------------
# Idle: start button
# -----------------------------------------------------------------------------

if st.session_state.fetch_phase == "idle":
    if st.button("Download CSV", type="primary"):
        st.session_state.fetch_phase = "fetching"
        st.session_state.fetch_cursor = "START"
        st.session_state.fetch_progress = 0
        st.session_state.csv_file_paths = []
        st.session_state.current_csv_path = None
        st.session_state.current_csv_row_count = 0
        st.session_state.current_csv_columns = None
        st.rerun()

# -----------------------------------------------------------------------------
# Fetching: progress and chunked fetch
# -----------------------------------------------------------------------------

if st.session_state.fetch_phase == "fetching":
    p = st.session_state.fetch_progress
    within_file = (p % FILE_CHUNK) / FILE_CHUNK if FILE_CHUNK else 0
    st.subheader("Progress")
    st.progress(within_file)
    st.caption(
        "Fetching trades... (updates every 100,000 trades; new file every 1,000,000)"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Trades fetched", f"{st.session_state.fetch_progress:,}")
    with col2:
        file_count = len(st.session_state.csv_file_paths)
        st.metric("CSV files created", file_count)
        if file_count > st.session_state.last_file_count:
            st.success(f"New file ready ({file_count} total)")
            st.session_state.last_file_count = file_count

    if st.session_state.csv_file_paths:
        st.divider()
        n = len(st.session_state.csv_file_paths)
        st.subheader(f"Download CSV Files ({n} file(s) ready for download)")
        st.caption("Files appear as they are created (every 1,000,000 trades).")
        for idx, (csv_path, file_row_count) in enumerate(st.session_state.csv_file_paths):
            if not os.path.exists(csv_path):
                continue
            try:
                with open(csv_path, "rb") as f:
                    csv_data = f.read()
            except Exception as e:
                st.error(f"Error reading file {idx + 1}: {e}")
                continue
            base = f"kalshi_trades_{selected_month.strftime('%Y%m')}"
            filename = f"{base}_part{idx + 1}.csv" if n > 1 else f"{base}.csv"
            st.download_button(
                f"Download Part {idx + 1} ({file_row_count:,} rows)",
                csv_data,
                file_name=filename,
                mime="text/csv",
                key=f"dl_fetching_{idx}",
            )

    try:
        trades, next_cursor, has_more = fetch_one_progress_chunk(
            min_ts, max_ts, st.session_state.fetch_cursor
        )
        if not trades:
            st.session_state.fetch_cursor = ""
            st.session_state.fetch_phase = "done"
            if st.session_state.current_csv_path and st.session_state.current_csv_row_count > 0:
                st.session_state.csv_file_paths.append(
                    (st.session_state.current_csv_path, st.session_state.current_csv_row_count)
                )
                st.session_state.current_csv_path = None
                st.session_state.current_csv_row_count = 0
            st.session_state.row_count = st.session_state.fetch_progress
            st.session_state.column_count = (
                len(st.session_state.current_csv_columns)
                if st.session_state.current_csv_columns
                else 0
            )
            st.rerun()
        else:
            sample = _flatten_dict(trades[0])
            column_names = st.session_state.current_csv_columns or sorted(sample.keys())
            st.session_state.current_csv_columns = column_names
            if st.session_state.current_csv_path is None:
                fd, path = tempfile.mkstemp(suffix=".csv", prefix="kalshi_trades_")
                os.close(fd)
                st.session_state.current_csv_path = path
                st.session_state.current_csv_row_count = 0
            append = st.session_state.current_csv_row_count > 0
            write_trades_to_csv(
                st.session_state.current_csv_path, trades, column_names, append=append
            )
            st.session_state.current_csv_row_count += len(trades)
            st.session_state.fetch_progress += len(trades)
            st.session_state.fetch_cursor = next_cursor or ""

            if st.session_state.current_csv_row_count >= FILE_CHUNK:
                completed = st.session_state.current_csv_path
                if completed and os.path.exists(completed) and os.path.getsize(completed) > 100:
                    st.session_state.csv_file_paths.append((completed, FILE_CHUNK))
                    st.session_state.last_file_count = len(st.session_state.csv_file_paths)
                st.session_state.current_csv_path = None
                st.session_state.current_csv_row_count = 0

            if not has_more and st.session_state.fetch_cursor == "":
                st.session_state.fetch_phase = "done"
                if st.session_state.current_csv_path and st.session_state.current_csv_row_count > 0:
                    st.session_state.csv_file_paths.append(
                        (
                            st.session_state.current_csv_path,
                            st.session_state.current_csv_row_count,
                        )
                    )
                    st.session_state.current_csv_path = None
                    st.session_state.current_csv_row_count = 0

            st.session_state.row_count = st.session_state.fetch_progress
            st.session_state.column_count = len(column_names)
            st.rerun()
    except Exception as e:
        st.session_state.fetch_phase = "idle"
        st.error(str(e))
        if "cryptography" in str(e).lower():
            st.code("pip install cryptography", language="bash")

# -----------------------------------------------------------------------------
# Done: summary and downloads
# -----------------------------------------------------------------------------

if st.session_state.fetch_phase == "done":
    st.metric("Total Rows", f"{st.session_state.row_count:,}")
    st.metric("Columns", st.session_state.column_count)
    st.metric("Number of Files", len(st.session_state.csv_file_paths))
    st.success(
        f"Done. Extracted {st.session_state.row_count:,} trades into "
        f"{len(st.session_state.csv_file_paths)} file(s)."
    )
    if not st.session_state.csv_file_paths:
        st.info("No files to download; no data was returned for this selection.")

if st.session_state.csv_file_paths and st.session_state.fetch_phase in ("fetching", "done"):
    st.divider()
    st.subheader("Download your CSV file(s)")
    for idx, (csv_path, file_row_count) in enumerate(st.session_state.csv_file_paths):
        if not os.path.exists(csv_path):
            continue
        try:
            with open(csv_path, "rb") as f:
                csv_data = f.read()
        except Exception:
            continue
        base = f"kalshi_trades_{selected_month.strftime('%Y%m')}"
        n = len(st.session_state.csv_file_paths)
        filename = f"{base}_part{idx + 1}.csv" if n > 1 else f"{base}.csv"
        st.download_button(
            f"Download Part {idx + 1} ({file_row_count:,} rows) — {filename}",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            key=f"dl_csv_{idx}",
        )

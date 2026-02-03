# Kalshi Trends

Data pipeline and pattern analysis for Kalshi prediction markets. Fetches market data from the same source used by [kalshidata.com](https://www.kalshidata.com) (Kalshi's public API), stores it locally, and runs trend/pattern analysis.

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Environment (optional)

```bash
cp .env.example .env
# Edit .env if you need to change rate limits
```

## Usage

### Fetch market data

```bash
python -m kalshi_trends fetch --days 30
```

Options:
- `--days 30` – Days of history for candlestick sampling (default: 30)
- `--no-cache` – Ignore cache and always fetch fresh data
- `--limit-events 100` – Limit number of events (0 = all)
- `--data-dir data` – Base directory for raw/cache/processed data

Fetched data is stored in:
- `data/cache/` – Cached API responses (JSON)
- `data/raw/` – Events, markets, candlesticks (CSV/Parquet)

### Run analysis

```bash
python -m kalshi_trends analyze --days 30
```

Options:
- `--days 30` – Analysis window (for future time-series features)
- `--data-dir data` – Input data directory
- `--output data/processed` – Output directory for report and charts

Outputs:
- `data/processed/report.md` – Findings summary and chart references
- `data/processed/charts/` – PNG charts (category dist, top movers, volume, clusters)
- `data/processed/*.csv` – Ranked tables (top_movers, most_volatile, highest_volume, clustering, pattern_ranked_markets)

## Analyses

| Analysis | Description |
|----------|-------------|
| **Trend snapshots** | Top movers (distance from 50¢), most volatile (spread/volume proxy), highest volume, highest open interest |
| **Category aggregates** | Average price, volume, and count by category |
| **Clustering** | K-means on last_price, volume, open_interest to group similar markets |
| **Pattern-ranked markets** | Markets with strongest signal metrics (move × volume) |
| **Time-to-expiry** | Days to expiration vs price/volume (when data available) |

## Project structure

```
kalshi_trends/
├── data_sources/
│   └── kalshidata_com.py   # KalshiData adapter (Kalshi public API)
├── storage/
│   ├── writer.py          # Write raw/processed data
│   └── reader.py          # Read cached data
├── analysis/
│   ├── trends.py          # Trend snapshots, top movers, volatility
│   └── patterns.py        # Clustering, momentum/mean-reversion
├── report.py              # Report + charts generation
├── cli.py                 # CLI entry point
└── __main__.py
```

## Data source

This pipeline uses **Kalshi's public API** (`api.elections.kalshi.com`), the same backend that powers [kalshidata.com](https://www.kalshidata.com). No API key is required for market data. Rate limits and retries with exponential backoff are applied.

## Live web dashboard

```bash
python -m kalshi_trends serve
```

Opens a Streamlit dashboard at http://localhost:8501 with charts and tables. Run `fetch` and `analyze` first to populate data.

Options: `--host 0.0.0.0` (default), `--port 8501`, `--data-dir data`

## Example commands

```bash
# Fetch last 30 days (with candlestick sampling for top markets)
python -m kalshi_trends fetch --days 30

# Fetch with limit for quick testing
python -m kalshi_trends fetch --days 7 --limit-events 50

# Force fresh fetch
python -m kalshi_trends fetch --days 30 --no-cache

# Run analysis and generate report
python -m kalshi_trends analyze --days 30

# Run live dashboard
python -m kalshi_trends serve
```

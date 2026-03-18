"""
Be the Maker — Kalshi limit order strategy.

Based on research by Bürgi, Deng & Whelan (UCD, 2025):
- Makers average ~10% loss vs Takers who average ~32% loss
- 22 percentage-point gap from order type alone
- Sweet spot: 50¢–80¢ contracts, 1–3 days to resolution
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Optional

import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Be the Maker — Kalshi Strategy", layout="wide")
st.title("Be the Maker")
st.caption(
    "Limit order strategy · 50¢–80¢ contracts · 1–3 days to resolution · "
    "Based on Bürgi, Deng & Whelan (UCD, 2025)"
)

# ─────────────────────────────────────────────────────────────────────────────
# Expanders
# ─────────────────────────────────────────────────────────────────────────────

with st.expander("Strategy Overview (Bürgi, Deng & Whelan — UCD 2025)", expanded=False):
    st.markdown("""
    **The Core Insight**

    On Kalshi, Makers (limit order posters) average ~10% loss while Takers average ~32% loss.
    That 22-percentage-point gap comes purely from order type.

    **The Four Rules**

    1. **Limit orders only** — Never use market orders. Post bids into the book; do not lift offers.
    2. **50¢–80¢ contracts** — The sweet spot. Contracts under 10¢ lose >60% on average; avoid them.
    3. **1–3 days to resolution** — Markets are more efficient near close; your edge is cleaner.
    4. **Avoid cheap longshots** — Contracts under 10¢ are the trap that wrecks most participants.

    **How to Be a Maker**

    Post a limit BUY order at `best_bid + 1¢`. This puts you at the front of the bid queue.
    Your order sits in the book until someone accepts your price — you are the maker, not the taker.

    **Caveat**

    The bias advantage appears to be shrinking as Kalshi markets mature over time.
    """)

with st.expander("Risk Disclaimer", expanded=False):
    st.markdown("""
    **Important**

    - Prediction market trading involves real financial risk, including total loss of capital
    - Past performance of any strategy does not guarantee future results
    - The maker/taker advantage is a statistical average across thousands of trades, not a per-trade guarantee
    - Limit orders may not fill if the market moves away before resolution
    - This tool is for informational purposes only and does not constitute financial advice
    - Always trade within your risk tolerance
    """)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

def _init_session():
    defaults = {
        "scan_results": None,
        "selected_ticker": "",
        "orderbook_data": None,
        "orderbook_ticker": "",
        "orders": None,
        "positions": None,
        "order_placed": False,
        "order_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ─────────────────────────────────────────────────────────────────────────────
# Client helpers
# ─────────────────────────────────────────────────────────────────────────────


def get_client():
    """Return (KalshiDataClient, None) or (None, error_message)."""
    try:
        from kalshi_trends.data_sources.kalshidata_com import KalshiDataClient
        rate_limit = float(os.environ.get("KALSHIDATA_RATE_LIMIT", "0.5"))
        return KalshiDataClient(rate_limit_delay=rate_limit), None
    except ImportError as e:
        return None, str(e)


def _has_auth() -> bool:
    key_id = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pk = os.environ.get("KALSHI_PRIVATE_KEY", "").strip()
    return bool(key_id and pk and "-----BEGIN" in pk)


# ─────────────────────────────────────────────────────────────────────────────
# Scanner helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_close_time(ct) -> Optional[datetime]:
    if not ct:
        return None
    try:
        if isinstance(ct, str):
            ct = ct.replace("Z", "+00:00")
            return datetime.fromisoformat(ct)
    except Exception:
        pass
    return None


def scan_markets(
    min_price: int,
    max_price: int,
    min_days: float,
    max_days: float,
    max_scan: int,
) -> pd.DataFrame:
    """Scan open markets and filter by price range and days to resolution."""
    client, err = get_client()
    if err:
        raise RuntimeError(err)

    now = datetime.now(timezone.utc)
    earliest_close = now + timedelta(days=min_days)
    latest_close = now + timedelta(days=max_days)

    rows = []
    scanned = 0

    for m in client.iter_all_markets(status="open"):
        scanned += 1
        if scanned > max_scan:
            break

        # Price: use last_price, fall back to mid of bid/ask
        last_price = m.get("last_price") or 0
        yes_bid = m.get("yes_bid") or 0
        yes_ask = m.get("yes_ask") or 0
        price = last_price if last_price else ((yes_bid + yes_ask) // 2 if yes_bid and yes_ask else 0)

        if not (min_price <= price <= max_price):
            continue

        close_time = _parse_close_time(m.get("close_time"))
        if close_time is None:
            continue
        if not (earliest_close <= close_time <= latest_close):
            continue

        hours_left = (close_time - now).total_seconds() / 3600

        rows.append({
            "ticker": m.get("ticker", ""),
            "title": (m.get("title") or m.get("subtitle") or "")[:80],
            "close_time": close_time.strftime("%Y-%m-%d %H:%M UTC"),
            "hours_left": round(hours_left, 1),
            "last_price": price,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "spread": (yes_ask - yes_bid) if (yes_bid and yes_ask) else None,
            "volume": m.get("volume") or 0,
            "open_interest": m.get("open_interest") or 0,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.sort_values("hours_left", ascending=True).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Order book helpers
# ─────────────────────────────────────────────────────────────────────────────


def fetch_orderbook(ticker: str, depth: int = 20) -> dict:
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    return client.get_market_orderbook(ticker, depth=depth)


def _parse_ob_side(side_data) -> list[tuple[int, int]]:
    """Parse orderbook side into list of (price_cents, quantity) sorted best first."""
    if not side_data:
        return []
    result = []
    for entry in side_data:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            result.append((int(entry[0]), int(entry[1])))
        elif isinstance(entry, dict):
            p = entry.get("price") or entry.get("price_cents") or 0
            q = entry.get("quantity") or entry.get("size") or 0
            result.append((int(p), int(q)))
    # Sort descending (best bid = highest price first)
    result.sort(key=lambda x: x[0], reverse=True)
    return result


def compute_maker_suggestion(yes_bids: list, no_bids: list) -> dict:
    """
    Compute key metrics and suggested maker price from order book.
    yes_bids / no_bids: list of (price_cents, qty) sorted best first.
    """
    metrics: dict = {
        "best_yes_bid": None,
        "best_yes_ask": None,
        "best_no_bid": None,
        "spread": None,
        "mid": None,
        "suggested_maker_price": None,
    }

    if yes_bids:
        metrics["best_yes_bid"] = yes_bids[0][0]

    if no_bids:
        metrics["best_no_bid"] = no_bids[0][0]
        # YES ask = complement of best NO bid
        metrics["best_yes_ask"] = 100 - no_bids[0][0]

    if metrics["best_yes_bid"] and metrics["best_yes_ask"]:
        metrics["spread"] = metrics["best_yes_ask"] - metrics["best_yes_bid"]
        metrics["mid"] = (metrics["best_yes_bid"] + metrics["best_yes_ask"]) / 2

    # Suggested maker price: 1¢ above best bid, must stay below best ask
    if metrics["best_yes_bid"] is not None:
        suggested = metrics["best_yes_bid"] + 1
        if metrics["best_yes_ask"] is None or suggested < metrics["best_yes_ask"]:
            metrics["suggested_maker_price"] = suggested
        else:
            metrics["suggested_maker_price"] = metrics["best_yes_bid"]

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio helpers
# ─────────────────────────────────────────────────────────────────────────────


def fetch_portfolio_orders(status: str = "resting") -> list[dict]:
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    data = client.get_portfolio_orders(status=status, limit=100)
    return data.get("orders", [])


def fetch_portfolio_positions() -> list[dict]:
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    data = client.get_portfolio_positions(limit=100)
    return data.get("market_positions", data.get("positions", []))


def cancel_order_by_id(order_id: str) -> dict:
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    return client.cancel_order(order_id)


def place_maker_order(ticker: str, side: str, count: int, price: int) -> dict:
    """Place a limit maker order. side: 'yes' or 'no'. price in cents (1–99)."""
    client, err = get_client()
    if err:
        raise RuntimeError(err)
    kwargs = {"yes_price": price} if side == "yes" else {"no_price": price}
    return client.create_order(
        ticker=ticker,
        action="buy",
        side=side,
        count=count,
        order_type="limit",
        **kwargs,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────

tab_scan, tab_book, tab_order, tab_monitor = st.tabs([
    "Market Scanner",
    "Order Book",
    "Place Order",
    "Portfolio Monitor",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: Market Scanner
# ═════════════════════════════════════════════════════════════════════════════

with tab_scan:
    st.subheader("Market Scanner")
    st.caption(
        "Finds open markets matching the strategy: 50¢–80¢ contracts resolving in 1–3 days."
    )

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.markdown("**Price range (¢)**")
        min_price = st.slider("Min price (¢)", 10, 89, 50, step=5, key="scan_min_price")
        max_price = st.slider(
            "Max price (¢)", min_price + 5, 95, 80, step=5, key="scan_max_price"
        )

    with col_f2:
        st.markdown("**Days to resolution**")
        min_days = st.number_input(
            "Min days", 0.0, 7.0, 1.0, step=0.5, key="scan_min_days"
        )
        max_days = st.number_input(
            "Max days", min_days + 0.5, 14.0, 3.0, step=0.5, key="scan_max_days"
        )

    with col_f3:
        st.markdown("**Scan limit**")
        max_scan = st.number_input(
            "Max markets to scan",
            100, 5000, 1000, step=100, key="scan_max_markets",
            help="Higher = more thorough but slower. Each page = 200 markets.",
        )

    if st.button("Scan Markets", type="primary", key="btn_scan"):
        with st.spinner(f"Scanning up to {int(max_scan):,} open markets..."):
            try:
                df = scan_markets(
                    min_price=int(min_price),
                    max_price=int(max_price),
                    min_days=float(min_days),
                    max_days=float(max_days),
                    max_scan=int(max_scan),
                )
                st.session_state.scan_results = df
            except Exception as e:
                st.error(f"Scan error: {e}")
                st.session_state.scan_results = None

    if st.session_state.scan_results is not None:
        df = st.session_state.scan_results

        if df.empty:
            st.info(
                "No markets matched the filters. "
                "Try widening the price range or extending the time window."
            )
        else:
            st.success(f"Found **{len(df)}** matching markets")

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Markets found", len(df))
            mc2.metric("Avg price (¢)", f"{df['last_price'].mean():.1f}")
            mc3.metric("Soonest close", f"{df['hours_left'].min():.1f}h")
            mc4.metric("Total volume", f"{df['volume'].sum():,}")

            st.dataframe(
                df[[
                    "ticker", "title", "hours_left", "last_price",
                    "yes_bid", "yes_ask", "spread", "volume", "open_interest",
                ]],
                use_container_width=True,
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "title": st.column_config.TextColumn("Market", width="large"),
                    "hours_left": st.column_config.NumberColumn("Hrs Left", format="%.1f"),
                    "last_price": st.column_config.NumberColumn("Price (¢)", format="%d"),
                    "yes_bid": st.column_config.NumberColumn("YES Bid", format="%d"),
                    "yes_ask": st.column_config.NumberColumn("YES Ask", format="%d"),
                    "spread": st.column_config.NumberColumn("Spread", format="%d"),
                    "volume": st.column_config.NumberColumn("Volume", format="%d"),
                    "open_interest": st.column_config.NumberColumn("Open Int", format="%d"),
                },
                hide_index=True,
            )

            st.divider()
            st.markdown("**Select a market to analyze in the Order Book tab:**")

            ticker_options = df["ticker"].tolist()

            def _label(t):
                row = df.loc[df["ticker"] == t]
                if row.empty:
                    return t
                title = row["title"].values[0][:55]
                price = row["last_price"].values[0]
                hrs = row["hours_left"].values[0]
                return f"{t} | {price}¢ | {hrs:.1f}h | {title}"

            selected = st.selectbox(
                "Market",
                ticker_options,
                format_func=_label,
                key="scanner_select",
            )

            if st.button("Open Order Book for this market", key="btn_open_book"):
                st.session_state.selected_ticker = selected
                st.session_state.orderbook_data = None
                st.info(
                    f"Switch to the **Order Book** tab — ticker `{selected}` has been pre-filled."
                )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: Order Book
# ═════════════════════════════════════════════════════════════════════════════

with tab_book:
    st.subheader("Order Book")
    st.caption(
        "View bid/ask depth for any market and compute the optimal maker price."
    )

    ob_ticker = st.text_input(
        "Market ticker",
        value=st.session_state.selected_ticker,
        placeholder="e.g. INXY-23-B5.5-B6",
        key="ob_ticker_input",
    )

    col_ob1, col_ob2 = st.columns([1, 4])
    with col_ob1:
        ob_depth = st.number_input("Depth levels", 3, 30, 10, key="ob_depth")

    if st.button("Fetch Order Book", type="primary", key="btn_fetch_ob"):
        if not ob_ticker.strip():
            st.warning("Enter a ticker first.")
        else:
            with st.spinner(f"Fetching order book for {ob_ticker.strip()}..."):
                try:
                    result = fetch_orderbook(ob_ticker.strip(), depth=int(ob_depth))
                    st.session_state.orderbook_data = result
                    st.session_state.orderbook_ticker = ob_ticker.strip()
                except Exception as e:
                    st.error(f"Error fetching order book: {e}")
                    st.session_state.orderbook_data = None

    if st.session_state.orderbook_data:
        ob = st.session_state.orderbook_data.get("orderbook", {})
        yes_bids = _parse_ob_side(ob.get("yes", []))
        no_bids = _parse_ob_side(ob.get("no", []))
        metrics = compute_maker_suggestion(yes_bids, no_bids)

        st.divider()

        # Key metrics row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(
            "Best YES Bid",
            f"{metrics['best_yes_bid']}¢" if metrics["best_yes_bid"] is not None else "—",
        )
        m2.metric(
            "Best YES Ask",
            f"{metrics['best_yes_ask']}¢" if metrics["best_yes_ask"] is not None else "—",
        )
        m3.metric(
            "Spread",
            f"{metrics['spread']}¢" if metrics["spread"] is not None else "—",
        )
        m4.metric(
            "Mid Price",
            f"{metrics['mid']:.1f}¢" if metrics["mid"] is not None else "—",
        )

        sug = metrics["suggested_maker_price"]
        delta_txt = None
        if sug and metrics["best_yes_bid"] and sug > metrics["best_yes_bid"]:
            delta_txt = "+1¢ above best bid"
        m5.metric("Suggested Maker Price", f"{sug}¢" if sug else "—", delta=delta_txt)

        # Order book tables
        st.divider()
        col_yes, col_no = st.columns(2)

        with col_yes:
            st.markdown("**YES Bids** — buyers willing to pay for YES")
            if yes_bids:
                yes_df = pd.DataFrame(
                    yes_bids[:int(ob_depth)], columns=["Price (¢)", "Quantity"]
                )
                yes_df["Cumulative Qty"] = yes_df["Quantity"].cumsum()
                st.dataframe(yes_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No YES bids in book")

        with col_no:
            st.markdown("**NO Bids** — buyers willing to pay for NO")
            if no_bids:
                no_df = pd.DataFrame(
                    no_bids[:int(ob_depth)], columns=["Price (¢)", "Quantity"]
                )
                no_df["Implied YES Ask"] = 100 - no_df["Price (¢)"]
                no_df["Cumulative Qty"] = no_df["Quantity"].cumsum()
                st.dataframe(no_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No NO bids in book")

        # Strategy suggestion callout
        if sug:
            st.divider()
            profit_per = 100 - sug
            yield_pct = (profit_per / sug) * 100
            st.info(
                f"**Maker suggestion for `{st.session_state.orderbook_ticker}`:** "
                f"Post a limit BUY YES at **{sug}¢** (1¢ above best bid of "
                f"{metrics['best_yes_bid']}¢). "
                f"If YES resolves: profit = **{profit_per}¢ per contract** "
                f"({yield_pct:.1f}% yield on cost). "
                f"Switch to **Place Order** to submit."
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: Place Order
# ═════════════════════════════════════════════════════════════════════════════

with tab_order:
    st.subheader("Place Maker Order")
    st.caption(
        "Post a limit order into the book — always making, never taking. "
        "Requires API credentials in `.env`."
    )

    if not _has_auth():
        st.warning(
            "API credentials not found. "
            "Set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY` in your `.env` file to place orders. "
            "You can still use Scanner and Order Book without credentials."
        )

    # Pre-fill from order book data
    _pre_ticker = st.session_state.orderbook_ticker or st.session_state.selected_ticker or ""
    _pre_price = 65
    if st.session_state.orderbook_data:
        ob = st.session_state.orderbook_data.get("orderbook", {})
        _yb = _parse_ob_side(ob.get("yes", []))
        _nb = _parse_ob_side(ob.get("no", []))
        _m = compute_maker_suggestion(_yb, _nb)
        if _m["suggested_maker_price"] is not None:
            _pre_price = _m["suggested_maker_price"]

    col_ord1, col_ord2 = st.columns(2)

    with col_ord1:
        st.markdown("**Order Parameters**")
        order_ticker = st.text_input(
            "Ticker",
            value=_pre_ticker,
            placeholder="e.g. INXY-23-B5.5-B6",
            key="order_ticker",
        )
        side = st.radio("Side", ["YES", "NO"], horizontal=True, key="order_side")
        count = st.number_input(
            "Contracts",
            min_value=1, max_value=10_000, value=10, step=1, key="order_count",
        )
        price = st.number_input(
            "Limit price (¢)",
            min_value=1, max_value=99, value=int(_pre_price), key="order_price",
            help=(
                "Price in cents (1–99). "
                "Set to best_bid + 1¢ to be a maker at the front of the queue."
            ),
        )

    with col_ord2:
        st.markdown("**Order Summary**")
        cost_per = price / 100
        profit_per = (100 - price) / 100
        total_cost = cost_per * count
        total_profit = profit_per * count
        yield_pct = (profit_per / cost_per) * 100 if cost_per > 0 else 0

        st.metric("Cost per contract", f"${cost_per:.2f}")
        st.metric("Max profit per contract", f"${profit_per:.2f}")
        st.metric("Total cost", f"${total_cost:.2f}")
        st.metric("Max total profit", f"${total_profit:.2f}")
        st.metric("Yield if YES wins", f"{yield_pct:.1f}%")

        st.info(
            f"This is a **LIMIT** order — it sits in the book as a maker order. "
            f"It fills only when someone accepts your {price}¢ price."
        )

    st.divider()

    confirm = st.checkbox(
        f"I confirm: place a LIMIT BUY {side} order — "
        f"{count} contracts of `{order_ticker or '[ticker]'}` at {price}¢ each "
        f"(total cost: ${total_cost:.2f})",
        key="order_confirm",
    )

    place_disabled = not (confirm and _has_auth() and order_ticker.strip())
    if st.button(
        "Place Maker Order",
        type="primary",
        key="btn_place_order",
        disabled=place_disabled,
    ):
        with st.spinner("Placing order..."):
            try:
                result = place_maker_order(
                    ticker=order_ticker.strip(),
                    side=side.lower(),
                    count=int(count),
                    price=int(price),
                )
                st.session_state.order_placed = True
                st.session_state.order_result = result
            except Exception as e:
                st.error(f"Order failed: {e}")
                st.session_state.order_placed = False
                st.session_state.order_result = None

    if place_disabled and confirm and not _has_auth():
        st.caption("Add API credentials to `.env` to enable order placement.")
    elif place_disabled and not confirm:
        st.caption("Check the confirmation box to enable the button.")

    if st.session_state.order_placed and st.session_state.order_result:
        raw = st.session_state.order_result
        order = raw.get("order", raw)
        order_id = order.get("order_id") or order.get("id", "N/A")
        status = order.get("status", "N/A")
        st.success(
            f"Order placed! ID: `{order_id}` | Status: `{status}` | "
            f"Check the **Portfolio Monitor** tab for updates."
        )
        if st.button("Dismiss", key="btn_dismiss_order"):
            st.session_state.order_placed = False
            st.session_state.order_result = None
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: Portfolio Monitor
# ═════════════════════════════════════════════════════════════════════════════

with tab_monitor:
    st.subheader("Portfolio Monitor")
    st.caption("Track your open orders and positions. Requires API credentials.")

    if not _has_auth():
        st.warning(
            "API credentials required. "
            "Set `KALSHI_API_KEY_ID` and `KALSHI_PRIVATE_KEY` in `.env`."
        )
    else:
        col_mon1, col_mon2 = st.columns(2)
        with col_mon1:
            if st.button("Refresh Orders", key="btn_refresh_orders"):
                with st.spinner("Loading resting orders..."):
                    try:
                        st.session_state.orders = fetch_portfolio_orders(status="resting")
                    except Exception as e:
                        st.error(f"Error loading orders: {e}")
                        st.session_state.orders = []

        with col_mon2:
            if st.button("Refresh Positions", key="btn_refresh_positions"):
                with st.spinner("Loading positions..."):
                    try:
                        st.session_state.positions = fetch_portfolio_positions()
                    except Exception as e:
                        st.error(f"Error loading positions: {e}")
                        st.session_state.positions = []

        # ── Open Orders ──────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Open (Resting) Orders")

        if st.session_state.orders is None:
            st.caption("Click **Refresh Orders** to load your open orders.")
        elif not st.session_state.orders:
            st.info("No resting orders found.")
        else:
            orders = st.session_state.orders
            order_rows = []
            for o in orders:
                yes_p = o.get("yes_price")
                no_p = o.get("no_price")
                price_val = yes_p if yes_p is not None else no_p
                order_rows.append({
                    "Order ID": o.get("order_id") or o.get("id", ""),
                    "Ticker": o.get("ticker", ""),
                    "Side": (o.get("side") or "").upper(),
                    "Action": (o.get("action") or "").upper(),
                    "Price (¢)": price_val,
                    "Remaining": o.get("remaining_count") or o.get("count") or 0,
                    "Status": o.get("status", ""),
                    "Created": (o.get("created_time") or "")[:19],
                })

            orders_df = pd.DataFrame(order_rows)
            st.dataframe(orders_df, use_container_width=True, hide_index=True)

            st.markdown("**Cancel a resting order:**")
            cancel_id = st.selectbox(
                "Select order ID to cancel",
                [r["Order ID"] for r in order_rows if r["Order ID"]],
                key="cancel_select",
            )
            if st.button("Cancel Order", type="secondary", key="btn_cancel"):
                with st.spinner(f"Cancelling order {cancel_id}..."):
                    try:
                        cancel_order_by_id(cancel_id)
                        st.success(f"Order `{cancel_id}` cancelled.")
                        st.session_state.orders = None
                        st.rerun()
                    except Exception as e:
                        st.error(f"Cancel failed: {e}")

        # ── Positions ─────────────────────────────────────────────────────────
        st.divider()
        st.markdown("### Current Positions")

        if st.session_state.positions is None:
            st.caption("Click **Refresh Positions** to load your current positions.")
        elif not st.session_state.positions:
            st.info("No positions found.")
        else:
            pos_rows = []
            for p in st.session_state.positions:
                ticker = p.get("ticker") or p.get("market_ticker", "")
                yes_pos = p.get("position") or p.get("yes_position") or 0
                total_traded = p.get("total_traded") or 0
                fees_paid = p.get("fees_paid") or 0
                resting = p.get("resting_orders_count") or 0
                pos_rows.append({
                    "Ticker": ticker,
                    "YES Position": yes_pos,
                    "Total Traded ($)": f"${total_traded / 100:.2f}" if total_traded else "—",
                    "Fees Paid ($)": f"${fees_paid / 100:.4f}" if fees_paid else "—",
                    "Resting Orders": resting,
                })

            pos_df = pd.DataFrame(pos_rows)
            st.dataframe(pos_df, use_container_width=True, hide_index=True)

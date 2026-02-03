"""
Bid/ask edge analysis for prediction markets (options-style).

Identifies edge from bid-ask spread and implied probability:
- Narrow spread + mispriced mid = potential edge
- Buy YES when implied (ask) < estimated fair value
- Sell YES when implied (bid) > estimated fair value
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_bid_ask_edge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bid/ask edge metrics for each market (options-style).

    Metrics:
    - mid: (yes_bid + yes_ask) / 2 = implied probability
    - spread: yes_ask - yes_bid = round-trip cost
    - spread_pct: spread / mid (or 100-mid) = relative cost
    - edge_score: inverse of spread, scaled by liquidity (higher = more tradeable)
    - buy_yes_edge: distance from 50 when mid < 50 (potential undervalued)
    - sell_yes_edge: distance from 50 when mid > 50 (potential overvalued)
    """
    required = ["yes_bid", "yes_ask"]
    for c in required:
        if c not in df.columns:
            return pd.DataFrame()

    out = df.copy()
    out["yes_bid"] = pd.to_numeric(out["yes_bid"], errors="coerce").fillna(0)
    out["yes_ask"] = pd.to_numeric(out["yes_ask"], errors="coerce").fillna(100)
    out["no_bid"] = pd.to_numeric(out["no_bid"], errors="coerce").fillna(0)
    out["no_ask"] = pd.to_numeric(out["no_ask"], errors="coerce").fillna(100)

    out["mid"] = (out["yes_bid"] + out["yes_ask"]) / 2
    out["spread"] = out["yes_ask"] - out["yes_bid"]
    out["spread_pct"] = out["spread"] / np.where(out["mid"] > 0, out["mid"], 1)
    out["spread_pct"] = out["spread_pct"].replace([np.inf, -np.inf], 0)

    # Edge score: lower spread + higher liquidity = better
    liq = out["liquidity"].fillna(0) if "liquidity" in out.columns else pd.Series(0, index=out.index)
    out["edge_score"] = (100 - out["spread"]) / 100 * np.log1p(liq)
    out["edge_score"] = out["edge_score"].fillna(0)

    # Potential edge: how far from "fair" 50
    out["buy_yes_edge"] = np.where(out["mid"] < 50, 50 - out["yes_ask"], 0)
    out["sell_yes_edge"] = np.where(out["mid"] > 50, out["yes_bid"] - 50, 0)
    out["max_edge"] = out[["buy_yes_edge", "sell_yes_edge"]].max(axis=1)

    return out


def rank_by_edge(df: pd.DataFrame, min_liquidity: float = 1000) -> pd.DataFrame:
    """Rank markets by edge score, filtering low liquidity."""
    edge_df = compute_bid_ask_edge(df)
    if edge_df.empty:
        return pd.DataFrame()
    if "liquidity" in edge_df.columns:
        edge_df = edge_df[edge_df["liquidity"] >= min_liquidity]
    edge_df = edge_df[edge_df["spread"] < 50]  # Exclude wide spreads
    edge_df = edge_df.sort_values("edge_score", ascending=False)
    return edge_df

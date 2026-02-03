"""
Pattern analysis from settled winning markets.

Finds common, consistent features across bets that win (YES or NO).
Uses settled market data to derive category-level and cross-sectional patterns.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_winner_patterns(settled_df: pd.DataFrame) -> dict:
    """
    Analyze settled markets to find common patterns among winners.

    Returns dict with:
    - winner_profile: avg price, spread, volume for YES vs NO winners
    - by_category: category-level win rates and typical price ranges
    - price_buckets: win rate by price bucket (0-25, 25-50, 50-75, 75-100)
    - summary_rules: human-readable rules derived from the data
    """
    if settled_df.empty or "result" not in settled_df.columns:
        return {}

    df = settled_df.copy()
    df["last_price"] = pd.to_numeric(df["last_price"], errors="coerce").fillna(50)
    df = df[(df["last_price"] > 0) & (df["last_price"] < 100)]

    out = {}

    # Winner profile: avg price for YES vs NO winners
    yes_winners = df[df["result"] == "yes"]
    no_winners = df[df["result"] == "no"]

    out["yes_winner_avg_price"] = yes_winners["last_price"].mean() if not yes_winners.empty else 50
    out["no_winner_avg_price"] = no_winners["last_price"].mean() if not no_winners.empty else 50
    out["yes_winner_median_price"] = yes_winners["last_price"].median() if not yes_winners.empty else 50
    out["no_winner_median_price"] = no_winners["last_price"].median() if not no_winners.empty else 50
    out["yes_count"] = len(yes_winners)
    out["no_count"] = len(no_winners)

    # Price buckets: win rate by implied probability bucket
    def _bucket(p):
        if p < 25:
            return "0-25"
        if p < 50:
            return "25-50"
        if p < 75:
            return "50-75"
        return "75-100"

    df["price_bucket"] = df["last_price"].apply(_bucket)
    bucket_stats = df.groupby("price_bucket").agg(
        total=("result", "count"),
        yes_wins=("result", lambda s: (s == "yes").sum()),
        no_wins=("result", lambda s: (s == "no").sum()),
        avg_price=("last_price", "mean"),
    ).round(2)
    bucket_stats["yes_win_rate"] = (bucket_stats["yes_wins"] / bucket_stats["total"]).round(2)
    bucket_stats["no_win_rate"] = (bucket_stats["no_wins"] / bucket_stats["total"]).round(2)
    out["price_buckets"] = bucket_stats

    # By category
    if "category" in df.columns:
        cat_stats = df.groupby("category").agg(
            total=("result", "count"),
            yes_wins=("result", lambda s: (s == "yes").sum()),
            no_wins=("result", lambda s: (s == "no").sum()),
            avg_price=("last_price", "mean"),
        ).round(2)
        cat_stats["yes_win_rate"] = (cat_stats["yes_wins"] / cat_stats["total"]).round(2)
        cat_stats["no_win_rate"] = (cat_stats["no_wins"] / cat_stats["total"]).round(2)
        out["by_category"] = cat_stats

    # Summary rules
    rules = []
    if out.get("yes_winner_avg_price") and out.get("no_winner_avg_price"):
        rules.append(
            f"YES winners: avg implied probability {out['yes_winner_avg_price']:.1f}c. "
            f"NO winners: avg {out['no_winner_avg_price']:.1f}c."
        )
    if "price_buckets" in out and not out["price_buckets"].empty:
        pb = out["price_buckets"]
        for idx in pb.index:
            row = pb.loc[idx]
            rules.append(
                f"Price {idx}c: YES win rate {row['yes_win_rate']:.0%}, NO win rate {row['no_win_rate']:.0%} "
                f"(n={int(row['total'])})"
            )
    if "by_category" in out and not out["by_category"].empty:
        cat = out["by_category"]
        for c in cat.index[:5]:
            row = cat.loc[c]
            rules.append(
                f"Category '{c}': YES {row['yes_win_rate']:.0%} / NO {row['no_win_rate']:.0%} "
                f"(n={int(row['total'])})"
            )
    out["summary_rules"] = rules

    # Simple strategy: derive actionable rules from price buckets
    strategy = []
    if "price_buckets" in out and not out["price_buckets"].empty:
        pb = out["price_buckets"]
        try:
            low = pb.loc["0-25"]
            if low["yes_win_rate"] > 0.5:
                strategy.append(f"Low prices (0-25c): YES wins {low['yes_win_rate']:.0%} of the time. Consider BUY YES when price is low.")
        except KeyError:
            pass
        try:
            high = pb.loc["75-100"]
            if high["no_win_rate"] > 0.5:
                strategy.append(f"High prices (75-100c): NO wins {high['no_win_rate']:.0%} of the time. Consider SELL YES (or BUY NO) when price is high.")
        except KeyError:
            pass
        if not strategy:
            strategy.append("Price buckets show mixed results. Focus on narrow spread + liquidity (edge_score) for best odds.")
    else:
        strategy.append("Run `python -m kalshi_trends backtest` to load settled data and derive a strategy.")
    out["simple_strategy"] = strategy

    return out

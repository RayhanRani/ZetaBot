"""Trend snapshots and top movers analysis."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """Compute trend snapshots: top movers, volatility, volume, category aggregates."""

    def __init__(self, df: pd.DataFrame):
        """
        df: DataFrame with columns like ticker, last_price, volume, volume_24h,
            open_interest, category, etc. Prices in cents (0-100) for probability.
        """
        self.df = df.copy()
        self._ensure_numeric()

    def _ensure_numeric(self) -> None:
        for col in ["last_price", "volume", "volume_24h", "open_interest", "liquidity"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0)

    def top_movers(
        self,
        price_col: str = "last_price",
        days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Rank markets by price change. For snapshot-only data, we use cross-sectional
        variation (e.g. price vs category median) as a proxy for "movers".
        With time series, we'd compute 1d/7d/30d changes.
        """
        if price_col not in self.df.columns:
            return pd.DataFrame()
        df = self.df[self.df[price_col].notna() & (self.df[price_col] > 0)].copy()
        if df.empty:
            return pd.DataFrame()
        # Use distance from 50 as proxy for "move" (50 = neutral)
        df["move_from_neutral"] = np.abs(df[price_col] - 50)
        df = df.sort_values("move_from_neutral", ascending=False).head(50)
        return df[["ticker", "title", "category", price_col, "move_from_neutral"]].dropna(axis=1, how="all")

    def most_volatile(
        self,
        window: int = 7,
        price_col: str = "last_price",
    ) -> pd.DataFrame:
        """
        Rank by rolling volatility. With snapshot data, use bid-ask spread or
        volume as volatility proxy. With time series, use rolling std.
        """
        if "yes_bid" in self.df.columns and "yes_ask" in self.df.columns:
            self.df["spread"] = self.df["yes_ask"] - self.df["yes_bid"]
            vol_col = "spread"
        else:
            vol_col = "volume_24h" if "volume_24h" in self.df.columns else "volume"
        if vol_col not in self.df.columns:
            return pd.DataFrame()
        df = self.df.copy()
        df = df[df[vol_col].notna() & (df[vol_col] > 0)]
        df = df.nlargest(50, vol_col)
        return df[["ticker", "title", "category", vol_col, price_col]].dropna(axis=1, how="all")

    def highest_volume(self, limit: int = 50) -> pd.DataFrame:
        """Rank by volume (or volume_24h)."""
        col = "volume_24h" if "volume_24h" in self.df.columns else "volume"
        if col not in self.df.columns:
            return pd.DataFrame()
        df = self.df.nlargest(limit, col)
        return df[["ticker", "title", "category", col, "last_price"]].dropna(axis=1, how="all")

    def highest_open_interest(self, limit: int = 50) -> pd.DataFrame:
        """Rank by open interest."""
        if "open_interest" not in self.df.columns:
            return pd.DataFrame()
        df = self.df.nlargest(limit, "open_interest")
        return df[["ticker", "title", "category", "open_interest", "last_price"]].dropna(axis=1, how="all")

    def category_aggregates(self) -> pd.DataFrame:
        """Aggregate by category: avg price, avg volatility proxy, count."""
        if "category" not in self.df.columns:
            return pd.DataFrame()
        df = self.df[self.df["category"].notna()].copy()
        if df.empty:
            return pd.DataFrame()
        vol_col = "volume_24h" if "volume_24h" in df.columns else "volume"
        agg = df.groupby("category").agg(
            count=("ticker", "count"),
            avg_price=("last_price", "mean"),
            avg_volume=("volume", "mean") if "volume" in df.columns else ("ticker", "count"),
        ).round(2)
        if "volume" in df.columns:
            agg["avg_volume"] = df.groupby("category")["volume"].mean().round(2)
        return agg.reset_index()

"""Pattern recognition: momentum, mean reversion, volatility regimes, clustering."""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def _hurst_proxy(series: pd.Series) -> float:
    """
    Simplified Hurst exponent proxy. H < 0.5 -> mean reversion, H > 0.5 -> momentum.
    Uses R/S analysis approximation.
    """
    if len(series) < 10:
        return 0.5
    series = series.dropna()
    if len(series) < 10:
        return 0.5
    series = series - series.mean()
    cumsum = series.cumsum()
    R = cumsum.max() - cumsum.min()
    S = series.std()
    if S == 0 or R == 0:
        return 0.5
    return 0.5 + np.log(R / S) / (2 * np.log(len(series)))

def _autocorr(series: pd.Series, lag: int = 1) -> float:
    """Lag-1 autocorrelation."""
    if len(series) < lag + 5:
        return 0.0
    s = series.dropna()
    if len(s) < lag + 5:
        return 0.0
    try:
        ac = s.autocorr(lag=lag)
        return float(ac) if pd.notna(ac) else 0.0
    except Exception:
        return 0.0


class PatternDetector:
    """
    Detect patterns: momentum vs mean reversion, volatility regimes, clustering.
    Works with time series (candlesticks) or snapshot data.
    """

    def __init__(self, df: pd.DataFrame, price_col: str = "close"):
        """
        df: DataFrame with price/time series. If from candlesticks, should have
            'end_period_ts' or 'close'. For snapshot data, we use cross-sectional
            features only.
        """
        self.df = df.copy()
        self.price_col = price_col
        if "close" in self.df.columns and price_col not in self.df.columns:
            self.price_col = "close"

    def momentum_vs_mean_reversion(
        self,
        ticker_col: str = "ticker",
    ) -> pd.DataFrame:
        """
        For each market, compute Hurst proxy and autocorrelation.
        Returns DataFrame with ticker, hurst_proxy, autocorr, signal (momentum/mean_reversion).
        """
        results = []
        if ticker_col not in self.df.columns:
            # Single series
            s = self.df[self.price_col] if self.price_col in self.df.columns else self.df.iloc[:, 0]
            h = _hurst_proxy(s)
            ac = _autocorr(s)
            return pd.DataFrame([{
                "hurst_proxy": h,
                "autocorr": ac,
                "signal": "momentum" if h > 0.5 else "mean_reversion",
            }])
        for ticker, grp in self.df.groupby(ticker_col):
            s = grp[self.price_col] if self.price_col in grp.columns else grp.iloc[:, 0]
            s = s.sort_index()
            h = _hurst_proxy(s)
            ac = _autocorr(s)
            results.append({
                "ticker": ticker,
                "hurst_proxy": h,
                "autocorr": ac,
                "signal": "momentum" if h > 0.5 else "mean_reversion",
            })
        return pd.DataFrame(results)

    def volatility_regime(
        self,
        window: int = 7,
        threshold_high: float = 1.5,
    ) -> pd.DataFrame:
        """
        Detect high/low volatility regimes using rolling std.
        Returns DataFrame with regime labels.
        """
        if self.price_col not in self.df.columns:
            return pd.DataFrame()
        df = self.df.copy()
        df["returns"] = df[self.price_col].pct_change()
        df["volatility"] = df["returns"].rolling(window, min_periods=1).std()
        df["volatility"] = df["volatility"].fillna(0)
        median_vol = df["volatility"].median()
        if median_vol == 0:
            median_vol = 1e-8
        df["vol_ratio"] = df["volatility"] / median_vol
        df["regime"] = df["vol_ratio"].apply(
            lambda x: "high" if x >= threshold_high else ("low" if x < 1 / threshold_high else "normal")
        )
        return df


def cluster_markets(
    df: pd.DataFrame,
    features: list[str],
    n_clusters: int = 5,
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Cluster markets by behavior (vol, trend slope, etc.).
    features: column names to use (e.g. ['last_price', 'volume', 'open_interest']).
    """
    available = [f for f in features if f in df.columns]
    if not available:
        return pd.DataFrame()
    X = df[available].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    out = df[[ticker_col] + available].copy() if ticker_col in df.columns else df[available].copy()
    out["cluster"] = labels
    return out


def time_to_expiry_effects(
    df: pd.DataFrame,
    expiry_col: str = "expiration_time",
    price_col: str = "last_price",
) -> pd.DataFrame:
    """
    Analyze if markets drift or get jumpy near expiration.
    Requires datetime expiry and time series of prices. For snapshot data,
    we compute days_to_expiry and correlate with price/volatility.
    """
    if expiry_col not in df.columns or price_col not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce")
    df = df[df[expiry_col].notna()]
    # Ensure both are tz-aware UTC for comparison
    if getattr(df[expiry_col].dtype, "tz", None) is None:
        df[expiry_col] = df[expiry_col].dt.tz_localize("UTC")
    else:
        df[expiry_col] = df[expiry_col].dt.tz_convert("UTC")
    now = pd.Timestamp.now(tz="UTC")
    df["days_to_expiry"] = (df[expiry_col] - now).dt.days
    df = df[df["days_to_expiry"] >= 0]  # Only future expirations
    if df.empty:
        return pd.DataFrame()
    return df[["ticker", "days_to_expiry", price_col, "volume"]].dropna(axis=1, how="all")

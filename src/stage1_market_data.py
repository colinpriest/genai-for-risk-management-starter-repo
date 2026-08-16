"""
Stage 1 — Market data

OWNER: solution author

Loads ASX 200 daily prices plus Australian macro indicators, and builds the volatility
features the regime model consumes.

WRITES data/processed/market_data.parquet
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import config

warnings.filterwarnings("ignore")


def _download(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
    # yfinance returns a MultiIndex column frame even for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df


def run() -> pd.DataFrame:
    px = _download(config.MARKET_TICKER, config.MARKET_START)
    if px.empty:
        raise RuntimeError(f"No data returned for {config.MARKET_TICKER}")

    df = pd.DataFrame(index=px.index)
    df["close"] = px["Close"]
    df["ret"] = np.log(df["close"]).diff()

    # --- volatility features (the regime model runs on these) --------------
    df["rv_5"] = df["ret"].rolling(5).std() * np.sqrt(252)
    df["rv_21"] = df["ret"].rolling(21).std() * np.sqrt(252)
    df["rv_63"] = df["ret"].rolling(63).std() * np.sqrt(252)
    # Parkinson high-low estimator: uses intraday range, less noisy than close-to-close
    hl = np.log(px["High"] / px["Low"])
    df["parkinson_21"] = np.sqrt((hl ** 2).rolling(21).mean() / (4 * np.log(2))) * np.sqrt(252)
    df["ret_abs"] = df["ret"].abs()
    # min_periods is essential: .where() leaves ~half the window NaN, and rolling().std()
    # defaults to requiring a full window, which silently returns an all-NaN column.
    df["downside_21"] = (df["ret"].where(df["ret"] < 0)
                         .rolling(21, min_periods=5).std() * np.sqrt(252))

    # --- macro / cross-asset context --------------------------------------
    aud = _download("AUDUSD=X", config.MARKET_START)
    if not aud.empty:
        df["aud_ret"] = np.log(aud["Close"]).diff().reindex(df.index)
        df["aud_vol_21"] = df["aud_ret"].rolling(21).std() * np.sqrt(252)

    vix = _download("^VIX", config.MARKET_START)
    if not vix.empty:
        df["vix"] = vix["Close"].reindex(df.index).ffill()

    df = df.dropna(subset=["ret"])
    df.index.name = "date"
    out = config.DATA_PROCESSED / "market_data.parquet"
    df.reset_index().to_parquet(out, index=False)
    print(f"  market_data: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")
    print(f"  columns: {list(df.columns)}")
    print(f"  missing per column:\n{df.isna().sum().to_string()}")
    return df


if __name__ == "__main__":
    run()

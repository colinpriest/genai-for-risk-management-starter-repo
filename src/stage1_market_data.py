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
    # =========================================================================================
    # YOUR CHOICE: which Australian macro / cross-asset series belong in the model.
    #
    # Add tickers to config.MACRO_TICKERS and they appear here as <name>_ret and <name>_vol_21.
    # Whatever you add must then be named in stage4.MACRO_FEATURES to actually enter the fit -
    # a series that is downloaded but not listed there is decoration.
    #
    # Two things to think about, both of which are marked:
    #   1. RELEASE TIMING. A daily market price is known the same day. An ABS statistic is not:
    #      CPI is published weeks after the quarter it describes. config.MACRO_LAG_DAYS shifts
    #      every macro series by that many days so the model only sees what was published.
    #      Daily market series (FX, VIX, commodities) need no lag - set it per series if they
    #      differ, and say what you did.
    #   2. WHY THIS SERIES. "It was available" is not a reason. Australia is a commodity
    #      exporter with a floating currency; argue from that.
    # =========================================================================================
    for name, ticker in config.MACRO_TICKERS.items():
        s = _download(ticker, config.MARKET_START)
        if s.empty:
            print(f"  WARNING: {name} ({ticker}) returned nothing - excluded")
            continue
        px_s = s["Close"].reindex(df.index).ffill()
        if config.MACRO_LAG_DAYS:
            px_s = px_s.shift(config.MACRO_LAG_DAYS)
        df[f"{name}"] = px_s
        df[f"{name}_ret"] = np.log(px_s).diff()
        df[f"{name}_vol_21"] = df[f"{name}_ret"].rolling(21, min_periods=5).std() * np.sqrt(252)

    # Kept under their original names because stage 4 and the dashboard refer to them.
    if "aud" in config.MACRO_TICKERS:
        df["aud_ret"] = df["aud_ret"]
        df["aud_vol_21"] = df["aud_vol_21"]

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

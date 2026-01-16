"""
TradingView link generation utilities.

Generates TradingView chart links for manual replay and analysis.
"""

from __future__ import annotations

import polars as pl


def add_tradingview_links(
    df: pl.DataFrame,
    exchange_prefix: str | None = None,
    ticker_col: str = "ticker",
) -> pl.DataFrame:
    """
    Add TradingView chart links to DataFrame.
    
    Adds columns:
      tv_1m: TradingView link for 1-minute chart
      tv_5m: TradingView link for 5-minute chart
    
    Args:
        df: DataFrame with ticker column
        exchange_prefix: Optional exchange prefix (e.g., "NASDAQ", "NYSE", "AMEX")
        ticker_col: Name of ticker column (default: "ticker")
    
    Returns:
        DataFrame with added tv_1m and tv_5m columns
    """
    sym = pl.col(ticker_col)
    if exchange_prefix:
        sym = pl.lit(exchange_prefix + ":") + sym
    
    base = pl.lit("https://www.tradingview.com/chart/?symbol=") + sym
    
    return df.with_columns([
        (base + pl.lit("&interval=1")).alias("tv_1m"),
        (base + pl.lit("&interval=5")).alias("tv_5m"),
    ])


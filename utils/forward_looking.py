"""
Forward-looking window utilities for computing future extremes.

Uses the "reverse trick" to efficiently compute forward-looking rolling windows
without loading all data into memory.
"""

from __future__ import annotations

import polars as pl


def add_forward_extremes(df: pl.DataFrame, y_bars: int) -> pl.DataFrame:
    """
    Adds forward-looking extreme values using the reverse trick.
    
    Adds columns:
      fwd_max_high_Y: max(high) over next y_bars (excluding current bar)
      fwd_min_low_Y:  min(low)  over next y_bars (excluding current bar)
    
    Requires df sorted by ticker, dt (or datetime).
    Uses the reverse trick: reverse -> rolling -> reverse to get forward-looking windows.
    
    Args:
        df: DataFrame with columns ticker, dt (or datetime), high, low
        y_bars: Number of bars to look forward
    
    Returns:
        DataFrame with added fwd_max_high_Y and fwd_min_low_Y columns
    """
    # Determine time column name
    time_col = "dt" if "dt" in df.columns else "datetime"
    
    # Ensure sorted
    df = df.sort(["ticker", time_col])
    
    # Reverse per ticker, rolling, reverse back
    high_rev = pl.col("high").reverse().over("ticker")
    low_rev = pl.col("low").reverse().over("ticker")
    
    # rolling_max on reversed series => future max in original
    fwd_max = (
        high_rev
        .rolling_max(window_size=y_bars, min_periods=y_bars)
        .reverse()
        .over("ticker")
    )
    fwd_min = (
        low_rev
        .rolling_min(window_size=y_bars, min_periods=y_bars)
        .reverse()
        .over("ticker")
    )
    
    # shift by -1 to exclude current bar (so "next Y minutes")
    return df.with_columns([
        fwd_max.shift(-1).over("ticker").alias(f"fwd_max_high_{y_bars}"),
        fwd_min.shift(-1).over("ticker").alias(f"fwd_min_low_{y_bars}"),
    ])


def label_move(
    df: pl.DataFrame,
    x: float,
    y_bars: int,
    entry_col: str = "close",
) -> pl.DataFrame:
    """
    Label moves based on forward-looking extremes.
    
    Adds columns:
      label_up_Xbp_Ym: binary label for upward move >= X
      label_dn_Xbp_Ym: binary label for downward move >= X
      fwd_up_ret_Ym: continuous return for upward move
      fwd_dn_ret_Ym: continuous return for downward move
    
    Args:
        df: DataFrame with forward extremes (from add_forward_extremes)
        x: Target move percentage (e.g., 0.008 for 0.8%)
        y_bars: Number of bars in forward window
        entry_col: Column name for entry price (default: "close")
    
    Returns:
        DataFrame with added label and return columns
    """
    entry = pl.col(entry_col)
    fwd_max_col = f"fwd_max_high_{y_bars}"
    fwd_min_col = f"fwd_min_low_{y_bars}"
    
    up_ret = (pl.col(fwd_max_col) / entry - 1.0)
    dn_ret = (pl.col(fwd_min_col) / entry - 1.0)
    
    label_suffix = f"{int(x*100)}bp_{y_bars}m"
    
    return df.with_columns([
        (up_ret >= x).cast(pl.Int8).alias(f"label_up_{label_suffix}"),
        (dn_ret <= -x).cast(pl.Int8).alias(f"label_dn_{label_suffix}"),
        up_ret.alias(f"fwd_up_ret_{y_bars}m"),
        dn_ret.alias(f"fwd_dn_ret_{y_bars}m"),
    ])


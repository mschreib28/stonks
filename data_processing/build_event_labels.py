#!/usr/bin/env python3
"""
Build event labels from MACD histogram cross events.

Detects MACD histogram cross events (up/down) on 1m and 5m timeframes
and labels whether they led to target moves in the next Y minutes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl

from build_macd_day_features_incremental import (
    add_macd,
    resample_5m,
    read_one_day_csv,
    filter_rth_et,
)
from utils.forward_looking import add_forward_extremes, label_move
from utils.tradingview_links import add_tradingview_links


def macd_cross_up_events(df_macd: pl.DataFrame) -> pl.DataFrame:
    """
    Extract MACD histogram cross-up events.
    
    Event: hist_prev <= 0 AND hist > 0
    
    Args:
        df_macd: DataFrame with MACD computed (columns: ticker, dt, hist)
    
    Returns:
        DataFrame filtered to cross-up events with event_type='cross_up'
    """
    df_macd = df_macd.sort(["ticker", "dt"]).with_columns(
        pl.col("hist").shift(1).over("ticker").alias("hist_prev")
    )
    
    events = df_macd.filter(
        (pl.col("hist_prev") <= 0) & (pl.col("hist") > 0)
    ).with_columns(
        pl.lit("cross_up").alias("event_type")
    )
    
    return events


def macd_cross_down_events(df_macd: pl.DataFrame) -> pl.DataFrame:
    """
    Extract MACD histogram cross-down events.
    
    Event: hist_prev >= 0 AND hist < 0
    
    Args:
        df_macd: DataFrame with MACD computed (columns: ticker, dt, hist)
    
    Returns:
        DataFrame filtered to cross-down events with event_type='cross_down'
    """
    df_macd = df_macd.sort(["ticker", "dt"]).with_columns(
        pl.col("hist").shift(1).over("ticker").alias("hist_prev")
    )
    
    events = df_macd.filter(
        (pl.col("hist_prev") >= 0) & (pl.col("hist") < 0)
    ).with_columns(
        pl.lit("cross_down").alias("event_type")
    )
    
    return events


def process_day_events(
    df_1m: pl.DataFrame,
    df_5m: pl.DataFrame,
    x: float,
    y_bars: int,
    include_up: bool = True,
    include_down: bool = True,
) -> pl.DataFrame:
    """
    Process events for a single day across 1m and 5m timeframes.
    
    Args:
        df_1m: 1-minute bars DataFrame
        df_5m: 5-minute bars DataFrame
        x: Target move percentage (e.g., 0.008 for 0.8%)
        y_bars: Number of bars to look forward
        include_up: Include cross-up events
        include_down: Include cross-down events
    
    Returns:
        DataFrame with event rows and labels
    """
    all_events = []
    
    # Process 1m timeframe
    if df_1m.height > 0:
        df_1m_macd = add_macd(df_1m)
        df_1m_macd = add_forward_extremes(df_1m_macd, y_bars)
        df_1m_macd = label_move(df_1m_macd, x, y_bars)
        
        events_1m = []
        if include_up:
            up_events = macd_cross_up_events(df_1m_macd)
            if up_events.height > 0:
                events_1m.append(up_events)
        if include_down:
            down_events = macd_cross_down_events(df_1m_macd)
            if down_events.height > 0:
                events_1m.append(down_events)
        
        if events_1m:
            events_1m_df = pl.concat(events_1m)
            events_1m_df = events_1m_df.with_columns(pl.lit("1m").alias("tf"))
            all_events.append(events_1m_df)
    
    # Process 5m timeframe
    if df_5m.height > 0:
        df_5m_macd = add_macd(df_5m)
        df_5m_macd = add_forward_extremes(df_5m_macd, y_bars)
        df_5m_macd = label_move(df_5m_macd, x, y_bars)
        
        events_5m = []
        if include_up:
            up_events = macd_cross_up_events(df_5m_macd)
            if up_events.height > 0:
                events_5m.append(up_events)
        if include_down:
            down_events = macd_cross_down_events(df_5m_macd)
            if down_events.height > 0:
                events_5m.append(down_events)
        
        if events_5m:
            events_5m_df = pl.concat(events_5m)
            events_5m_df = events_5m_df.with_columns(pl.lit("5m").alias("tf"))
            all_events.append(events_5m_df)
    
    if not all_events:
        # Return empty DataFrame with expected schema
        # Use dynamic column names based on parameters
        label_suffix = f"{int(x*100)}bp_{y_bars}m"
        return pl.DataFrame({
            "ticker": pl.Series([], dtype=pl.Utf8),
            "dt": pl.Series([], dtype=pl.Datetime),
            "date": pl.Series([], dtype=pl.Date),
            "tf": pl.Series([], dtype=pl.Utf8),
            "event_type": pl.Series([], dtype=pl.Utf8),
            "macd": pl.Series([], dtype=pl.Float64),
            "signal": pl.Series([], dtype=pl.Float64),
            "hist": pl.Series([], dtype=pl.Float64),
            "hist_prev": pl.Series([], dtype=pl.Float64),
            "entry_close": pl.Series([], dtype=pl.Float64),
            f"label_up_{label_suffix}": pl.Series([], dtype=pl.Int8),
            f"label_dn_{label_suffix}": pl.Series([], dtype=pl.Int8),
            f"fwd_up_ret_{y_bars}m": pl.Series([], dtype=pl.Float64),
            f"fwd_dn_ret_{y_bars}m": pl.Series([], dtype=pl.Float64),
            f"fwd_max_high_{y_bars}": pl.Series([], dtype=pl.Float64),
            f"fwd_min_low_{y_bars}": pl.Series([], dtype=pl.Float64),
            "tv_1m": pl.Series([], dtype=pl.Utf8),
            "tv_5m": pl.Series([], dtype=pl.Utf8),
        })
    
    # Combine all events
    events_df = pl.concat(all_events)
    
    # Select and rename columns for output
    events_df = events_df.with_columns(
        pl.col("close").alias("entry_close")
    ).select([
        "ticker",
        "dt",
        "date",
        "tf",
        "event_type",
        "macd",
        "signal",
        "hist",
        "hist_prev",
        "entry_close",
        f"label_up_{int(x*100)}bp_{y_bars}m",
        f"label_dn_{int(x*100)}bp_{y_bars}m",
        f"fwd_up_ret_{y_bars}m",
        f"fwd_dn_ret_{y_bars}m",
        f"fwd_max_high_{y_bars}",
        f"fwd_min_low_{y_bars}",
    ])
    
    # Add TradingView links
    events_df = add_tradingview_links(events_df)
    
    # Rename label columns to standard names for default case (80bp_30m)
    # This makes the output consistent with plan specifications
    label_suffix = f"{int(x*100)}bp_{y_bars}m"
    if x == 0.008 and y_bars == 30:
        events_df = events_df.rename({
            f"label_up_{label_suffix}": "label_up_80bp_30m",
            f"label_dn_{label_suffix}": "label_dn_80bp_30m",
        })
    
    return events_df


def write_day_events(df: pl.DataFrame, out_dir: Path, date_str: str) -> None:
    """Write events DataFrame to parquet file."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"events_{date_str}.parquet"
    df.write_parquet(out_path, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build event labels from MACD histogram cross events"
    )
    parser.add_argument(
        "--input-root",
        default="data/2025/polygon_minute_aggs",
        help="Root directory containing minute CSV files",
    )
    parser.add_argument(
        "--out-root",
        default="data/2025/cache/training",
        help="Output directory for event parquet files",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "rth"],
        default="all",
        help="Compute features using all sessions or RTH-only (ET)",
    )
    parser.add_argument(
        "--target-move",
        type=float,
        default=0.008,
        help="Target move percentage (default: 0.008 = 0.8%%)",
    )
    parser.add_argument(
        "--time-window",
        type=int,
        default=30,
        help="Forward-looking time window in bars (default: 30)",
    )
    parser.add_argument(
        "--long-only",
        action="store_true",
        help="Only include cross-up (long) events",
    )
    parser.add_argument(
        "--short-only",
        action="store_true",
        help="Only include cross-down (short) events",
    )
    
    args = parser.parse_args()
    
    if args.long_only and args.short_only:
        parser.error("Cannot specify both --long-only and --short-only")
    
    include_up = not args.short_only
    include_down = not args.long_only
    
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    files = sorted(input_root.rglob("*.csv.gz"))
    if not files:
        raise SystemExit(f"No csv.gz files found under {input_root}")
    
    for f in files:
        # Derive date string from filename: 2025-01-02.csv.gz -> 2025-01-02
        date_str = f.name.replace(".csv.gz", "")
        
        out_path = out_root / f"events_{date_str}.parquet"
        if out_path.exists():
            print(f"Skipping {f.name} (already processed)")
            continue
        
        # Read minute bars
        df_1m = read_one_day_csv(f)
        
        if args.mode == "rth":
            df_1m = filter_rth_et(df_1m)
        
        if df_1m.height == 0:
            print(f"Skipping {f.name} (no rows after filtering)")
            continue
        
        # Resample to 5m
        df_5m = resample_5m(df_1m)
        
        # Process events
        events_df = process_day_events(
            df_1m=df_1m,
            df_5m=df_5m,
            x=args.target_move,
            y_bars=args.time_window,
            include_up=include_up,
            include_down=include_down,
        )
        
        if events_df.height > 0:
            write_day_events(events_df, out_root, date_str)
            print(f"Wrote {events_df.height} events to {out_path}")
        else:
            print(f"No events found for {f.name}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
Build a reusable Parquet cache from Polygon/Massive flat files:
s3://flatfiles/us_stocks_sip/day_aggs_v1/YYYY/MM/YYYY-MM-DD.csv.gz

Assumes you've already downloaded files into:
  polygon_day_aggs_2025/01/2025-01-02.csv.gz
  polygon_day_aggs_2025/02/...
  ...

This script:
- scans all *.csv.gz with Polars (lazy)
- normalizes schema + types
- optionally adds daily returns and weekly returns
- writes parquet cache (single file or partitioned)

Tested patterns assume Polygon day aggs v1 columns: ticker, volume, open, close, high, low, window_start
(Some vendors use 'timestamp' or 't' instead of 'window_start' - adjust mapping below if needed.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ticker_filter import filter_tradable_tickers_df


def build_daily_lazy(input_glob: str) -> pl.LazyFrame:
    lf = pl.scan_csv(
        input_glob,
        has_header=True,
        ignore_errors=False,
    )

    schema_names = set(lf.collect_schema().names())

    # Map alternative column names if needed
    if "ticker" not in schema_names and "symbol" in schema_names:
        lf = lf.rename({"symbol": "ticker"})
        schema_names.add("ticker")

    if "window_start" not in schema_names:
        if "timestamp" in schema_names:
            lf = lf.rename({"timestamp": "window_start"})
            schema_names.add("window_start")
        elif "t" in schema_names:
            lf = lf.rename({"t": "window_start"})
            schema_names.add("window_start")

    # Type normalization (only for columns that exist)
    casts = []
    for col, dtype in [
        ("ticker", pl.Utf8),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),
        ("volume", pl.Int64),
    ]:
        if col in schema_names:
            casts.append(pl.col(col).cast(dtype))

    lf = lf.with_columns(casts)

    # window_start -> date (assumes epoch nanoseconds)
    if "window_start" in schema_names:
        lf = lf.with_columns(
            pl.from_epoch(pl.col("window_start"), time_unit="ns").dt.date().alias("date")
        )
        schema_names.add("date")

    # Keep only needed columns
    keep = [c for c in ["date", "ticker", "open", "high", "low", "close", "volume"] if c in schema_names]
    lf = lf.select(keep)

    # Partition keys
    lf = lf.with_columns(
        [
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.month().alias("month"),
        ]
    )

    return lf


def add_daily_returns(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds close-to-close daily return per ticker: (close / prev_close) - 1
    Requires ordering by date within ticker.
    """
    return lf.sort(["ticker", "date"]).with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("ticker") - 1.0).alias("ret_d")
    )


def add_computed_columns_daily(lf: pl.LazyFrame, price_min: float = 1.0, price_max: float = 9.99) -> pl.LazyFrame:
    """
    Adds computed columns for filtering:
    - close_in_band: boolean indicating if close price is within price range
    - abs_ret_d: absolute daily return percentage (|ret_d|)
    - abs_change_d: absolute daily price change in dollars (|close - prev_close|)
    """
    lf = lf.sort(["ticker", "date"])
    
    computed = [
        pl.col("close").is_between(price_min, price_max).alias("close_in_band"),
    ]
    
    # Add abs_ret_d if ret_d exists
    if "ret_d" in lf.collect_schema().names():
        computed.append(pl.col("ret_d").abs().alias("abs_ret_d"))
    
    # Add abs_change_d (always works, doesn't need ret_d)
    computed.append(
        (pl.col("close") - pl.col("close").shift(1).over("ticker")).abs().alias("abs_change_d")
    )
    
    return lf.with_columns(computed)


def build_weekly_returns(daily_lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Creates weekly close-to-close returns per ticker.
    Week is anchored to Monday by default in Polars for truncate("1w") behavior.
    If you want 'week ending Friday', you can shift dates or use a different grouping scheme.

    Output columns: week_start, ticker, close_w, ret_w, year, week
    """
    # Ensure sorted for stable "last" close selection
    lf = daily_lf.sort(["ticker", "date"])

    weekly = (
        lf.group_by(
            [
                pl.col("ticker"),
                pl.col("date").dt.truncate("1w").alias("week_start"),
            ]
        )
        .agg(
            [
                pl.col("close").last().alias("close_w"),
            ]
        )
        .sort(["ticker", "week_start"])
        .with_columns(
            (pl.col("close_w") / pl.col("close_w").shift(1).over("ticker") - 1.0).alias("ret_w"),
            pl.col("week_start").dt.year().alias("year"),
            pl.col("week_start").dt.week().alias("week"),
        )
    )
    return weekly


def add_computed_columns_weekly(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Adds computed columns to weekly dataset:
    - abs_ret_w: absolute weekly return percentage (|ret_w|)
    - abs_change_w: absolute weekly price change in dollars (|close_w - prev_close_w|)
    """
    lf = lf.sort(["ticker", "week_start"])
    
    computed = [
        pl.col("ret_w").abs().alias("abs_ret_w"),
        (pl.col("close_w") - pl.col("close_w").shift(1).over("ticker")).abs().alias("abs_change_w"),
    ]
    
    return lf.with_columns(computed)


def build_prefiltered_dataset(daily_lf: pl.LazyFrame, price_min: float, price_max: float) -> pl.LazyFrame:
    """
    Filters daily data to stocks where close is within the specified price range.
    Includes all computed columns.
    """
    return daily_lf.filter(
        pl.col("close").is_between(price_min, price_max)
    )


def build_minute_lazy(input_glob: str) -> pl.LazyFrame:
    """
    Loads and normalizes minute-by-minute data from Polygon CSV files.
    Similar to build_daily_lazy() but preserves time information.
    """
    lf = pl.scan_csv(
        input_glob,
        has_header=True,
        ignore_errors=False,
    )

    schema_names = set(lf.collect_schema().names())

    # Map alternative column names if needed
    if "ticker" not in schema_names and "symbol" in schema_names:
        lf = lf.rename({"symbol": "ticker"})
        schema_names.add("ticker")

    if "window_start" not in schema_names:
        if "timestamp" in schema_names:
            lf = lf.rename({"timestamp": "window_start"})
            schema_names.add("window_start")
        elif "t" in schema_names:
            lf = lf.rename({"t": "window_start"})
            schema_names.add("window_start")

    # Type normalization (only for columns that exist)
    casts = []
    for col, dtype in [
        ("ticker", pl.Utf8),
        ("open", pl.Float64),
        ("high", pl.Float64),
        ("low", pl.Float64),
        ("close", pl.Float64),
        ("volume", pl.Int64),
    ]:
        if col in schema_names:
            casts.append(pl.col(col).cast(dtype))

    lf = lf.with_columns(casts)

    # window_start -> datetime and date (assumes epoch nanoseconds for minute data)
    if "window_start" in schema_names:
        lf = lf.with_columns(
            [
                pl.from_epoch(pl.col("window_start"), time_unit="ns").alias("datetime"),
                pl.from_epoch(pl.col("window_start"), time_unit="ns").dt.date().alias("date"),
                pl.from_epoch(pl.col("window_start"), time_unit="ns").dt.time().alias("time"),
            ]
        )
        schema_names.update(["datetime", "date", "time"])

    # Keep needed columns
    keep = [c for c in ["date", "datetime", "time", "ticker", "open", "high", "low", "close", "volume"] if c in schema_names]
    lf = lf.select(keep)

    return lf


def build_minute_features(minute_lf: pl.LazyFrame, daily_lf: pl.LazyFrame, vol_lookback_days: int = 20, price_min: float = None, price_max: float = None) -> pl.LazyFrame:
    """
    Aggregates minute data to ticker-day level with computed features.
    Returns one row per ticker-day with all intraday features.
    
    Args:
        minute_lf: LazyFrame of minute-by-minute data
        daily_lf: LazyFrame of daily data (for volume spike and price filtering)
        vol_lookback_days: Number of days for volume spike calculation
        price_min: Optional minimum price filter (filters to ticker-dates in price range)
        price_max: Optional maximum price filter (filters to ticker-dates in price range)
    """
    # Filter minute data by price range if specified
    # Join with daily data to get close prices, then filter
    if price_min is not None and price_max is not None:
        # Get ticker-date pairs that are in the price range
        price_filtered_dates = (
            daily_lf.select(["ticker", "date", "close"])
            .filter(pl.col("close").is_between(price_min, price_max))
            .select(["ticker", "date"])
        )
        # Join to filter minute data to only ticker-dates in price range
        minute_lf = minute_lf.join(price_filtered_dates, on=["ticker", "date"], how="inner")
        print(f"Filtered minute data to price range ${price_min:.2f}-${price_max:.2f}")
    
    # Ensure sorted by ticker, date, datetime
    minute_lf = minute_lf.sort(["ticker", "date", "datetime"])
    
    # Calculate 1-minute and 5-minute returns (within same day)
    minute_lf = minute_lf.with_columns(
        [
            (pl.col("close") / pl.col("close").shift(1).over(["ticker", "date"]) - 1.0).alias("ret_1m"),
            (pl.col("close") / pl.col("close").shift(4).over(["ticker", "date"]) - 1.0).alias("ret_5m"),
        ]
    )
    
    # Calculate rolling 5-minute drawdown (rolling max within same day)
    # Use rolling_max with window_size=5 over ticker+date groups
    minute_lf = minute_lf.with_columns(
        [
            pl.col("close").rolling_max(window_size=5).over(["ticker", "date"]).alias("rolling_max_5m"),
        ]
    )
    minute_lf = minute_lf.with_columns(
        (
            (pl.col("close") - pl.col("rolling_max_5m")) / pl.col("rolling_max_5m")
        ).abs().alias("drawdown_5m")
    )
    
    # Get opening range (first 5 and 15 minutes)
    # Filter by hour and minute (9:30-9:35 and 9:30-9:45)
    opening_5m = (
        minute_lf.filter(
            (pl.col("time").dt.hour() == 9) & (pl.col("time").dt.minute() >= 30) & (pl.col("time").dt.minute() < 35)
        )
        .group_by(["ticker", "date"])
        .agg(
            [
                pl.col("high").max().alias("first_5m_high"),
                pl.col("low").min().alias("first_5m_low"),
            ]
        )
    )
    
    opening_15m = (
        minute_lf.filter(
            (pl.col("time").dt.hour() == 9) & (pl.col("time").dt.minute() >= 30) & (pl.col("time").dt.minute() < 45)
        )
        .group_by(["ticker", "date"])
        .agg(
            [
                pl.col("high").max().alias("first_15m_high"),
                pl.col("low").min().alias("first_15m_low"),
            ]
        )
    )
    
    # Aggregate to ticker-day level
    # Filter out nulls before max() to get actual max values (not null when all are null)
    daily_features = (
        minute_lf.group_by(["ticker", "date"])
        .agg(
            [
                pl.col("open").first().alias("day_open"),
                pl.col("high").max().alias("day_high"),
                pl.col("low").min().alias("day_low"),
                pl.col("close").last().alias("day_close"),
                pl.col("volume").sum().alias("day_volume"),
                # Filter nulls before max to avoid null results when some values exist
                pl.col("ret_1m").filter(pl.col("ret_1m").is_not_null()).max().alias("max_1m_ret"),
                pl.col("ret_5m").filter(pl.col("ret_5m").is_not_null()).max().alias("max_5m_ret"),
                pl.col("drawdown_5m").filter(pl.col("drawdown_5m").is_not_null()).max().alias("max_drawdown_5m"),
            ]
        )
        .with_columns(
            ((pl.col("day_high") - pl.col("day_low")) / pl.col("day_open")).alias("day_range_pct")
        )
    )
    
    # Join opening range data
    daily_features = daily_features.join(opening_5m, on=["ticker", "date"], how="left")
    daily_features = daily_features.join(opening_15m, on=["ticker", "date"], how="left")
    
    # Calculate opening range percentages and breakout flags
    # Handle nulls: if opening range data is missing, set flags to False and percentages to null
    daily_features = daily_features.with_columns(
        [
            pl.when(pl.col("first_5m_high").is_not_null() & pl.col("first_5m_low").is_not_null())
            .then((pl.col("first_5m_high") - pl.col("first_5m_low")) / pl.col("day_open"))
            .otherwise(None)
            .alias("range_5m_pct"),
            pl.when(pl.col("first_15m_high").is_not_null() & pl.col("first_15m_low").is_not_null())
            .then((pl.col("first_15m_high") - pl.col("first_15m_low")) / pl.col("day_open"))
            .otherwise(None)
            .alias("range_15m_pct"),
            pl.when(pl.col("first_5m_high").is_not_null())
            .then(pl.col("day_close") > pl.col("first_5m_high"))
            .otherwise(False)
            .alias("breakout_5m_up"),
            pl.when(pl.col("first_5m_low").is_not_null())
            .then(pl.col("day_close") < pl.col("first_5m_low"))
            .otherwise(False)
            .alias("breakout_5m_down"),
            pl.when(pl.col("first_15m_high").is_not_null())
            .then(pl.col("day_close") > pl.col("first_15m_high"))
            .otherwise(False)
            .alias("breakout_15m_up"),
            pl.when(pl.col("first_15m_low").is_not_null())
            .then(pl.col("day_close") < pl.col("first_15m_low"))
            .otherwise(False)
            .alias("breakout_15m_down"),
        ]
    )
    
    # Calculate volume spike (current day volume vs average of prior N days)
    # Get daily volume averages from daily_lf
    daily_vol = (
        daily_lf.select(["ticker", "date", "volume"])
        .sort(["ticker", "date"])
        .with_columns(
            pl.col("volume")
            .rolling_mean(window_size=vol_lookback_days)
            .over("ticker")
            .alias("avg_vol_prior")
        )
    )
    
    # Join volume data
    daily_features = daily_features.join(
        daily_vol.select(["ticker", "date", "avg_vol_prior"]),
        on=["ticker", "date"],
        how="left",
    )
    
    # Calculate volume spike
    # If avg_vol_prior is null (insufficient history), set vol_spike to null
    daily_features = daily_features.with_columns(
        pl.when(pl.col("avg_vol_prior").is_not_null() & (pl.col("avg_vol_prior") > 0))
        .then(pl.col("day_volume") / pl.col("avg_vol_prior"))
        .otherwise(None)
        .alias("vol_spike")
    )
    
    # Select final columns
    return daily_features.select(
        [
            "ticker",
            "date",
            "day_range_pct",
            "max_1m_ret",
            "max_5m_ret",
            "max_drawdown_5m",
            "vol_spike",
            "first_5m_high",
            "first_5m_low",
            "first_15m_high",
            "first_15m_low",
            "breakout_5m_up",
            "breakout_5m_down",
            "breakout_15m_up",
            "breakout_15m_down",
            "range_5m_pct",
            "range_15m_pct",
        ]
    )


def write_parquet(
    lf: pl.LazyFrame,
    out_path: Path,
    partitioned: bool,
    compression: str = "zstd",
    filter_non_tradable: bool = True,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = lf.collect(engine="streaming")  # <-- new API; replaces streaming=True

    # Filter out non-tradable tickers (warrants, units, rights, etc.)
    if filter_non_tradable and "ticker" in df.columns:
        df = filter_tradable_tickers_df(df)

    if partitioned:
        df.write_parquet(
            out_path,
            compression=compression,
            partition_by=["year", "month"],
            use_pyarrow=True,
        )
    else:
        df.write_parquet(out_path, compression=compression)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-root",
        default="data/2025/polygon_day_aggs",
        help="Folder containing month subfolders with *.csv.gz files",
    )
    parser.add_argument(
        "--out-daily",
        default="data/cache/daily_2025.parquet",
        help="Output Parquet path (file if --single-file, or directory if --partitioned)",
    )
    parser.add_argument(
        "--out-weekly",
        default="data/cache/weekly.parquet",
        help="Output weekly Parquet (optional if --build-weekly)",
    )
    parser.add_argument(
        "--partitioned",
        action="store_true",
        help="Write partitioned Parquet dataset (recommended). out paths become directories.",
    )
    parser.add_argument(
        "--add-returns",
        action="store_true",
        help="Add per-ticker daily returns column ret_d.",
    )
    parser.add_argument(
        "--build-weekly",
        action="store_true",
        help="Also build weekly close-to-close returns and write weekly Parquet.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "lz4", "uncompressed"],
        help="Parquet compression codec",
    )
    parser.add_argument(
        "--prefilter-price-range",
        nargs=2,
        type=float,
        default=[1.0, 9.99],
        metavar=("MIN", "MAX"),
        help="Price range for pre-filtered dataset (default: 1.0 9.99)",
    )
    parser.add_argument(
        "--out-prefiltered",
        default="data/cache/daily_filtered_1_9.99.parquet",
        help="Output path for pre-filtered dataset",
    )
    parser.add_argument(
        "--input-minute-root",
        default="data/2025/polygon_minute_aggs",
        help="Folder containing month subfolders with minute *.csv.gz files",
    )
    parser.add_argument(
        "--out-minute-features",
        default="data/cache/minute_features.parquet",
        help="Output path for minute features Parquet",
    )
    parser.add_argument(
        "--build-minute-features",
        action="store_true",
        help="Build minute-level feature cache aggregated to ticker-day level",
    )
    parser.add_argument(
        "--vol-spike-lookback",
        type=int,
        default=20,
        help="Number of days for volume spike comparison (default: 20)",
    )

    args = parser.parse_args()

    input_glob = str(Path(args.input_root) / "**" / "*.csv.gz")

    daily = build_daily_lazy(input_glob)

    if args.add_returns:
        daily = add_daily_returns(daily)

    # Add computed columns to daily dataset
    daily = add_computed_columns_daily(daily, price_min=args.prefilter_price_range[0], price_max=args.prefilter_price_range[1])

    out_daily = Path(args.out_daily)
    write_parquet(daily, out_daily, partitioned=args.partitioned, compression=args.compression)
    print(f"Wrote daily cache to: {out_daily} (partitioned={args.partitioned})")

    if args.build_weekly:
        weekly = build_weekly_returns(daily)
        weekly = add_computed_columns_weekly(weekly)
        out_weekly = Path(args.out_weekly)
        write_parquet(weekly, out_weekly, partitioned=args.partitioned, compression=args.compression)
        print(f"Wrote weekly cache to: {out_weekly} (partitioned={args.partitioned})")

    # Build pre-filtered dataset
    prefiltered = build_prefiltered_dataset(
        daily, price_min=args.prefilter_price_range[0], price_max=args.prefilter_price_range[1]
    )
    out_prefiltered = Path(args.out_prefiltered)
    write_parquet(prefiltered, out_prefiltered, partitioned=args.partitioned, compression=args.compression)
    print(f"Wrote pre-filtered cache to: {out_prefiltered} (partitioned={args.partitioned})")

    # Build minute features if requested
    if args.build_minute_features:
        minute_input_glob = str(Path(args.input_minute_root) / "**" / "*.csv.gz")
        minute_lf = build_minute_lazy(minute_input_glob)
        # Apply same price filter as pre-filtered dataset
        minute_features = build_minute_features(
            minute_lf, 
            daily, 
            vol_lookback_days=args.vol_spike_lookback,
            price_min=args.prefilter_price_range[0],
            price_max=args.prefilter_price_range[1]
        )
        out_minute_features = Path(args.out_minute_features)
        write_parquet(minute_features, out_minute_features, partitioned=args.partitioned, compression=args.compression)
        print(f"Wrote minute features cache to: {out_minute_features} (partitioned={args.partitioned})")


if __name__ == "__main__":
    main()


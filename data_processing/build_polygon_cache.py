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
from typing import Optional, Tuple, List
from multiprocessing import Pool, cpu_count
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


def build_prefiltered_dataset(
    daily_lf: pl.LazyFrame, 
    price_min: float, 
    price_max: float,
    min_avg_volume: float | None = None,
    min_days: int | None = None,
) -> pl.LazyFrame:
    """
    Filters daily data to stocks that meet the specified criteria.
    
    Args:
        daily_lf: LazyFrame of daily data
        price_min: Minimum price filter
        price_max: Maximum price filter
        min_avg_volume: Optional minimum average daily volume (filters out tickers with low volume)
        min_days: Optional minimum number of days of data required (filters out tickers with insufficient history)
    
    Returns:
        Filtered LazyFrame with only tickers that meet all criteria
    """
    # If we need to filter by volume or days, calculate per-ticker statistics on ALL data first
    # This ensures avg_volume is calculated across all days, not just price-filtered days
    if min_avg_volume is not None or min_days is not None:
        # Calculate per-ticker statistics on all data
        ticker_stats = (
            daily_lf
            .group_by("ticker")
            .agg([
                pl.col("volume").mean().alias("avg_volume"),
                pl.len().alias("day_count"),
            ])
        )
        
        # Build filter conditions
        conditions = []
        if min_avg_volume is not None:
            conditions.append(pl.col("avg_volume") >= min_avg_volume)
        if min_days is not None:
            conditions.append(pl.col("day_count") >= min_days)
        
        # Filter tickers that meet criteria
        if conditions:
            ticker_filter = ticker_stats.filter(pl.all_horizontal(conditions))
            valid_tickers = ticker_filter.select("ticker")
            
            # First filter by valid tickers, then by price range
            filtered = daily_lf.join(valid_tickers, on="ticker", how="inner")
            filtered = filtered.filter(
                pl.col("close").is_between(price_min, price_max)
            )
        else:
            # No volume/days filter, just filter by price
            filtered = daily_lf.filter(
                pl.col("close").is_between(price_min, price_max)
            )
    else:
        # No volume/days filter, just filter by price
        filtered = daily_lf.filter(
            pl.col("close").is_between(price_min, price_max)
        )
    
    return filtered


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


def process_minute_batch(
    args: Tuple[
        List[str],  # batch_dates
        int,  # batch_num
        str,  # minute_input_glob
        str,  # daily_path (path to daily parquet for reloading)
        int,  # vol_lookback_days
        float,  # price_min
        float,  # price_max
        Path,  # temp_dir
        str,  # compression
        bool,  # partitioned
    ]
) -> Optional[Path]:
    """
    Process a single batch of minute features.
    This function is designed to be called in parallel.
    
    Returns:
        Path to the temporary batch file if successful, None otherwise
    """
    (
        batch_dates,
        batch_num,
        minute_input_glob,
        daily_path,
        vol_lookback_days,
        price_min,
        price_max,
        temp_dir,
        compression,
        partitioned,
    ) = args
    
    try:
        # Reload daily data in worker (LazyFrame can't be pickled)
        daily_lf = pl.scan_parquet(daily_path) if daily_path else None
        
        # Load minute data for this batch
        minute_lf = build_minute_lazy(minute_input_glob)
        batch_minute_lf = minute_lf.filter(pl.col("date").is_in(batch_dates))
        
        # Process this batch
        batch_features = build_minute_features(
            batch_minute_lf,
            daily_lf,
            vol_lookback_days=vol_lookback_days,
            price_min=price_min,
            price_max=price_max
        )
        
        # Collect using streaming engine
        batch_df = batch_features.collect(streaming=True)
        
        if len(batch_df) > 0:
            # Add year/month columns if needed for partitioning
            if partitioned and "year" not in batch_df.columns:
                batch_df = batch_df.with_columns([
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                ])
            
            # Write batch to temporary file
            batch_file = temp_dir / f"batch_{batch_num:04d}.parquet"
            batch_df.write_parquet(batch_file, compression=compression)
            return batch_file
        else:
            return None
    except Exception as e:
        print(f"  ⚠️  Error processing batch {batch_num}: {e}")
        return None


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
        default="data",
        help="Root folder containing year subfolders (e.g., data/) or specific year folder (e.g., data/2025/polygon_day_aggs). "
             "If root folder, will search for **/*.csv.gz across all year directories.",
    )
    parser.add_argument(
        "--out-daily",
        default="data/cache/daily_all.parquet",
        help="Output Parquet path (file if --single-file, or directory if --partitioned). "
             "Defaults to daily_all.parquet for multi-year processing.",
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
        "--prefilter-min-avg-volume",
        type=float,
        default=750000,
        help="Minimum average daily volume for pre-filtered dataset (default: 750000, matching Swing Trading Default preset). Use 0 to disable.",
    )
    parser.add_argument(
        "--prefilter-min-days",
        type=int,
        default=60,
        help="Minimum number of days of data required for pre-filtered dataset (default: 60, matching Swing Trading Default preset). Use 0 to disable.",
    )
    parser.add_argument(
        "--input-minute-root",
        default="data",
        help="Root folder containing year subfolders (e.g., data/) or specific year folder (e.g., data/2025/polygon_minute_aggs). "
             "If root folder, will search for **/*.csv.gz across all year directories.",
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers for minute features batch processing (default: auto-detect, use 1 for sequential)",
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
    # Convert 0 values to None to disable filters
    min_avg_volume = args.prefilter_min_avg_volume if args.prefilter_min_avg_volume and args.prefilter_min_avg_volume > 0 else None
    min_days = args.prefilter_min_days if args.prefilter_min_days and args.prefilter_min_days > 0 else None
    
    prefiltered = build_prefiltered_dataset(
        daily, 
        price_min=args.prefilter_price_range[0], 
        price_max=args.prefilter_price_range[1],
        min_avg_volume=min_avg_volume,
        min_days=min_days,
    )
    out_prefiltered = Path(args.out_prefiltered)
    write_parquet(prefiltered, out_prefiltered, partitioned=args.partitioned, compression=args.compression)
    
    # Print filter summary
    filter_summary = [f"price ${args.prefilter_price_range[0]:.2f}-${args.prefilter_price_range[1]:.2f}"]
    if min_avg_volume is not None:
        filter_summary.append(f"min avg volume {min_avg_volume:,.0f}")
    if min_days is not None:
        filter_summary.append(f"min {min_days} days")
    print(f"Wrote pre-filtered cache to: {out_prefiltered} (partitioned={args.partitioned})")
    print(f"  Filters: {', '.join(filter_summary)}")

    # Build minute features if requested
    if args.build_minute_features:
        minute_input_glob = str(Path(args.input_minute_root) / "**" / "*.csv.gz")
        out_minute_features = Path(args.out_minute_features)
        
        # Check if we should use batch processing for memory efficiency
        # Process in date batches to reduce memory usage
        use_batch_processing = True  # Always use batch processing for minute features
        
        if use_batch_processing:
            print("Processing minute features in batches to reduce memory usage...")
            minute_lf = build_minute_lazy(minute_input_glob)
            
            # Get unique dates from minute data using streaming to avoid loading all data
            print("Scanning dates from minute data...")
            dates_df = minute_lf.select("date").unique().sort("date").collect(streaming=True)
            unique_dates = [str(row[0]) for row in dates_df.iter_rows()]
            print(f"Found {len(unique_dates)} unique dates to process")
            
            # Process in smaller batches to keep memory manageable
            batch_size = 10  # Reduced from 30 to process smaller chunks
            
            # Create temporary directory for batch files
            import tempfile
            import shutil
            temp_dir = Path(tempfile.mkdtemp(prefix="minute_features_batch_"))
            
            # Determine number of workers
            n_jobs = args.n_jobs
            if n_jobs is None:
                n_jobs = max(1, cpu_count() - 1)  # Leave one core free
            elif n_jobs == 1:
                n_jobs = 1  # Sequential processing
            
            print(f"Using {n_jobs} parallel worker(s) for batch processing")
            
            # Prepare batch arguments
            batch_args = []
            for i in range(0, len(unique_dates), batch_size):
                batch_dates = unique_dates[i:i+batch_size]
                batch_num = i//batch_size + 1
                batch_args.append((
                    batch_dates,
                    batch_num,
                    minute_input_glob,
                    str(Path(args.out_daily)),  # Path to daily data for reloading in workers
                    args.vol_spike_lookback,
                    args.prefilter_price_range[0],
                    args.prefilter_price_range[1],
                    temp_dir,
                    args.compression,
                    args.partitioned,
                ))
            
            batch_files = []
            try:
                if n_jobs == 1:
                    # Sequential processing (fallback)
                    for batch_arg in batch_args:
                        batch_dates, batch_num = batch_arg[0], batch_arg[1]
                        total_batches = len(batch_args)
                        print(f"Processing batch {batch_num}/{total_batches}: {batch_dates[0]} to {batch_dates[-1]}")
                        
                        batch_file = process_minute_batch(batch_arg)
                        if batch_file is not None:
                            batch_files.append(batch_file)
                            print(f"  Wrote batch {batch_num} to temp file")
                else:
                    # Parallel processing
                    total_batches = len(batch_args)
                    with Pool(processes=n_jobs) as pool:
                        results = pool.imap_unordered(process_minute_batch, batch_args)
                        processed = 0
                        for batch_file in results:
                            processed += 1
                            if batch_file is not None:
                                batch_files.append(batch_file)
                                print(f"  Completed batch {processed}/{total_batches}")
                            else:
                                print(f"  ⚠️  Batch {processed}/{total_batches} produced no data")
                    
                    # Sort batch files by batch number for consistent ordering
                    batch_files.sort(key=lambda p: int(p.stem.split('_')[1]))
                
                # Combine all batch files using lazy concatenation
                print(f"Combining {len(batch_files)} batches...")
                if batch_files:
                    # Read all batch files lazily and concatenate
                    batch_lfs = [pl.scan_parquet(str(f)) for f in batch_files]
                    combined_lf = pl.concat(batch_lfs)
                    
                    # Write final output
                    write_parquet(
                        combined_lf,
                        out_minute_features,
                        partitioned=args.partitioned,
                        compression=args.compression
                    )
                    print(f"Wrote minute features cache to: {out_minute_features} (partitioned={args.partitioned})")
                else:
                    print("⚠️  No minute features generated")
            finally:
                # Clean up temporary files
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                    print("Cleaned up temporary batch files")
        else:
            # Original single-pass processing (uses more memory)
            minute_lf = build_minute_lazy(minute_input_glob)
            minute_features = build_minute_features(
                minute_lf, 
                daily, 
                vol_lookback_days=args.vol_spike_lookback,
                price_min=args.prefilter_price_range[0],
                price_max=args.prefilter_price_range[1]
            )
            write_parquet(minute_features, out_minute_features, partitioned=args.partitioned, compression=args.compression)
            print(f"Wrote minute features cache to: {out_minute_features} (partitioned={args.partitioned})")


if __name__ == "__main__":
    main()


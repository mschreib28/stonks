#!/usr/bin/env python3
"""
Build a raw minute-level OHLCV cache for low-price stocks.

This script:
1. Identifies tickers with average close price <= $10 (configurable)
2. Loads raw minute data from polygon_minute_aggs CSV files
3. Filters to only include qualifying tickers
4. Writes a Parquet cache with minute-level OHLCV bars

The resulting cache can be used for intraday chart display at various resolutions
(15m, 1h, 2h, etc.) by aggregating at query time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime
import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.ticker_filter import filter_tradable_tickers_df


def get_low_price_tickers(
    daily_path: str | Path,
    max_price: float = 10.0,
    min_avg_volume: int = 100000,
) -> set[str]:
    """
    Get set of tickers with average close price <= max_price and sufficient volume.
    
    Args:
        daily_path: Path to daily parquet file
        max_price: Maximum average close price to include
        min_avg_volume: Minimum average daily volume
    
    Returns:
        Set of qualifying ticker symbols
    """
    print(f"Loading daily data to identify low-price tickers (max ${max_price:.2f})...")
    
    lf = pl.scan_parquet(str(daily_path))
    
    # Group by ticker and compute stats
    ticker_stats = (
        lf.group_by("ticker")
        .agg([
            pl.col("close").mean().alias("avg_close"),
            pl.col("close").last().alias("last_close"),
            pl.col("volume").mean().alias("avg_volume"),
            pl.len().alias("day_count"),
        ])
        .filter(
            (pl.col("avg_close") <= max_price) &
            (pl.col("avg_volume") >= min_avg_volume)
        )
        .collect()
    )
    
    tickers = set(ticker_stats["ticker"].to_list())
    print(f"  Found {len(tickers):,} tickers with avg price <= ${max_price:.2f} and avg volume >= {min_avg_volume:,}")
    
    return tickers


def build_minute_ohlcv_lazy(input_glob: str) -> pl.LazyFrame:
    """
    Load raw minute data from CSV files.
    
    Args:
        input_glob: Glob pattern for CSV.gz files
    
    Returns:
        LazyFrame with minute bars
    """
    lf = pl.scan_csv(
        input_glob,
        has_header=True,
        ignore_errors=False,
    )
    
    schema_names = set(lf.collect_schema().names())
    
    # Map alternative column names
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
    
    # Type normalization
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
    
    # Convert window_start (nanoseconds) to datetime
    if "window_start" in schema_names:
        lf = lf.with_columns([
            pl.from_epoch(pl.col("window_start"), time_unit="ns").alias("datetime"),
            pl.from_epoch(pl.col("window_start"), time_unit="ns").dt.date().alias("date"),
        ])
    
    # Select final columns
    keep = ["datetime", "date", "ticker", "open", "high", "low", "close", "volume"]
    keep = [c for c in keep if c in lf.collect_schema().names()]
    lf = lf.select(keep)
    
    return lf


def process_minute_data_batched(
    input_glob: str,
    output_path: Path,
    valid_tickers: set[str],
    batch_days: int = 30,
    compression: str = "zstd",
) -> int:
    """
    Process minute data in batches to manage memory.
    
    Args:
        input_glob: Glob pattern for minute CSV files
        output_path: Output parquet path
        valid_tickers: Set of tickers to include
        batch_days: Number of days to process per batch
        compression: Parquet compression codec
    
    Returns:
        Total number of rows written
    """
    import tempfile
    import shutil
    
    print(f"Scanning minute data from: {input_glob}")
    
    # Load lazy frame
    lf = build_minute_ohlcv_lazy(input_glob)
    
    # Get unique dates
    print("Getting unique dates...")
    dates_df = lf.select("date").unique().sort("date").collect(streaming=True)
    all_dates = dates_df["date"].to_list()
    print(f"  Found {len(all_dates)} unique dates")
    
    if not all_dates:
        print("  No dates found!")
        return 0
    
    # Create temp directory for batch files
    temp_dir = Path(tempfile.mkdtemp(prefix="minute_ohlcv_batch_"))
    batch_files = []
    total_rows = 0
    
    try:
        # Process in batches
        for i in range(0, len(all_dates), batch_days):
            batch_dates = all_dates[i:i + batch_days]
            batch_num = i // batch_days + 1
            total_batches = (len(all_dates) + batch_days - 1) // batch_days
            
            print(f"  Processing batch {batch_num}/{total_batches}: {batch_dates[0]} to {batch_dates[-1]}")
            
            # Filter to batch dates and valid tickers
            batch_lf = lf.filter(
                pl.col("date").is_in(batch_dates) &
                pl.col("ticker").is_in(list(valid_tickers))
            )
            
            # Collect
            batch_df = batch_lf.collect(streaming=True)
            
            if len(batch_df) > 0:
                # Filter non-tradable tickers
                batch_df = filter_tradable_tickers_df(batch_df)
                
                # Write batch file
                batch_file = temp_dir / f"batch_{batch_num:04d}.parquet"
                batch_df.write_parquet(batch_file, compression=compression)
                batch_files.append(batch_file)
                total_rows += len(batch_df)
                print(f"    Wrote {len(batch_df):,} rows")
            else:
                print(f"    No data for this batch")
        
        # Combine all batch files
        if batch_files:
            print(f"Combining {len(batch_files)} batch files...")
            batch_lfs = [pl.scan_parquet(str(f)) for f in batch_files]
            combined_lf = pl.concat(batch_lfs)
            
            # Sort by ticker, datetime for efficient queries
            combined_lf = combined_lf.sort(["ticker", "datetime"])
            
            # Write final output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            combined_df = combined_lf.collect(streaming=True)
            combined_df.write_parquet(output_path, compression=compression)
            print(f"Wrote {total_rows:,} rows to {output_path}")
        else:
            print("No data to write!")
            
    finally:
        # Cleanup temp files
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary files")
    
    return total_rows


def main():
    parser = argparse.ArgumentParser(
        description="Build minute-level OHLCV cache for low-price stocks"
    )
    parser.add_argument(
        "--input-root",
        default="data",
        help="Root folder containing year subfolders with polygon_minute_aggs",
    )
    parser.add_argument(
        "--daily-path",
        default="data/cache/daily_all.parquet",
        help="Path to daily parquet for determining qualifying tickers",
    )
    parser.add_argument(
        "--output",
        default="data/cache/minute_ohlcv.parquet",
        help="Output parquet path",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=10.0,
        help="Maximum average price to include (default: $10)",
    )
    parser.add_argument(
        "--min-avg-volume",
        type=int,
        default=100000,
        help="Minimum average daily volume (default: 100,000)",
    )
    parser.add_argument(
        "--batch-days",
        type=int,
        default=30,
        help="Days to process per batch (default: 30)",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        choices=["zstd", "snappy", "gzip", "lz4", "uncompressed"],
        help="Parquet compression codec (default: zstd)",
    )
    
    args = parser.parse_args()
    
    # Find daily parquet
    daily_path = Path(args.daily_path)
    if not daily_path.exists():
        # Try to find it
        cache_dir = Path("data/cache")
        candidates = list(cache_dir.glob("daily_*.parquet"))
        candidates = [c for c in candidates if "filtered" not in c.name and "scored" not in c.name]
        if candidates:
            daily_path = candidates[0]
            print(f"Using daily data: {daily_path}")
        else:
            print(f"ERROR: Daily parquet not found at {args.daily_path}")
            print("Run data processing first to generate daily cache.")
            sys.exit(1)
    
    # Get qualifying tickers
    valid_tickers = get_low_price_tickers(
        daily_path,
        max_price=args.max_price,
        min_avg_volume=args.min_avg_volume,
    )
    
    if not valid_tickers:
        print("No qualifying tickers found!")
        sys.exit(1)
    
    # Build input glob for minute data
    input_glob = str(Path(args.input_root) / "**/polygon_minute_aggs/**/*.csv.gz")
    
    # Check if any minute files exist
    from glob import glob
    minute_files = glob(input_glob, recursive=True)
    if not minute_files:
        print(f"No minute data files found matching: {input_glob}")
        sys.exit(1)
    print(f"Found {len(minute_files)} minute data files")
    
    # Process and write
    output_path = Path(args.output)
    total_rows = process_minute_data_batched(
        input_glob,
        output_path,
        valid_tickers,
        batch_days=args.batch_days,
        compression=args.compression,
    )
    
    if total_rows > 0:
        # Print stats
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\nSummary:")
        print(f"  Total rows: {total_rows:,}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  Tickers: {len(valid_tickers):,}")
    else:
        print("\nNo data was written.")


if __name__ == "__main__":
    main()

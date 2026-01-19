#!/usr/bin/env python3
"""
Build a filtered dataset based on scoring criteria from the daily dataset.

This script:
1. Loads the full daily dataset
2. Scores all tickers using default scoring criteria
3. Filters to only tickers that meet minimum quality thresholds
4. Writes a filtered parquet file with only high-quality tickers
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
from typing import Dict, Any, List, Tuple, Optional
from multiprocessing import Pool, cpu_count

# Import scoring functions from build_scoring_index
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.build_scoring_index import (
    compute_ticker_metrics,
    score_ticker,
    DEFAULT_CRITERIA,
    DEFAULT_MONTHS_BACK,
    DEFAULT_MIN_DAYS,
    DEFAULT_MIN_AVG_VOLUME,
)


def process_single_ticker(
    args: Tuple[
        str,  # ticker
        str,  # daily_path
        Dict[str, float],  # most_recent_prices
        List[Dict[str, Any]],  # criteria
        int,  # months_back
        int,  # min_days
        int,  # min_avg_volume
        float,  # min_score
        float,  # min_daily_range
        float,  # min_price
        float,  # max_price
    ]
) -> Optional[Dict[str, Any]]:
    """
    Process a single ticker and return its score and metrics if it meets criteria.
    This function is designed to be called in parallel.
    
    Returns:
        Dict with ticker, score, and metrics if ticker meets criteria, None otherwise
    """
    (
        ticker,
        daily_path,
        most_recent_prices,
        criteria,
        months_back,
        min_days,
        min_avg_volume,
        min_score,
        min_daily_range,
        min_price,
        max_price,
    ) = args
    
    # Load daily data and filter to this ticker
    daily_lf = pl.scan_parquet(daily_path)
    ticker_lf = daily_lf.filter(pl.col("ticker") == ticker)
    
    # Check if enough data
    ticker_count = ticker_lf.select(pl.len().alias("count")).collect()["count"][0]
    if ticker_count < min_days:
        return None
    
    # Materialize for metrics computation
    ticker_df = ticker_lf.collect()
    
    # Compute metrics
    metrics = compute_ticker_metrics(ticker_df, months_back)
    if not metrics:
        return None
    
    # Update current_price
    if ticker in most_recent_prices:
        metrics["current_price"] = most_recent_prices[ticker]
    
    # Apply price filter
    current_price = metrics.get("current_price", 0)
    if current_price < min_price or current_price > max_price:
        return None
    
    # Apply minimum average volume filter
    if min_avg_volume > 0:
        avg_vol = metrics.get("avg_volume", 0)
        if avg_vol < min_avg_volume:
            return None
    
    # Apply minimum daily range filter (critical for swing trading)
    avg_range = metrics.get("avg_daily_range_dollars", 0)
    if avg_range < min_daily_range:
        return None
    
    # Score the ticker
    total_score, criterion_scores = score_ticker(metrics, criteria)
    
    # Apply minimum score filter
    if total_score < min_score:
        return None
    
    return {
        "ticker": ticker,
        "score": total_score,
        "metrics": metrics,
    }


def build_filtered_dataset_from_scoring(
    daily_path: Path,
    output_path: Path,
    criteria: List[Dict[str, Any]] = None,
    months_back: int = DEFAULT_MONTHS_BACK,
    min_days: int = DEFAULT_MIN_DAYS,
    min_avg_volume: int = DEFAULT_MIN_AVG_VOLUME,
    min_score: float = 0.0,
    min_daily_range: float = 0.05,  # Minimum average daily range in dollars
    min_price: float = 0.0,
    max_price: float = 8.0,
    max_tickers: int = None,  # Limit to top N tickers by score
    force: bool = False,
    n_jobs: Optional[int] = None,  # Number of parallel workers
) -> None:
    """
    Build a filtered dataset containing only tickers that meet scoring criteria.
    
    Args:
        daily_path: Path to daily parquet file
        output_path: Path to write filtered parquet file
        criteria: Scoring criteria (defaults to DEFAULT_CRITERIA)
        months_back: Months of data to analyze
        min_days: Minimum days of data required
        min_avg_volume: Minimum average daily volume
        min_score: Minimum total score to include
        min_daily_range: Minimum average daily range in dollars (filters out flat stocks)
        min_price: Minimum current price
        max_price: Maximum current price
        max_tickers: Limit to top N tickers by score (None = no limit)
        force: Overwrite existing file
        n_jobs: Number of parallel workers (None = auto-detect, 1 = sequential)
    """
    if output_path.exists() and not force:
        print(f"Output file exists: {output_path}")
        print("Use --force to overwrite")
        return
    
    if criteria is None:
        criteria = DEFAULT_CRITERIA
    
    print(f"Loading daily dataset from: {daily_path}")
    daily_lf = pl.scan_parquet(daily_path)
    
    # Get unique tickers
    print("Getting unique tickers...")
    tickers_df = daily_lf.select("ticker").unique().collect()
    tickers = tickers_df["ticker"].to_list()
    print(f"Found {len(tickers)} unique tickers")
    
    # Get most recent prices for each ticker
    print("Getting most recent prices...")
    most_recent_prices = {}
    ticker_prices = (
        daily_lf.sort("date", descending=True)
        .group_by("ticker")
        .agg([
            pl.col("close").first().alias("latest_close")
        ])
        .collect()
    )
    for row in ticker_prices.iter_rows(named=True):
        if row["latest_close"] is not None:
            most_recent_prices[row["ticker"]] = float(row["latest_close"])
    
    print(f"Got prices for {len(most_recent_prices)} tickers")
    
    # Score each ticker (parallel processing)
    print("\nScoring tickers...")
    
    # Determine number of workers
    if n_jobs is None:
        n_jobs = max(1, cpu_count() - 1)  # Leave one core free
    elif n_jobs == 1:
        n_jobs = 1  # Sequential processing
    
    print(f"  Using {n_jobs} parallel worker(s)")
    
    # Prepare arguments for each ticker
    ticker_args = [
        (
            ticker,
            str(daily_path),  # Convert Path to string for multiprocessing
            most_recent_prices,
            criteria,
            months_back,
            min_days,
            min_avg_volume,
            min_score,
            min_daily_range,
            min_price,
            max_price,
        )
        for ticker in tickers
    ]
    
    # Process tickers in parallel or sequentially
    scored_tickers = []
    if n_jobs == 1:
        # Sequential processing (fallback)
        processed = 0
        for args in ticker_args:
            processed += 1
            if processed % 100 == 0:
                print(f"  Processed {processed}/{len(tickers)} tickers...")
            
            result = process_single_ticker(args)
            if result is not None:
                scored_tickers.append(result)
    else:
        # Parallel processing
        with Pool(processes=n_jobs) as pool:
            # Use imap_unordered for progress tracking
            results = pool.imap_unordered(process_single_ticker, ticker_args)
            processed = 0
            for result in results:
                processed += 1
                if processed % 100 == 0:
                    print(f"  Processed {processed}/{len(tickers)} tickers...")
                
                if result is not None:
                    scored_tickers.append(result)
    
    print(f"\nScored {len(scored_tickers)} tickers that meet criteria")
    
    # Sort by score (descending)
    scored_tickers.sort(key=lambda x: x["score"], reverse=True)
    
    # Limit to top N if specified
    if max_tickers is not None and len(scored_tickers) > max_tickers:
        print(f"Limiting to top {max_tickers} tickers by score")
        scored_tickers = scored_tickers[:max_tickers]
    
    # Get list of valid tickers
    valid_tickers = [t["ticker"] for t in scored_tickers]
    print(f"Filtering dataset to {len(valid_tickers)} tickers")
    
    # Filter daily dataset to only these tickers
    filtered_lf = daily_lf.filter(pl.col("ticker").is_in(valid_tickers))
    
    # Write filtered dataset
    print(f"\nWriting filtered dataset to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Collect and write
    filtered_df = filtered_lf.collect()
    filtered_df.write_parquet(output_path, compression="zstd")
    
    print(f"✓ Wrote filtered dataset:")
    print(f"  Rows: {filtered_df.height:,}")
    print(f"  Columns: {len(filtered_df.columns)}")
    print(f"  Unique tickers: {filtered_df['ticker'].n_unique()}")
    print(f"  Date range: {filtered_df['date'].min()} to {filtered_df['date'].max()}")
    
    # Print summary of top tickers
    print(f"\nTop 10 tickers by score:")
    for i, ticker_data in enumerate(scored_tickers[:10], 1):
        ticker = ticker_data["ticker"]
        score = ticker_data["score"]
        metrics = ticker_data["metrics"]
        price = metrics.get("current_price", 0)
        avg_range = metrics.get("avg_daily_range_dollars", 0)
        liquidity = metrics.get("liquidity_multiple", 0)
        print(f"  {i}. {ticker}: score={score:.2f}, price=${price:.2f}, range=${avg_range:.3f}, liquidity={liquidity:.1f}x")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build filtered dataset from scoring results"
    )
    parser.add_argument(
        "--daily-path",
        type=Path,
        default=None,  # Will be resolved dynamically
        help="Path to daily parquet file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cache/daily_filtered_scored.parquet"),
        help="Output path for filtered parquet file",
    )
    parser.add_argument(
        "--min-daily-range",
        type=float,
        default=0.05,
        help="Minimum average daily range in dollars (default: 0.05)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum total score to include (default: 0.0)",
    )
    parser.add_argument(
        "--max-tickers",
        type=int,
        default=None,
        help="Limit to top N tickers by score (default: no limit)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=0.0,
        help="Minimum current price (default: 0.0)",
    )
    parser.add_argument(
        "--max-price",
        type=float,
        default=8.0,
        help="Maximum current price (default: 8.0)",
    )
    parser.add_argument(
        "--min-days",
        type=int,
        default=DEFAULT_MIN_DAYS,
        help=f"Minimum days of data required (default: {DEFAULT_MIN_DAYS})",
    )
    parser.add_argument(
        "--min-avg-volume",
        type=int,
        default=DEFAULT_MIN_AVG_VOLUME,
        help=f"Minimum average daily volume (default: {DEFAULT_MIN_AVG_VOLUME:,})",
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=DEFAULT_MONTHS_BACK,
        help=f"Months of data to analyze (default: {DEFAULT_MONTHS_BACK})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect, use 1 for sequential)",
    )
    
    args = parser.parse_args()
    
    # Resolve daily path dynamically if not provided
    if args.daily_path is None:
        cache_dir = Path("data/cache")
        # Try to find daily dataset file
        daily_files = list(cache_dir.glob("daily_*.parquet"))
        if not daily_files:
            print(f"Error: No daily dataset found in {cache_dir}")
            print("Please run: python data_processing/process_all_data.py to generate the cache files.")
            return
        
        # Prefer "daily_all.parquet" or most recent
        preferred = [f for f in daily_files if f.name == "daily_all.parquet"]
        if preferred:
            args.daily_path = preferred[0]
        else:
            # Use the most recent file (by modification time)
            args.daily_path = max(daily_files, key=lambda p: p.stat().st_mtime)
        
        print(f"Using daily dataset: {args.daily_path}")
    
    if not args.daily_path.exists():
        print(f"Error: Daily dataset not found: {args.daily_path}")
        print("Please run: python data_processing/process_all_data.py to generate the cache files.")
        return
    
    build_filtered_dataset_from_scoring(
        daily_path=args.daily_path,
        output_path=args.output,
        criteria=DEFAULT_CRITERIA,
        months_back=args.months_back,
        min_days=args.min_days,
        min_avg_volume=args.min_avg_volume,
        min_score=args.min_score,
        min_daily_range=args.min_daily_range,
        min_price=args.min_price,
        max_price=args.max_price,
        max_tickers=args.max_tickers,
        force=args.force,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()

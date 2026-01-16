#!/usr/bin/env python3
"""
Build ML-ready training dataset from event labels and day-level context.

Combines event features with MACD day summaries and volatility context
to create a complete feature set for model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def load_events(events_glob: str) -> pl.LazyFrame:
    """Load all event parquet files as a lazy frame."""
    return pl.scan_parquet(events_glob)


def add_event_features(df: pl.LazyFrame) -> pl.LazyFrame:
    """
    Add feature engineering at event time.
    
    Adds:
    - MACD momentum: hist - hist_prev, abs(hist)
    - MACD strength: abs(hist)
    """
    return df.with_columns([
        (pl.col("hist") - pl.col("hist_prev")).alias("hist_momentum"),
        pl.col("hist").abs().alias("hist_abs"),
        pl.col("macd").abs().alias("macd_abs"),
        pl.col("signal").abs().alias("signal_abs"),
    ])


def join_macd_day_features(events: pl.LazyFrame, macd_glob: str) -> pl.LazyFrame:
    """
    Join MACD day-level summary features.
    
    Adds features like:
    - hist_last__1m, hist_last__5m
    - hist_range__1m (hist_max - hist_min)
    - hist_zero_cross_up_count__1m, etc.
    """
    macd_day = pl.scan_parquet(macd_glob)
    
    # Compute hist_range (hist_max - hist_min) if not present
    macd_day = macd_day.with_columns([
        (pl.col("hist_max__1m") - pl.col("hist_min__1m")).alias("hist_range__1m"),
        (pl.col("hist_max__5m") - pl.col("hist_min__5m")).alias("hist_range__5m"),
    ])
    
    # Join on ticker and date
    return events.join(
        macd_day,
        on=["ticker", "date"],
        how="left",
    )


def join_volatility_features(events: pl.LazyFrame, vol_path: str) -> pl.LazyFrame:
    """
    Join minute-level volatility features.
    
    Adds features like:
    - day_range_pct, range_15m_pct
    - vol_spike, breakout_5m_up, etc.
    """
    vol_features = pl.scan_parquet(vol_path)
    
    # Join on ticker and date
    return events.join(
        vol_features,
        on=["ticker", "date"],
        how="left",
    )


def filter_price_range(
    events: pl.LazyFrame,
    daily_path: str,
    price_min: float = 1.0,
    price_max: float = 9.99,
) -> pl.LazyFrame:
    """
    Filter events to only tickers in the specified price range.
    
    Uses daily cache to determine which ticker-dates are in range.
    """
    daily = pl.scan_parquet(daily_path)
    
    # Get ticker-date pairs in price range
    price_filtered = (
        daily
        .select(["ticker", "date", "close"])
        .filter(pl.col("close").is_between(price_min, price_max))
        .select(["ticker", "date"])
    )
    
    # Join to filter events
    return events.join(
        price_filtered,
        on=["ticker", "date"],
        how="inner",
    )


def build_training_dataset(
    events_glob: str,
    macd_glob: str,
    vol_path: str,
    daily_path: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    output_path: str | None = None,
) -> pl.DataFrame:
    """
    Build complete ML-ready training dataset.
    
    Args:
        events_glob: Glob pattern for event parquet files
        macd_glob: Glob pattern for MACD day features
        vol_path: Path to volatility features parquet
        daily_path: Optional path to daily cache (for price filtering)
        price_min: Optional minimum price filter
        price_max: Optional maximum price filter
        output_path: Optional output path
    
    Returns:
        Complete training dataset DataFrame
    """
    # Load events
    events = load_events(events_glob)
    
    # Add event-level features
    events = add_event_features(events)
    
    # Join MACD day features
    events = join_macd_day_features(events, macd_glob)
    
    # Join volatility features
    events = join_volatility_features(events, vol_path)
    
    # Filter by price range if specified
    if daily_path and price_min is not None and price_max is not None:
        events = filter_price_range(events, daily_path, price_min, price_max)
    
    # Collect and sort
    result = events.sort(["ticker", "dt"]).collect()
    
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(out_path, compression="zstd")
        print(f"Wrote training dataset to: {out_path}")
        print(f"  Rows: {result.height:,}")
        print(f"  Columns: {len(result.columns)}")
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ML-ready training dataset from event labels"
    )
    parser.add_argument(
        "--events-glob",
        default="data/cache/training/events_*.parquet",
        help="Glob pattern for event parquet files",
    )
    parser.add_argument(
        "--macd-glob",
        default="data/cache/macd_day_features_inc/mode=all/*.parquet",
        help="Glob pattern for MACD day features",
    )
    parser.add_argument(
        "--vol-path",
        default="data/cache/minute_features.parquet",
        help="Path to volatility features parquet",
    )
    parser.add_argument(
        "--daily-path",
        default="data/cache/daily_filtered_1_9.99.parquet",
        help="Path to daily cache (for price filtering)",
    )
    parser.add_argument(
        "--output",
        default="data/cache/training/ml_dataset.parquet",
        help="Output path for training dataset",
    )
    parser.add_argument(
        "--price-min",
        type=float,
        default=1.0,
        help="Minimum price filter (default: 1.0)",
    )
    parser.add_argument(
        "--price-max",
        type=float,
        default=9.99,
        help="Maximum price filter (default: 9.99)",
    )
    parser.add_argument(
        "--no-price-filter",
        action="store_true",
        help="Skip price filtering",
    )
    
    args = parser.parse_args()
    
    build_training_dataset(
        events_glob=args.events_glob,
        macd_glob=args.macd_glob,
        vol_path=args.vol_path,
        daily_path=args.daily_path if not args.no_price_filter else None,
        price_min=args.price_min if not args.no_price_filter else None,
        price_max=args.price_max if not args.no_price_filter else None,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


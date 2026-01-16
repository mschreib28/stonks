#!/usr/bin/env python3
"""
Build next-day range expansion labels from daily cache.

Labels daily bars based on whether the next day had significant range expansion.
Range expansion is defined as: (high_next - low_next) / close_today >= threshold
"""

from __future__ import annotations

import argparse
from pathlib import Path
import polars as pl


def build_range_expansion_labels(
    daily_path: str,
    threshold: float = 0.08,
    output_path: str | None = None,
) -> pl.DataFrame:
    """
    Create labels for daily bars indicating next-day range expansion.
    
    Args:
        daily_path: Path to daily cache parquet file
        threshold: Range expansion threshold (default 0.08 = 8%)
        output_path: Optional output path (if None, returns DataFrame)
    
    Returns:
        DataFrame with columns: ticker, date, next_day_range_pct, label_next_day_range_expand
    """
    daily = pl.scan_parquet(daily_path)
    
    # Sort by ticker and date
    daily = daily.sort(["ticker", "date"])
    
    # Compute next-day range percentage
    labels = (
        daily
        .with_columns([
            pl.col("high").shift(-1).over("ticker").alias("high_next"),
            pl.col("low").shift(-1).over("ticker").alias("low_next"),
            pl.col("close").alias("close_today"),
        ])
        .with_columns([
            ((pl.col("high_next") - pl.col("low_next")) / pl.col("close_today")).alias("next_day_range_pct"),
            (pl.col("next_day_range_pct") >= threshold).cast(pl.Int8).alias("label_next_day_range_expand"),
        ])
        .select(["ticker", "date", "next_day_range_pct", "label_next_day_range_expand"])
        .filter(pl.col("next_day_range_pct").is_not_null())  # Remove last day (no next day)
    )
    
    result = labels.collect()
    
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.write_parquet(out_path, compression="zstd")
        print(f"Wrote range expansion labels to: {out_path}")
    
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build next-day range expansion labels from daily cache"
    )
    parser.add_argument(
        "--input",
        default="data/cache/daily_2025.parquet",
        help="Path to daily cache parquet file",
    )
    parser.add_argument(
        "--output",
        default="data/cache/range_expansion_labels.parquet",
        help="Output path for range expansion labels",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
        help="Range expansion threshold (default: 0.08 = 8%%)",
    )
    
    args = parser.parse_args()
    
    build_range_expansion_labels(
        daily_path=args.input,
        threshold=args.threshold,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()


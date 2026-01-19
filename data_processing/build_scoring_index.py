#!/usr/bin/env python3
"""
Build a precomputed scoring index using the default Swing Trading Default preset criteria.

This index allows fast queries when using the default scoring criteria without
having to recompute metrics and scores for all tickers on every request.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any
from datetime import datetime, timedelta

import polars as pl

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the compute_ticker_metrics function from the API server
# We'll duplicate it here to avoid circular dependencies
def compute_ticker_metrics(df: pl.DataFrame, months_back: int = 12) -> Dict[str, Any]:
    """
    Compute time-series metrics for a ticker's data.
    Returns metrics like trend, average swings, volatility, etc.
    """
    if len(df) == 0:
        return {}
    
    # Ensure sorted by date
    if "date" in df.columns:
        df = df.sort("date")
    
    # Filter to last N months if date column exists
    if "date" in df.columns and months_back > 0:
        # Get the most recent date in the dataset
        max_date = df["date"].max()
        if max_date is not None:
            # Polars date columns are typically date objects
            # Calculate cutoff date (approximately months_back * 30 days ago)
            try:
                # Try to convert to date if it's a datetime
                if isinstance(max_date, datetime):
                    max_date_val = max_date.date()
                else:
                    max_date_val = max_date
                
                # Calculate cutoff
                cutoff_days = months_back * 30
                cutoff_date = max_date_val - timedelta(days=cutoff_days)
                df = df.filter(pl.col("date") >= cutoff_date)
            except Exception as e:
                # Continue without date filtering
                pass
    
    if len(df) == 0:
        return {}
    
    metrics = {}
    
    # Price trend (slope over time)
    if "close" in df.columns and "date" in df.columns:
        closes = df["close"].to_list()
        dates = df["date"].to_list()
        
        if len(closes) >= 2:
            # Simple linear regression slope (trend)
            # Convert dates to numeric for regression
            date_nums = []
            first_date = dates[0]
            # Normalize first_date to date object
            if isinstance(first_date, datetime):
                first_date_val = first_date.date()
            else:
                first_date_val = first_date
            
            for d in dates:
                # Normalize date to date object
                if isinstance(d, datetime):
                    d_val = d.date()
                else:
                    d_val = d
                
                # Calculate days difference
                try:
                    delta = d_val - first_date_val
                    date_nums.append(delta.days if hasattr(delta, 'days') else 0)
                except:
                    # Fallback to index if date arithmetic fails
                    date_nums.append(len(date_nums))
            
            n = len(closes)
            sum_x = sum(date_nums)
            sum_y = sum(closes)
            sum_xy = sum(x * y for x, y in zip(date_nums, closes))
            sum_x2 = sum(x * x for x in date_nums)
            
            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                metrics["trend_slope"] = float(slope)
                metrics["trend_pct"] = float((slope * len(closes)) / closes[0] * 100) if closes[0] > 0 else 0.0
            else:
                metrics["trend_slope"] = 0.0
                metrics["trend_pct"] = 0.0
            
            # Overall return over period
            if closes[0] > 0:
                metrics["total_return_pct"] = float((closes[-1] / closes[0] - 1) * 100)
            else:
                metrics["total_return_pct"] = 0.0
    
    # Daily swing metrics
    if "abs_ret_d" in df.columns:
        abs_returns = df["abs_ret_d"].drop_nulls().to_list()
        if abs_returns:
            metrics["avg_daily_swing_pct"] = float(sum(abs_returns) / len(abs_returns))
            metrics["max_daily_swing_pct"] = float(max(abs_returns))
            # Count days with swings >= 0.20 (20%)
            large_swings = sum(1 for r in abs_returns if r >= 0.20)
            metrics["large_swing_days"] = large_swings
            metrics["large_swing_frequency"] = float(large_swings / len(abs_returns)) if abs_returns else 0.0
    elif "high" in df.columns and "low" in df.columns and "close" in df.columns:
        # Calculate daily range percentage
        ranges = []
        for row in df.iter_rows(named=True):
            if row.get("close", 0) > 0:
                day_range = (row.get("high", 0) - row.get("low", 0)) / row.get("close", 1)
                ranges.append(day_range)
        if ranges:
            metrics["avg_daily_swing_pct"] = float(sum(ranges) / len(ranges))
            metrics["max_daily_swing_pct"] = float(max(ranges))
            large_swings = sum(1 for r in ranges if r >= 0.20)
            metrics["large_swing_days"] = large_swings
            metrics["large_swing_frequency"] = float(large_swings / len(ranges))
    
    # Volatility (std of returns)
    if "ret_d" in df.columns:
        returns = df["ret_d"].drop_nulls().to_list()
        if len(returns) > 1:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            metrics["volatility"] = float(variance ** 0.5)
    
    # Volume metrics
    if "volume" in df.columns:
        volumes = df["volume"].drop_nulls().to_list()
        if volumes:
            metrics["avg_volume"] = float(sum(volumes) / len(volumes))
            metrics["max_volume"] = float(max(volumes))
            metrics["min_volume"] = float(min(volumes))
    
    # Price metrics
    if "close" in df.columns:
        closes = df["close"].drop_nulls().to_list()
        if closes:
            metrics["current_price"] = float(closes[-1])
            metrics["min_price"] = float(min(closes))
            metrics["max_price"] = float(max(closes))
            metrics["price_range_pct"] = float((max(closes) - min(closes)) / min(closes) * 100) if min(closes) > 0 else 0.0
    
    # ============================================
    # TRADABILITY METRICS
    # ============================================
    TARGET_SHARES = 10000  # Target position size
    
    if "volume" in df.columns and volumes:
        avg_vol = sum(volumes) / len(volumes)
        metrics["liquidity_multiple"] = float(avg_vol / TARGET_SHARES) if TARGET_SHARES > 0 else 0.0
        metrics["position_pct_of_volume"] = float(TARGET_SHARES / avg_vol * 100) if avg_vol > 0 else 100.0
    
    # ============================================
    # DAILY RANGE IN DOLLARS (not percentage)
    # ============================================
    if "high" in df.columns and "low" in df.columns:
        daily_ranges_dollars = []
        for row in df.iter_rows(named=True):
            high = row.get("high", 0)
            low = row.get("low", 0)
            if high > 0 and low > 0:
                daily_ranges_dollars.append(high - low)
        
        if daily_ranges_dollars:
            metrics["avg_daily_range_dollars"] = float(sum(daily_ranges_dollars) / len(daily_ranges_dollars))
            metrics["median_daily_range_dollars"] = float(sorted(daily_ranges_dollars)[len(daily_ranges_dollars) // 2])
            metrics["min_daily_range_dollars"] = float(min(daily_ranges_dollars))
            metrics["max_daily_range_dollars"] = float(max(daily_ranges_dollars))
            
            # Count days in the "sweet spot" of $0.20-$0.60 range
            sweet_spot_days = sum(1 for r in daily_ranges_dollars if 0.20 <= r <= 0.60)
            metrics["sweet_spot_range_days"] = sweet_spot_days
            metrics["sweet_spot_range_pct"] = float(sweet_spot_days / len(daily_ranges_dollars) * 100)
            
            # Consistency: std dev of daily ranges (lower = more predictable)
            if len(daily_ranges_dollars) > 1:
                mean_range = sum(daily_ranges_dollars) / len(daily_ranges_dollars)
                variance = sum((r - mean_range) ** 2 for r in daily_ranges_dollars) / (len(daily_ranges_dollars) - 1)
                metrics["daily_range_std_dollars"] = float(variance ** 0.5)
                # Coefficient of variation (CV) - lower is more consistent
                metrics["daily_range_cv"] = float((variance ** 0.5) / mean_range) if mean_range > 0 else 0.0
    
    # ============================================
    # PROFIT POTENTIAL SCORE
    # ============================================
    if "avg_daily_range_dollars" in metrics and "liquidity_multiple" in metrics:
        metrics["profit_potential_score"] = float(
            metrics["avg_daily_range_dollars"] * min(metrics["liquidity_multiple"], 10)  # Cap liquidity benefit at 10x
        )
    
    # ============================================
    # TRADABILITY SCORE (0-100)
    # ============================================
    if "avg_volume" in metrics and "avg_daily_range_dollars" in metrics:
        # Factors:
        # 1. Liquidity (can we trade 10k shares easily?)
        liquidity_score = min(metrics.get("liquidity_multiple", 0) / 10, 1.0) * 40  # Max 40 points
        
        # 2. Daily range in sweet spot ($0.20-$0.60)?
        avg_range = metrics.get("avg_daily_range_dollars", 0)
        if 0.20 <= avg_range <= 0.60:
            range_score = 40  # Perfect range
        elif 0.10 <= avg_range < 0.20 or 0.60 < avg_range <= 1.00:
            range_score = 25  # Acceptable range
        elif avg_range < 0.10:
            range_score = 5  # Too little movement
        else:
            range_score = 10  # Too volatile
        
        # 3. Consistency (low CV is better)
        cv = metrics.get("daily_range_cv", 1.0)
        if cv < 0.3:
            consistency_score = 20  # Very consistent
        elif cv < 0.5:
            consistency_score = 15
        elif cv < 0.7:
            consistency_score = 10
        else:
            consistency_score = 5  # Unpredictable
        
        metrics["tradability_score"] = float(liquidity_score + range_score + consistency_score)
    
    # Integer metrics (no decimals needed)
    metrics["days_analyzed"] = int(len(df))
    if "large_swing_days" in metrics:
        metrics["large_swing_days"] = int(metrics["large_swing_days"])
    if "sweet_spot_range_days" in metrics:
        metrics["sweet_spot_range_days"] = int(metrics["sweet_spot_range_days"])
    if "avg_volume" in metrics:
        metrics["avg_volume"] = int(metrics["avg_volume"])
    if "max_volume" in metrics:
        metrics["max_volume"] = int(metrics["max_volume"])
    if "min_volume" in metrics:
        metrics["min_volume"] = int(metrics["min_volume"])
    
    return metrics


# Default Swing Trading Default preset criteria
DEFAULT_CRITERIA = [
    {"name": "tradability_score", "weight": 3.0, "min_value": 0, "max_value": 100, "invert": False},
    {"name": "avg_daily_range_dollars", "weight": 2.5, "min_value": 0.10, "max_value": 1.00, "invert": False},
    {"name": "liquidity_multiple", "weight": 2.0, "min_value": 0, "max_value": 100, "invert": False},
    {"name": "sweet_spot_range_pct", "weight": 1.5, "min_value": 0, "max_value": 100, "invert": False},
    {"name": "daily_range_cv", "weight": 1.0, "min_value": 0, "max_value": 1.0, "invert": True},
    {"name": "current_price", "weight": 10.0, "min_value": 0, "max_value": 8, "invert": False},
]

DEFAULT_MONTHS_BACK = 12
DEFAULT_MIN_DAYS = 60
DEFAULT_MIN_AVG_VOLUME = 750000


def score_ticker(metrics: Dict[str, Any], criteria: list[Dict[str, Any]]) -> tuple[float, Dict[str, Any]]:
    """Score a ticker based on criteria. Returns (total_score, criterion_scores)."""
    total_score = 0.0
    criterion_scores = {}
    
    for criterion in criteria:
        value = metrics.get(criterion["name"])
        if value is None:
            continue
        
        # Normalize value to 0-1 range based on min/max
        if criterion.get("min_value") is not None and criterion.get("max_value") is not None:
            min_val = criterion["min_value"]
            max_val = criterion["max_value"]
            if max_val > min_val:
                normalized = (value - min_val) / (max_val - min_val)
                normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
            else:
                normalized = 0.5
        elif criterion.get("min_value") is not None:
            # Only min threshold
            normalized = 1.0 if value >= criterion["min_value"] else 0.0
        elif criterion.get("max_value") is not None:
            # Only max threshold
            normalized = 1.0 if value <= criterion["max_value"] else 0.0
        else:
            # No bounds, use raw value (normalize by assuming reasonable range)
            normalized = 1.0 if value != 0 else 0.0
        
        # Invert if needed
        if criterion.get("invert", False):
            normalized = 1.0 - normalized
        
        criterion_score = normalized * criterion["weight"]
        total_score += criterion_score
        criterion_scores[criterion["name"]] = {
            "value": value,
            "normalized": normalized,
            "score": criterion_score,
        }
    
    return total_score, criterion_scores


def build_scoring_index(
    dataset_path: Path,
    output_path: Path,
    criteria: list[Dict[str, Any]] = None,
    months_back: int = DEFAULT_MONTHS_BACK,
    min_days: int = DEFAULT_MIN_DAYS,
    min_avg_volume: int = DEFAULT_MIN_AVG_VOLUME,
    force: bool = False,
) -> None:
    """Build scoring index for all tickers in the dataset."""
    if criteria is None:
        criteria = DEFAULT_CRITERIA
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if output_path.exists() and not force:
        # Check if we need to rebuild based on dates
        try:
            dataset_df = pl.read_parquet(dataset_path)
            index_df = pl.read_parquet(output_path)
            
            if "date" in dataset_df.columns:
                dataset_latest = dataset_df["date"].max()
                if "last_updated" in index_df.columns:
                    index_latest = index_df["last_updated"].max()
                    if dataset_latest <= index_latest:
                        print(f"⏭️  Scoring index is up to date (dataset: {dataset_latest}, index: {index_latest})")
                        return
        except Exception as e:
            print(f"⚠️  Could not check index freshness: {e}")
            print("   Rebuilding index...")
    
    print(f"Loading dataset: {dataset_path}")
    df = pl.read_parquet(dataset_path)
    
    if "ticker" not in df.columns:
        raise ValueError("Dataset must have a 'ticker' column")
    
    if "date" not in df.columns:
        raise ValueError("Dataset must have a 'date' column")
    
    # Get most recent prices for each ticker
    print("Getting most recent prices...")
    most_recent_prices = {}
    for ticker in df["ticker"].unique().to_list():
        ticker_data = df.filter(pl.col("ticker") == ticker)
        if len(ticker_data) > 0:
            ticker_sorted = ticker_data.sort("date", descending=True)
            most_recent_close = ticker_sorted["close"].first()
            if most_recent_close is not None:
                most_recent_prices[ticker] = float(most_recent_close)
    
    # Get unique tickers
    tickers = df["ticker"].unique().to_list()
    print(f"Processing {len(tickers)} tickers...")
    
    # Compute metrics and scores for each ticker
    index_rows = []
    processed = 0
    
    for ticker in tickers:
        ticker_df = df.filter(pl.col("ticker") == ticker)
        
        # Skip if not enough data
        if len(ticker_df) < min_days:
            continue
        
        # Compute metrics
        metrics = compute_ticker_metrics(ticker_df, months_back)
        if not metrics:
            continue
        
        # Update current_price with most recent price
        if ticker in most_recent_prices:
            metrics["current_price"] = most_recent_prices[ticker]
        
        # Apply minimum average volume filter
        if min_avg_volume > 0:
            avg_vol = metrics.get("avg_volume", 0)
            if avg_vol < min_avg_volume:
                continue
        
        # Apply price filter (0-8 for default)
        current_price = metrics.get("current_price", 0)
        if current_price < 0 or current_price > 8:
            continue
        
        # Score the ticker
        total_score, criterion_scores = score_ticker(metrics, criteria)
        
        # Build row for index
        row = {
            "ticker": ticker,
            "total_score": total_score,
            **metrics,  # Include all metrics
        }
        
        # Add criterion scores as separate columns (for debugging/analysis)
        for criterion_name, criterion_data in criterion_scores.items():
            row[f"{criterion_name}_score"] = criterion_data["score"]
            row[f"{criterion_name}_normalized"] = criterion_data["normalized"]
        
        index_rows.append(row)
        
        processed += 1
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(tickers)} tickers...")
    
    print(f"Processed {processed} tickers total")
    
    # Create DataFrame and sort by score
    index_df = pl.DataFrame(index_rows)
    index_df = index_df.sort("total_score", descending=True)
    
    # Add last_updated timestamp
    latest_date = df["date"].max()
    index_df = index_df.with_columns(pl.lit(latest_date).alias("last_updated"))
    
    # Write to parquet
    print(f"Writing index to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.write_parquet(output_path, compression="zstd")
    
    print(f"✅ Scoring index built: {len(index_df)} tickers")
    print(f"   Top 5 tickers by score:")
    for i, row in enumerate(index_df.head(5).iter_rows(named=True), 1):
        print(f"   {i}. {row['ticker']}: {row['total_score']:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build precomputed scoring index using default Swing Trading Default preset criteria"
    )
    parser.add_argument(
        "--dataset",
        default="data/cache/daily_filtered_1_9.99.parquet",
        help="Path to dataset parquet file (default: filtered dataset)",
    )
    parser.add_argument(
        "--output",
        default="data/cache/scoring_index_default.parquet",
        help="Output path for scoring index",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if index exists and is up to date",
    )
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    
    build_scoring_index(
        dataset_path=dataset_path,
        output_path=output_path,
        force=args.force,
    )


if __name__ == "__main__":
    main()

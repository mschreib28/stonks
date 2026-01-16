#!/usr/bin/env python3
"""
FastAPI server to serve stock data from Parquet files.
Provides endpoints for querying, filtering, and sorting stock data.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback
import logging

import polars as pl

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stonks Data API")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Cache for loaded dataframes
_data_cache: Dict[str, pl.DataFrame] = {}


class FilterRequest(BaseModel):
    dataset: str = "daily"  # daily, weekly, filtered, minute_features
    filters: Dict[str, Any] = {}
    sort_by: Optional[str] = None
    sort_desc: bool = False
    limit: int = 1000
    offset: int = 0


class ScoringCriteria(BaseModel):
    """Individual scoring criterion with weight and range."""
    name: str
    weight: float = 1.0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    invert: bool = False  # If True, lower values score higher


class ScoringRequest(BaseModel):
    dataset: str = "daily"
    filters: Dict[str, Any] = {}
    criteria: List[ScoringCriteria] = []
    min_days: int = 60  # Minimum days of data required
    months_back: int = 12  # How many months back to analyze
    min_avg_volume: int = 750000  # Minimum average daily volume (default 750k)
    min_price: Optional[float] = None  # Minimum current price filter
    max_price: Optional[float] = None  # Maximum current price filter


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool


def load_dataset(dataset: str) -> pl.DataFrame:
    """Load and cache a dataset from Parquet file."""
    if dataset in _data_cache:
        return _data_cache[dataset]
    
    cache_dir = Path("data/2025/cache")
    
    dataset_map = {
        "daily": "daily_2025.parquet",
        "weekly": "weekly.parquet",
        "filtered": "daily_filtered_1_9.99.parquet",
        "minute_features": "minute_features.parquet",
        "minute_features_plus_macd": "minute_features_plus_macd.parquet",
    }
    
    if dataset not in dataset_map:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    parquet_path = cache_dir / dataset_map[dataset]
    
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {parquet_path}. "
            f"Please run: python data_processing/process_all_data.py to generate the cache files."
        )
    
    try:
        df = pl.read_parquet(parquet_path)
        _data_cache[dataset] = df
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset}: {str(e)}")


def apply_filters(df: pl.DataFrame, filters: Dict[str, Any]) -> pl.DataFrame:
    """Apply filters to dataframe."""
    for col, value in filters.items():
        if col not in df.columns:
            continue
        
        if value is None:
            continue
        
        col_type = df[col].dtype
        
        if isinstance(value, dict):
            # Range filter: {"min": x, "max": y}
            if "min" in value and value["min"] is not None:
                df = df.filter(pl.col(col) >= value["min"])
            if "max" in value and value["max"] is not None:
                df = df.filter(pl.col(col) <= value["max"])
            # Boolean filter: {"value": true/false}
            if "value" in value and value["value"] is not None:
                df = df.filter(pl.col(col) == value["value"])
        elif isinstance(value, list):
            # In filter
            df = df.filter(pl.col(col).is_in(value))
        elif isinstance(value, bool):
            df = df.filter(pl.col(col) == value)
        elif isinstance(value, (int, float)):
            if col_type == pl.Utf8:
                df = df.filter(pl.col(col).str.contains(str(value), literal=True))
            else:
                df = df.filter(pl.col(col) == value)
        elif isinstance(value, str):
            if col_type == pl.Utf8:
                df = df.filter(pl.col(col).str.contains(value, literal=False))
            else:
                # Try to convert string to number for numeric columns
                try:
                    num_val = float(value) if "." in value else int(value)
                    df = df.filter(pl.col(col) == num_val)
                except ValueError:
                    df = df.filter(pl.col(col) == value)
    
    return df


@app.get("/")
def root():
    return {"message": "Stonks Data API", "version": "1.0.0"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/api/test")
def test_endpoint():
    """Test endpoint to verify API is working."""
    try:
        import polars as pl
        # Test basic Polars operation
        test_df = pl.DataFrame({"test": [1, 2, 3]})
        return {
            "status": "ok",
            "polars_version": pl.__version__,
            "test_df_length": len(test_df),
        }
    except Exception as e:
        logger.error(f"Test endpoint error: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
        }


@app.get("/api/datasets")
def list_datasets():
    """List available datasets."""
    try:
        cache_dir = Path("data/2025/cache")
        datasets = []
        
        for name, filename in [
            ("daily", "daily_2025.parquet"),
            ("weekly", "weekly.parquet"),
            ("filtered", "daily_filtered_1_9.99.parquet"),
            ("minute_features", "minute_features.parquet"),
            ("minute_features_plus_macd", "minute_features_plus_macd.parquet"),
        ]:
            path = cache_dir / filename
            if path.exists():
                try:
                    df = load_dataset(name)
                    datasets.append({
                        "name": name,
                        "filename": filename,
                        "rows": len(df),
                        "columns": len(df.columns),
                    })
                except Exception as e:
                    logger.error(f"Error loading dataset {name}: {e}")
                    logger.error(traceback.format_exc())
                    datasets.append({
                        "name": name,
                        "filename": filename,
                        "error": str(e),
                    })
            else:
                datasets.append({
                    "name": name,
                    "filename": filename,
                    "error": f"File not found: {path}",
                })
        
        return {"datasets": datasets}
    except Exception as e:
        logger.error(f"Error in list_datasets: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error listing datasets: {str(e)}")


@app.get("/api/dataset/{dataset}/columns")
def get_columns(dataset: str):
    """Get column information for a dataset."""
    try:
        df = load_dataset(dataset)
        columns = []
        for col in df.columns:
            try:
                dtype = df[col].dtype
                null_count = df[col].null_count()
                columns.append({
                    "name": col,
                    "type": str(dtype),
                    "nullable": null_count > 0,
                })
            except Exception as e:
                logger.error(f"Error processing column {col}: {e}")
                logger.error(traceback.format_exc())
                # Still add the column with basic info
                columns.append({
                    "name": col,
                    "type": "unknown",
                    "nullable": True,
                })
        return {"columns": columns}
    except Exception as e:
        logger.error(f"Error in get_columns for dataset {dataset}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading columns: {str(e)}")


@app.post("/api/query")
def query_data(request: FilterRequest):
    """Query data with filters, sorting, and pagination."""
    try:
        df = load_dataset(request.dataset)
        
        # Apply filters
        if request.filters:
            df = apply_filters(df, request.filters)
        
        # Apply sorting
        if request.sort_by and request.sort_by in df.columns:
            df = df.sort(request.sort_by, descending=request.sort_desc)
        
        # Get total count before pagination
        total = len(df)
        
        # Apply pagination
        if request.offset > 0:
            df = df.slice(request.offset, request.limit)
        else:
            df = df.head(request.limit)
        
        # Convert to dict for JSON response
        return {
            "data": df.to_dicts(),
            "total": total,
            "limit": request.limit,
            "offset": request.offset,
            "columns": list(df.columns),
        }
    except Exception as e:
        logger.error(f"Error in query_data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error querying data: {str(e)}")


@app.get("/api/dataset/{dataset}/stats")
def get_stats(dataset: str):
    """Get statistical summary of a dataset."""
    try:
        df = load_dataset(dataset)
        
        # Get numeric columns - handle both Int64/Float64 and other numeric types
        numeric_cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]:
                numeric_cols.append(col)
        
        stats = {}
        for col in numeric_cols:
            try:
                col_series = df[col]
                null_count = col_series.null_count()
                non_null_count = len(df) - null_count
                
                if non_null_count > 0:
                    stats[col] = {
                        "min": float(col_series.min()) if non_null_count > 0 else None,
                        "max": float(col_series.max()) if non_null_count > 0 else None,
                        "mean": float(col_series.mean()) if non_null_count > 0 else None,
                        "std": float(col_series.std()) if non_null_count > 1 else None,
                    }
                else:
                    stats[col] = {
                        "min": None,
                        "max": None,
                        "mean": None,
                        "std": None,
                    }
            except Exception as e:
                logger.error(f"Error computing stats for column {col}: {e}")
                logger.error(traceback.format_exc())
                stats[col] = {
                    "min": None,
                    "max": None,
                    "mean": None,
                    "std": None,
                }
        
        return {"stats": stats, "total_rows": len(df)}
    except Exception as e:
        logger.error(f"Error in get_stats for dataset {dataset}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error computing stats: {str(e)}")


@app.get("/api/dataset/{dataset}/ticker/{ticker}")
def get_ticker_data(dataset: str, ticker: str, limit: int = 100):
    """Get all data for a specific ticker."""
    try:
        df = load_dataset(dataset)
        
        if "ticker" not in df.columns:
            raise HTTPException(status_code=400, detail="Dataset does not have ticker column")
        
        df = df.filter(pl.col("ticker") == ticker)
        df = df.sort("date" if "date" in df.columns else df.columns[1], descending=True)
        df = df.head(limit)
        
        return {
            "ticker": ticker,
            "data": df.to_dicts(),
            "total": len(df),
        }
    except Exception as e:
        logger.error(f"Error in get_ticker_data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading ticker data: {str(e)}")


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
                logger.warning(f"Could not filter by date: {e}")
                # Continue without date filtering
    
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
    # How easily can you trade 10,000 shares?
    # If avg_volume is 100k, you're trading 10% of daily volume - very feasible
    # If avg_volume is 10k, you're trading 100% of daily volume - will move price
    TARGET_SHARES = 10000  # Target position size
    
    if "volume" in df.columns and volumes:
        avg_vol = sum(volumes) / len(volumes)
        # Liquidity multiple: how many times your target position fits in daily volume
        # Higher is better (more liquid)
        metrics["liquidity_multiple"] = float(avg_vol / TARGET_SHARES) if TARGET_SHARES > 0 else 0.0
        
        # Position as % of daily volume (lower is better, means less market impact)
        metrics["position_pct_of_volume"] = float(TARGET_SHARES / avg_vol * 100) if avg_vol > 0 else 100.0
    
    # ============================================
    # DAILY RANGE IN DOLLARS (not percentage)
    # ============================================
    # For swing trading, we want stocks that move $0.20-$0.60 per day typically
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
    # Combines liquidity with price movement
    # profit_potential = (avg_daily_range * liquidity_multiple) 
    # Higher means: good price movement AND enough volume to trade it
    if "avg_daily_range_dollars" in metrics and "liquidity_multiple" in metrics:
        # Scale: if you can trade 10k shares (1x liquidity) and stock moves $0.30, profit potential = $3000
        # This represents theoretical max profit if you caught the full daily range
        metrics["profit_potential_score"] = float(
            metrics["avg_daily_range_dollars"] * min(metrics["liquidity_multiple"], 10)  # Cap liquidity benefit at 10x
        )
    
    # ============================================
    # TRADABILITY SCORE (0-100)
    # ============================================
    # Combines multiple factors into a single tradability score
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


@app.post("/api/score-tickers")
def score_tickers(request: ScoringRequest):
    """
    Score and rank tickers based on configurable criteria.
    Computes time-series metrics for each ticker and applies weighted scoring.
    """
    try:
        df = load_dataset(request.dataset)
        
        # Apply base filters first
        if request.filters:
            df = apply_filters(df, request.filters)
        
        if len(df) == 0:
            return {
                "ranked_tickers": [],
                "total": 0,
            }
        
        # Get unique tickers
        if "ticker" not in df.columns:
            raise HTTPException(status_code=400, detail="Dataset does not have ticker column")
        
        tickers = df["ticker"].unique().to_list()
        
        # Compute metrics for each ticker
        ticker_metrics = {}
        for ticker in tickers:
            ticker_df = df.filter(pl.col("ticker") == ticker)
            
            # Skip if not enough data
            if len(ticker_df) < request.min_days:
                continue
            
            metrics = compute_ticker_metrics(ticker_df, request.months_back)
            if metrics:
                # Apply minimum average volume filter
                if request.min_avg_volume > 0:
                    avg_vol = metrics.get("avg_volume", 0)
                    if avg_vol < request.min_avg_volume:
                        continue  # Skip tickers with insufficient volume
                
                # Apply price range filter based on current price
                current_price = metrics.get("current_price", 0)
                if request.min_price is not None and current_price < request.min_price:
                    continue  # Skip tickers below minimum price
                if request.max_price is not None and current_price > request.max_price:
                    continue  # Skip tickers above maximum price
                
                ticker_metrics[ticker] = metrics
        
        # Score each ticker based on criteria
        scored_tickers = []
        for ticker, metrics in ticker_metrics.items():
            total_score = 0.0
            criterion_scores = {}
            
            for criterion in request.criteria:
                value = metrics.get(criterion.name)
                if value is None:
                    continue
                
                # Normalize value to 0-1 range based on min/max
                if criterion.min_value is not None and criterion.max_value is not None:
                    if criterion.max_value > criterion.min_value:
                        normalized = (value - criterion.min_value) / (criterion.max_value - criterion.min_value)
                        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]
                    else:
                        normalized = 0.5
                elif criterion.min_value is not None:
                    # Only min threshold
                    normalized = 1.0 if value >= criterion.min_value else 0.0
                elif criterion.max_value is not None:
                    # Only max threshold
                    normalized = 1.0 if value <= criterion.max_value else 0.0
                else:
                    # No bounds, use raw value (normalize by assuming reasonable range)
                    # For now, just use 1.0 if value exists
                    normalized = 1.0 if value != 0 else 0.0
                
                # Invert if needed
                if criterion.invert:
                    normalized = 1.0 - normalized
                
                criterion_score = normalized * criterion.weight
                total_score += criterion_score
                criterion_scores[criterion.name] = {
                    "value": value,
                    "normalized": normalized,
                    "score": criterion_score,
                }
            
            scored_tickers.append({
                "ticker": ticker,
                "total_score": total_score,
                "metrics": metrics,
                "criterion_scores": criterion_scores,
            })
        
        # Sort by total score (descending)
        scored_tickers.sort(key=lambda x: x["total_score"], reverse=True)
        
        return {
            "ranked_tickers": scored_tickers,
            "total": len(scored_tickers),
        }
    except Exception as e:
        logger.error(f"Error in score_tickers: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error scoring tickers: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


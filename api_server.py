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

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import subprocess
import sys
import threading
import os

# Set up logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import dotenv, fall back to manual parsing if not available
try:
    from dotenv import load_dotenv
    HAS_DOTENV = True
except ImportError:
    HAS_DOTENV = False
    logger.warning("python-dotenv not installed, will use manual .env parsing")

# Load environment variables from .env file
project_root = Path.cwd()
env_path = project_root / ".env"
if env_path.exists():
    if HAS_DOTENV:
        load_dotenv(env_path)
        logger.info(f"Loaded environment variables from {env_path}")
    else:
        # Manual .env parsing
        try:
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
            logger.info(f"Manually loaded environment variables from {env_path}")
        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

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


# Cache for lazy frames (memory efficient - only scans, doesn't load)
_data_cache: Dict[str, pl.LazyFrame] = {}

# Status tracking for data updates
_update_status: Dict[str, Any] = {
    "status": "idle",  # idle, checking, downloading, processing, completed, error
    "message": "",
    "progress": 0,
}
_update_lock = threading.Lock()


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


def load_dataset(dataset: str, materialize: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """
    Load a dataset from Parquet file using lazy loading for memory efficiency.
    
    Args:
        dataset: Dataset name
        materialize: If True, return DataFrame (loads into memory). If False, return LazyFrame (memory efficient).
    
    Returns:
        LazyFrame if materialize=False, DataFrame if materialize=True
    """
    # Return cached lazy frame if available and not materializing
    if dataset in _data_cache and not materialize:
        return _data_cache[dataset]
    
    cache_dir = Path("data/cache")
    
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
            f"Please run: python process_all_data.py to generate the cache files."
        )
    
    try:
        # Use lazy loading (scan_parquet) for memory efficiency
        lf = pl.scan_parquet(parquet_path)
        
        # Cache the lazy frame
        if dataset not in _data_cache:
            _data_cache[dataset] = lf
        
        # Materialize only if requested
        if materialize:
            return lf.collect()
        return lf
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset {dataset}: {str(e)}")


def apply_filters(df: pl.DataFrame | pl.LazyFrame, filters: Dict[str, Any]) -> pl.DataFrame | pl.LazyFrame:
    """Apply filters to dataframe or lazyframe."""
    # Get schema to check columns (works for both DataFrame and LazyFrame)
    schema = df.schema if hasattr(df, 'schema') else {col: df[col].dtype for col in df.columns}
    
    for col, value in filters.items():
        if col not in schema:
            continue
        
        if value is None:
            continue
        
        col_type = schema[col]
        
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
                except (ValueError, TypeError):
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
        cache_dir = Path("data/cache")
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
                    # Use lazy loading to get metadata without loading full dataset
                    lf = load_dataset(name)
                    # Get row count efficiently using lazy evaluation
                    row_count = lf.select(pl.len().alias("count")).collect()["count"][0]
                    column_count = len(lf.schema)
                    datasets.append({
                        "name": name,
                        "filename": filename,
                        "rows": row_count,
                        "columns": column_count,
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
        lf = load_dataset(dataset)
        columns = []
        schema = lf.schema
        for col, dtype in schema.items():
            try:
                # Get null count efficiently using lazy evaluation
                null_count = lf.select(pl.col(col).null_count().alias("nulls")).collect()["nulls"][0]
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
                    "type": str(dtype),
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
        lf = load_dataset(request.dataset)
        
        # Apply filters (works with LazyFrame)
        if request.filters:
            lf = apply_filters(lf, request.filters)
        
        # Get total count efficiently before pagination
        total = lf.select(pl.len().alias("count")).collect()["count"][0]
        
        # Apply sorting
        if request.sort_by and request.sort_by in lf.schema:
            lf = lf.sort(request.sort_by, descending=request.sort_desc)
        
        # Apply pagination
        if request.offset > 0:
            lf = lf.slice(request.offset, request.limit)
        else:
            lf = lf.head(request.limit)
        
        # Materialize only the paginated results
        df = lf.collect()
        
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
        lf = load_dataset(dataset)
        
        # Get numeric columns from schema
        numeric_cols = []
        schema = lf.schema
        for col, dtype in schema.items():
            if dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64, 
                        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
                        pl.Float32, pl.Float64]:
                numeric_cols.append(col)
        
        # Compute stats efficiently using lazy evaluation
        stats = {}
        if numeric_cols:
            # Build aggregation expressions for all numeric columns
            agg_exprs = []
            for col in numeric_cols:
                agg_exprs.extend([
                    pl.col(col).min().alias(f"{col}_min"),
                    pl.col(col).max().alias(f"{col}_max"),
                    pl.col(col).mean().alias(f"{col}_mean"),
                    pl.col(col).std().alias(f"{col}_std"),
                    pl.col(col).null_count().alias(f"{col}_nulls"),
                ])
            
            # Get total count
            total_count = lf.select(pl.len().alias("count")).collect()["count"][0]
            
            # Compute all stats in one pass
            stats_df = lf.select(agg_exprs).collect()
            
            for col in numeric_cols:
                try:
                    null_count = stats_df[f"{col}_nulls"][0]
                    non_null_count = total_count - null_count
                    
                    if non_null_count > 0:
                        stats[col] = {
                            "min": float(stats_df[f"{col}_min"][0]) if stats_df[f"{col}_min"][0] is not None else None,
                            "max": float(stats_df[f"{col}_max"][0]) if stats_df[f"{col}_max"][0] is not None else None,
                            "mean": float(stats_df[f"{col}_mean"][0]) if stats_df[f"{col}_mean"][0] is not None else None,
                            "std": float(stats_df[f"{col}_std"][0]) if stats_df[f"{col}_std"][0] is not None and non_null_count > 1 else None,
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
        else:
            total_count = lf.select(pl.len().alias("count")).collect()["count"][0]
        
        return {"stats": stats, "total_rows": total_count}
    except Exception as e:
        logger.error(f"Error in get_stats for dataset {dataset}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error computing stats: {str(e)}")


@app.get("/api/dataset/{dataset}/ticker/{ticker}")
def get_ticker_data(dataset: str, ticker: str, limit: int = 100):
    """Get all data for a specific ticker."""
    try:
        lf = load_dataset(dataset)
        
        if "ticker" not in lf.schema:
            raise HTTPException(status_code=400, detail="Dataset does not have ticker column")
        
        # Filter to this ticker
        lf_filtered = lf.filter(pl.col("ticker") == ticker)
        
        # Get total count before limiting
        total_count = lf_filtered.select(pl.len().alias("count")).collect()["count"][0]
        
        # Sort and limit
        sort_col = "date" if "date" in lf_filtered.schema else list(lf_filtered.schema.keys())[1]
        lf_filtered = lf_filtered.sort(sort_col, descending=True)
        lf_filtered = lf_filtered.head(limit)
        
        # Materialize only the limited results
        df = lf_filtered.collect()
        
        return {
            "ticker": ticker,
            "data": df.to_dicts(),
            "total": total_count,  # Total available, not just returned
            "returned": len(df),  # Number actually returned
        }
    except Exception as e:
        logger.error(f"Error in get_ticker_data: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error loading ticker data: {str(e)}")


def compute_ticker_metrics(df: pl.DataFrame | pl.LazyFrame, months_back: int = 12) -> Dict[str, Any]:
    """
    Compute time-series metrics for a ticker's data.
    Returns metrics like trend, average swings, volatility, etc.
    Supports both daily and weekly datasets.
    """
    # Materialize if LazyFrame
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    
    if len(df) == 0:
        return {}
    
    # Detect dataset type based on available columns
    is_weekly = "week_start" in df.columns
    date_col = "week_start" if is_weekly else "date"
    close_col = "close_w" if is_weekly else "close"
    ret_col = "ret_w" if is_weekly else "ret_d"
    abs_ret_col = "abs_ret_w" if is_weekly else "abs_ret_d"
    
    # Ensure sorted by date
    if date_col in df.columns:
        df = df.sort(date_col)
    
    # Filter to last N months if date column exists
    if date_col in df.columns and months_back > 0:
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
                df = df.filter(pl.col(date_col) >= cutoff_date)
            except Exception as e:
                logger.warning(f"Could not filter by date: {e}")
                # Continue without date filtering
    
    if len(df) == 0:
        return {}
    
    metrics = {}
    
    # Price trend (slope over time)
    if close_col in df.columns and date_col in df.columns:
        closes = df[close_col].to_list()
        dates = df[date_col].to_list()
        
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
    
    # Daily/weekly swing metrics
    if abs_ret_col in df.columns:
        abs_returns = df[abs_ret_col].drop_nulls().to_list()
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
    if ret_col in df.columns:
        returns = df[ret_col].drop_nulls().to_list()
        if len(returns) > 1:
            mean_ret = sum(returns) / len(returns)
            variance = sum((r - mean_ret) ** 2 for r in returns) / (len(returns) - 1)
            metrics["volatility"] = float(variance ** 0.5)
    
    # Volume metrics (only available for daily dataset)
    volumes = None
    if "volume" in df.columns:
        volumes = df["volume"].drop_nulls().to_list()
        if volumes:
            metrics["avg_volume"] = float(sum(volumes) / len(volumes))
            metrics["max_volume"] = float(max(volumes))
            metrics["min_volume"] = float(min(volumes))
    
    # Price metrics
    if close_col in df.columns:
        closes = df[close_col].drop_nulls().to_list()
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
    # Note: Weekly datasets don't have volume, so these metrics won't be computed
    TARGET_SHARES = 10000  # Target position size
    
    if volumes is not None and len(volumes) > 0:
        avg_vol = sum(volumes) / len(volumes)
        # Liquidity multiple: how many times your target position fits in daily volume
        # Higher is better (more liquid)
        metrics["liquidity_multiple"] = float(avg_vol / TARGET_SHARES) if TARGET_SHARES > 0 else 0.0
        
        # Position as % of daily volume (lower is better, means less market impact)
        metrics["position_pct_of_volume"] = float(TARGET_SHARES / avg_vol * 100) if avg_vol > 0 else 100.0
    else:
        # Weekly datasets don't have volume - set defaults
        metrics["liquidity_multiple"] = 0.0
        metrics["position_pct_of_volume"] = 100.0
    
    # ============================================
    # DAILY/WEEKLY RANGE IN DOLLARS (not percentage)
    # ============================================
    # For swing trading, we want stocks that move $0.20-$0.60 per day typically
    # For weekly data, we'll scale the ranges appropriately
    if "high" in df.columns and "low" in df.columns:
        ranges_dollars = []
        for row in df.iter_rows(named=True):
            high = row.get("high", 0)
            low = row.get("low", 0)
            if high > 0 and low > 0:
                ranges_dollars.append(high - low)
        
        if ranges_dollars:
            # For weekly data, divide by ~5 to approximate daily range
            scale_factor = 1.0 if not is_weekly else 5.0
            scaled_ranges = [r / scale_factor for r in ranges_dollars]
            
            metrics["avg_daily_range_dollars"] = float(sum(scaled_ranges) / len(scaled_ranges))
            metrics["median_daily_range_dollars"] = float(sorted(scaled_ranges)[len(scaled_ranges) // 2])
            metrics["min_daily_range_dollars"] = float(min(scaled_ranges))
            metrics["max_daily_range_dollars"] = float(max(scaled_ranges))
            
            # Count periods in the "sweet spot" of $0.20-$0.60 range (scaled for weekly)
            sweet_spot_periods = sum(1 for r in scaled_ranges if 0.20 <= r <= 0.60)
            metrics["sweet_spot_range_days"] = sweet_spot_periods
            metrics["sweet_spot_range_pct"] = float(sweet_spot_periods / len(scaled_ranges) * 100)
            
            # Consistency: std dev of ranges (lower = more predictable)
            if len(scaled_ranges) > 1:
                mean_range = sum(scaled_ranges) / len(scaled_ranges)
                variance = sum((r - mean_range) ** 2 for r in scaled_ranges) / (len(scaled_ranges) - 1)
                metrics["daily_range_std_dollars"] = float(variance ** 0.5)
                # Coefficient of variation (CV) - lower is more consistent
                metrics["daily_range_cv"] = float((variance ** 0.5) / mean_range) if mean_range > 0 else 0.0
    elif is_weekly:
        # Weekly dataset doesn't have high/low, skip range metrics
        pass
    
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
    # Note: For weekly datasets, volume-based metrics won't be available
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
    period_label = "weeks" if is_weekly else "days"
    metrics["days_analyzed"] = int(len(df))  # Keep name for compatibility, but represents periods
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
        lf = load_dataset(request.dataset)
        
        # Get schema once to avoid performance warnings
        schema = lf.schema
        
        # Get most recent prices for each ticker BEFORE applying filters
        # This ensures we use the actual current price, not a stale price from filtered data
        # This is important when filtering by "high" column, as the most recent day might be filtered out
        most_recent_prices = {}
        # Detect dataset type
        is_weekly = "week_start" in schema
        date_col = "week_start" if is_weekly else "date"
        close_col = "close_w" if is_weekly else "close"
        
        if "ticker" in schema and close_col in schema and date_col in schema:
            # Get the most recent close price for each ticker using window functions
            ticker_prices = (
                lf.sort(date_col, descending=True)
                .group_by("ticker")
                .agg([
                    pl.col(close_col).first().alias("latest_close")
                ])
                .collect()
            )
            for row in ticker_prices.iter_rows(named=True):
                if row["latest_close"] is not None:
                    most_recent_prices[row["ticker"]] = float(row["latest_close"])
        
        # Apply base filters first
        if request.filters:
            lf = apply_filters(lf, request.filters)
        
        # Get unique tickers efficiently
        if "ticker" not in schema:
            raise HTTPException(status_code=400, detail="Dataset does not have ticker column")
        
        tickers_df = lf.select("ticker").unique().collect()
        tickers = tickers_df["ticker"].to_list()
        
        if len(tickers) == 0:
            return {
                "ranked_tickers": [],
                "total": 0,
            }
        
        # Compute metrics for each ticker
        ticker_metrics = {}
        for ticker in tickers:
            # Apply price filter using most recent price (before other filters)
            if request.min_price is not None or request.max_price is not None:
                current_price = most_recent_prices.get(ticker)
                if current_price is None:
                    # Fallback: get from filtered data if we don't have most recent
                    ticker_lf_temp = lf.filter(pl.col("ticker") == ticker)
                    ticker_df_temp = ticker_lf_temp.sort(date_col, descending=True).head(1).collect()
                    if len(ticker_df_temp) > 0 and close_col in ticker_df_temp.columns:
                        current_price = float(ticker_df_temp[close_col][0])
                    else:
                        current_price = 0
                
                if request.min_price is not None and current_price < request.min_price:
                    continue  # Skip tickers below minimum price
                if request.max_price is not None and current_price > request.max_price:
                    continue  # Skip tickers above maximum price
            
            # Filter to this ticker and materialize only what we need
            ticker_lf = lf.filter(pl.col("ticker") == ticker)
            
            # Check if enough data (efficiently)
            ticker_count = ticker_lf.select(pl.len().alias("count")).collect()["count"][0]
            if ticker_count < request.min_days:
                continue
            
            # Materialize for metrics computation
            ticker_df = ticker_lf.collect()
            
            metrics = compute_ticker_metrics(ticker_df, request.months_back)
            if metrics:
                # Always update current_price with the most recent price from the full dataset
                # This ensures accuracy even when filters exclude the most recent days
                if ticker in most_recent_prices:
                    metrics["current_price"] = most_recent_prices[ticker]
                
                # Apply minimum average volume filter (skip for weekly datasets which don't have volume)
                if request.min_avg_volume > 0 and not is_weekly:
                    avg_vol = metrics.get("avg_volume", 0)
                    if avg_vol < request.min_avg_volume:
                        continue  # Skip tickers with insufficient volume
                
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


def get_last_trading_day() -> datetime:
    """Get the last trading day (yesterday if weekday, or last Friday if weekend)."""
    today = datetime.now().date()
    # If today is Monday, last trading day is Friday (3 days ago)
    # If today is Sunday, last trading day is Friday (2 days ago)
    # If today is Saturday, last trading day is Friday (1 day ago)
    # Otherwise, last trading day is yesterday
    
    weekday = today.weekday()  # 0 = Monday, 6 = Sunday
    
    if weekday == 0:  # Monday
        days_back = 3
    elif weekday == 6:  # Sunday
        days_back = 2
    elif weekday == 5:  # Saturday
        days_back = 1
    else:  # Tuesday-Friday
        days_back = 1
    
    last_trading_day = today - timedelta(days=days_back)
    return datetime.combine(last_trading_day, datetime.min.time())


def get_latest_data_date() -> Optional[datetime]:
    """Get the latest date available in the datasets."""
    try:
        # Try to load the daily dataset to get the latest date (lazy)
        lf = load_dataset("daily")
        if "date" not in lf.schema:
            return None
        
        # Get max date efficiently using lazy evaluation
        max_date_df = lf.select(pl.col("date").max().alias("max_date")).collect()
        max_date = max_date_df["max_date"][0]
        if max_date is None:
            return None
        
        # Convert to datetime if needed
        if isinstance(max_date, datetime):
            return max_date
        elif hasattr(max_date, 'date'):
            return datetime.combine(max_date, datetime.min.time())
        else:
            # Try to parse as date string
            try:
                return datetime.strptime(str(max_date), "%Y-%m-%d")
            except:
                return None
    except Exception as e:
        logger.error(f"Error getting latest data date: {e}")
        return None


@app.get("/api/data-freshness")
def check_data_freshness():
    """Check if we have data up to the prior day's market close."""
    try:
        latest_data_date = get_latest_data_date()
        last_trading_day = get_last_trading_day()
        
        if latest_data_date is None:
            return {
                "is_fresh": False,
                "latest_data_date": None,
                "expected_date": last_trading_day.date().isoformat(),
                "message": "No data found in datasets",
            }
        
        # Convert latest_data_date to date for comparison
        if isinstance(latest_data_date, datetime):
            latest_date = latest_data_date.date()
        else:
            latest_date = latest_data_date
        
        expected_date = last_trading_day.date()
        is_fresh = latest_date >= expected_date
        
        return {
            "is_fresh": is_fresh,
            "latest_data_date": latest_date.isoformat(),
            "expected_date": expected_date.isoformat(),
            "message": "Data is up to date" if is_fresh else f"Data is missing for {expected_date.isoformat()}",
        }
    except Exception as e:
        logger.error(f"Error checking data freshness: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error checking data freshness: {str(e)}")


@app.get("/api/update-status")
def get_update_status():
    """Get the current status of data update operation."""
    with _update_lock:
        return _update_status.copy()


def run_data_update():
    """Run the data update script in the background."""
    global _update_status
    
    with _update_lock:
        _update_status = {
            "status": "checking",
            "message": "Checking for new data...",
            "progress": 0,
        }
    
    try:
        # Get the project root directory
        project_root = Path.cwd()
        
        # Try data_processing directory first, then root
        script_paths = [
            project_root / "data_processing" / "update_and_process_data.py",
            project_root / "update_and_process_data.py",
        ]
        
        script_path = None
        for path in script_paths:
            if path.exists():
                script_path = path
                break
        
        if script_path is None:
            raise FileNotFoundError(f"Update script not found. Tried: {[str(p) for p in script_paths]}. Project root: {project_root}")
        
        # Prepare environment variables for subprocess
        env = os.environ.copy()
        
        # Load AWS credentials from .env if they exist
        aws_access_key_id = os.getenv("aws_access_key_id") or os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = os.getenv("aws_secret_access_key") or os.getenv("AWS_SECRET_ACCESS_KEY")
        
        # Run the script from project root
        # Only pass --aws-profile if we don't have credentials in .env
        if aws_access_key_id and aws_secret_access_key:
            # Don't pass --aws-profile when using env vars, let the script use env vars
            cmd = [sys.executable, str(script_path)]
        else:
            # Use AWS profile if no env vars
            cmd = [sys.executable, str(script_path), "--aws-profile", "massive"]
        
        if aws_access_key_id and aws_secret_access_key:
            # Set AWS credentials in environment
            env["AWS_ACCESS_KEY_ID"] = aws_access_key_id
            env["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
            # Log masked credentials for debugging
            masked_key = f"{aws_access_key_id[:8]}...{aws_access_key_id[-4:]}" if len(aws_access_key_id) > 12 else "***"
            masked_secret = f"{'*' * 20}...{aws_secret_access_key[-4:]}" if len(aws_secret_access_key) > 24 else "***"
            logger.info(f"Using AWS credentials from .env file")
            logger.info(f"  Access Key ID: {masked_key} (length: {len(aws_access_key_id)})")
            logger.info(f"  Secret Key: {masked_secret} (length: {len(aws_secret_access_key)})")
        else:
            # Use AWS profile instead
            env["AWS_PROFILE"] = "massive"
            logger.info("Using AWS profile: massive")
            logger.info(f"  AWS_ACCESS_KEY_ID in env: {'Yes' if 'AWS_ACCESS_KEY_ID' in env else 'No'}")
            logger.info(f"  AWS_SECRET_ACCESS_KEY in env: {'Yes' if 'AWS_SECRET_ACCESS_KEY' in env else 'No'}")
        
        logger.info(f"Running update script: {' '.join(cmd)} from {project_root}")
        
        # Capture output
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=str(project_root),
            env=env,
        )
        
        output_lines = []
        in_download_phase = False
        in_process_phase = False
        
        for line in process.stdout:
            output_lines.append(line)
            line_stripped = line.strip()
            line_lower = line_stripped.lower()
            
            # Update status based on output
            if "checking" in line_lower or "check" in line_lower:
                with _update_lock:
                    _update_status["status"] = "checking"
                    _update_status["message"] = line_stripped[:100]  # Limit message length
                    _update_status["progress"] = 10
            elif "downloading" in line_lower or "download" in line_lower or "📥" in line:
                in_download_phase = True
                with _update_lock:
                    _update_status["status"] = "downloading"
                    _update_status["message"] = line_stripped[:100]
                    if _update_status["progress"] < 50:
                        _update_status["progress"] = min(50, _update_status.get("progress", 25) + 2)
            elif "processing" in line_lower or "process" in line_lower or "🔄" in line:
                in_process_phase = True
                in_download_phase = False
                with _update_lock:
                    _update_status["status"] = "processing"
                    _update_status["message"] = line_stripped[:100]
                    _update_status["progress"] = max(50, min(90, _update_status.get("progress", 50) + 2))
            elif in_download_phase:
                with _update_lock:
                    _update_status["message"] = line_stripped[:100]
            elif in_process_phase:
                with _update_lock:
                    _update_status["message"] = line_stripped[:100]
                    _update_status["progress"] = min(90, _update_status.get("progress", 50) + 1)
        
        process.wait()
        
        if process.returncode == 0:
            with _update_lock:
                _update_status = {
                    "status": "completed",
                    "message": "Data update completed successfully!",
                    "progress": 100,
                }
            
            # Clear cache so fresh data is loaded
            _data_cache.clear()
            logger.info("Data update completed successfully")
        else:
            error_msg = "\n".join(output_lines[-10:])  # Last 10 lines
            with _update_lock:
                _update_status = {
                    "status": "error",
                    "message": f"Update failed. See logs for details.",
                    "progress": 0,
                }
            logger.error(f"Update script failed with return code {process.returncode}")
            logger.error(f"Last output lines: {error_msg}")
    except Exception as e:
        logger.error(f"Error running data update: {e}")
        logger.error(traceback.format_exc())
        with _update_lock:
            _update_status = {
                "status": "error",
                "message": f"Error: {str(e)}",
                "progress": 0,
            }


@app.post("/api/update-data")
def trigger_data_update(background_tasks: BackgroundTasks):
    """Trigger a data update (download and process)."""
    with _update_lock:
        if _update_status["status"] in ["checking", "downloading", "processing"]:
            raise HTTPException(
                status_code=400,
                detail="Update already in progress"
            )
    
    # Start background task
    background_tasks.add_task(run_data_update)
    
    return {
        "status": "started",
        "message": "Data update started",
    }


# ============================================
# Factor Evaluation Endpoints
# ============================================

class FactorEvaluationRequest(BaseModel):
    factor_column: str
    periods: List[int] = [1, 5, 10]
    quantiles: int = 5


@app.post("/api/evaluate-factor")
def evaluate_factor(request: FactorEvaluationRequest):
    """
    Evaluate a factor/signal using Information Coefficient analysis.
    
    Computes:
    - IC (rank correlation with forward returns)
    - Factor quantile returns
    - Factor statistics
    """
    try:
        # Check if technical features exist
        tech_path = Path("data/cache/technical_features.parquet")
        daily_path = Path("data/cache/daily_2025.parquet")
        
        if not tech_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Technical features not found. Run: python data_processing/build_technical_features.py"
            )
        
        if not daily_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Daily data not found. Run data processing first."
            )
        
        # Import evaluation module
        from data_processing.evaluate_factors import load_factor_and_prices, compute_ic_simple, compute_factor_quantile_returns
        
        # Load data
        factor, prices = load_factor_and_prices(
            factor_path=str(tech_path),
            factor_column=request.factor_column,
            price_path=str(daily_path),
        )
        
        # Compute IC
        ic_results = compute_ic_simple(factor, prices, request.periods)
        
        # Compute quantile returns
        quantile_returns = compute_factor_quantile_returns(
            factor, prices, request.periods, request.quantiles
        )
        
        # Format response
        response = {
            "factor": request.factor_column,
            "ic_analysis": {},
            "quantile_returns": {},
        }
        
        for period_key, ic_data in ic_results.items():
            response["ic_analysis"][period_key] = {
                "mean_ic": ic_data["mean"],
                "ic_std": ic_data["std"],
                "t_stat": ic_data["t_stat"],
                "positive_pct": ic_data["positive_pct"],
            }
        
        for period_key, qr_data in quantile_returns.items():
            response["quantile_returns"][period_key] = {
                "returns_by_quantile": qr_data["mean_returns"],
                "spread": qr_data["spread"],
            }
        
        return response
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in evaluate_factor: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error evaluating factor: {str(e)}")


@app.get("/api/available-factors")
def get_available_factors():
    """Get list of available factors for evaluation."""
    try:
        tech_path = Path("data/cache/technical_features.parquet")
        
        if not tech_path.exists():
            return {"factors": [], "error": "Technical features not found"}
        
        # Load schema to get column names
        lf = pl.scan_parquet(tech_path)
        schema = lf.schema
        
        # Exclude non-factor columns
        exclude_cols = {'ticker', 'date', 'year', 'month', 'open', 'high', 'low', 'close', 'volume'}
        
        factors = []
        for col, dtype in schema.items():
            if col not in exclude_cols and dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                factors.append({
                    "name": col,
                    "type": str(dtype),
                })
        
        return {"factors": factors}
        
    except Exception as e:
        logger.error(f"Error getting available factors: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Backtesting Endpoints
# ============================================

class BacktestRequest(BaseModel):
    strategy: str = "macd"  # macd, rsi
    ticker: Optional[str] = None
    # MACD parameters
    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    # RSI parameters
    rsi_period: int = 14
    oversold: float = 30
    overbought: float = 70
    # Common parameters
    initial_cash: float = 100000
    fees: float = 0.001
    slippage: float = 0.001


@app.post("/api/backtest")
def run_backtest(request: BacktestRequest):
    """
    Run a backtest using VectorBT.
    
    Supported strategies:
    - macd: MACD histogram crossover
    - rsi: RSI mean reversion
    """
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        
        if not daily_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Daily data not found. Run data processing first."
            )
        
        # Import backtest module
        from data_processing.backtest_vectorbt import load_ohlcv_data, backtest_macd_strategy, backtest_rsi_strategy
        
        # Load data
        data = load_ohlcv_data(str(daily_path), ticker=request.ticker)
        
        if len(data) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for ticker: {request.ticker}" if request.ticker else "No data found"
            )
        
        # Run backtest
        if request.strategy.lower() == "macd":
            results = backtest_macd_strategy(
                data,
                fast_period=request.fast_period,
                slow_period=request.slow_period,
                signal_period=request.signal_period,
                initial_cash=request.initial_cash,
                fees=request.fees,
                slippage=request.slippage,
            )
        elif request.strategy.lower() == "rsi":
            results = backtest_rsi_strategy(
                data,
                rsi_period=request.rsi_period,
                oversold=request.oversold,
                overbought=request.overbought,
                initial_cash=request.initial_cash,
                fees=request.fees,
                slippage=request.slippage,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy}. Supported: macd, rsi"
            )
        
        # Clean up results for JSON serialization (remove large time series)
        if 'equity_curve' in results:
            del results['equity_curve']
        if 'drawdowns' in results:
            del results['drawdowns']
        if 'by_ticker' in results:
            # For multi-ticker results, clean each ticker
            for ticker_results in results['by_ticker'].values():
                if isinstance(ticker_results, dict):
                    if 'equity_curve' in ticker_results:
                        del ticker_results['equity_curve']
                    if 'drawdowns' in ticker_results:
                        del ticker_results['drawdowns']
        
        return results
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Missing dependency: {str(e)}")
    except Exception as e:
        logger.error(f"Error in backtest: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error running backtest: {str(e)}")


# ============================================
# Performance Analysis Endpoints
# ============================================

class PerformanceRequest(BaseModel):
    ticker: Optional[str] = None
    benchmark_ticker: Optional[str] = None


@app.post("/api/performance-analysis")
def analyze_performance(request: PerformanceRequest):
    """
    Analyze performance of a ticker or portfolio.
    
    Returns Sharpe ratio, drawdowns, and other risk metrics.
    """
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        
        if not daily_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Daily data not found. Run data processing first."
            )
        
        # Import performance module
        from data_processing.performance_analysis import load_and_analyze
        
        results = load_and_analyze(
            data_path=str(daily_path),
            ticker=request.ticker,
            benchmark_ticker=request.benchmark_ticker,
        )
        
        # Remove large time series for API response
        if 'time_series' in results:
            # Keep only summary, not full series
            ts = results['time_series']
            results['time_series'] = {
                'has_cumulative_returns': 'cumulative_returns' in ts,
                'has_rolling_sharpe': 'rolling_sharpe_60d' in ts,
                'has_drawdown': 'drawdown' in ts,
            }
        
        return results
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in performance analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing performance: {str(e)}")


@app.get("/api/ticker/{ticker}/performance")
def get_ticker_performance(ticker: str):
    """Get performance metrics for a specific ticker."""
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        
        if not daily_path.exists():
            raise HTTPException(status_code=404, detail="Daily data not found")
        
        from data_processing.performance_analysis import load_and_analyze
        
        results = load_and_analyze(
            data_path=str(daily_path),
            ticker=ticker,
        )
        
        # Return summary only
        return {
            "ticker": ticker,
            "summary": results.get("summary", {}),
            "risk_adjusted": results.get("risk_adjusted", {}),
            "drawdown": results.get("drawdown", {}),
            "trade_metrics": results.get("trade_metrics", {}),
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting ticker performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Factor Model Endpoints
# ============================================

class FactorExposureRequest(BaseModel):
    ticker: str
    model: str = "5-factor"  # "3-factor" or "5-factor"
    market_ticker: str = "SPY"


class CAPMRequest(BaseModel):
    ticker: str
    market_ticker: str = "SPY"
    rolling_window: int = 60


@app.get("/api/capm/{ticker}")
def get_capm_analysis(
    ticker: str,
    market_ticker: str = Query(default="SPY", description="Market benchmark ticker"),
    rolling_window: int = Query(default=60, description="Rolling window for beta"),
):
    """
    Get CAPM analysis for a ticker.
    
    Returns beta, alpha, R-squared, and rolling beta/alpha.
    """
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        
        if not daily_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Daily data not found. Run data processing first."
            )
        
        from data_processing.factor_models import analyze_ticker_capm
        
        results = analyze_ticker_capm(
            data_path=str(daily_path),
            ticker=ticker,
            market_ticker=market_ticker,
            rolling_window=rolling_window,
        )
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        # Remove large time series from response (keep only current values)
        response = {k: v for k, v in results.items() 
                   if k not in ['rolling_beta', 'rolling_alpha']}
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in CAPM analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing CAPM: {str(e)}")


@app.get("/api/rolling-beta/{ticker}")
def get_rolling_beta(
    ticker: str,
    market_ticker: str = Query(default="SPY", description="Market benchmark ticker"),
    window: int = Query(default=60, description="Rolling window size"),
):
    """
    Get rolling beta time series for a ticker.
    
    Returns beta values over time for charting.
    """
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        
        if not daily_path.exists():
            raise HTTPException(status_code=404, detail="Daily data not found")
        
        from data_processing.factor_models import (
            load_returns_data, get_market_returns, compute_rolling_beta
        )
        
        # Load data
        ticker_data = load_returns_data(str(daily_path), ticker)
        market_returns = get_market_returns(str(daily_path), market_ticker)
        
        # Compute rolling beta
        rolling_beta = compute_rolling_beta(
            ticker_data['returns'],
            market_returns,
            window=window,
        )
        
        # Convert to list of {date, value} for charting
        beta_series = [
            {"date": d.isoformat(), "beta": float(v)}
            for d, v in rolling_beta.dropna().items()
        ]
        
        return {
            "ticker": ticker,
            "market_ticker": market_ticker,
            "window": window,
            "data": beta_series,
            "current_beta": beta_series[-1]["beta"] if beta_series else None,
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting rolling beta: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/factor-exposure/{ticker}")
def get_factor_exposure(
    ticker: str,
    model: str = Query(default="5-factor", description="3-factor or 5-factor"),
):
    """
    Get Fama-French factor exposures for a ticker.
    
    Returns factor betas, alpha, and statistical significance.
    """
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        cache_dir = Path("data/cache")
        
        if not daily_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Daily data not found. Run data processing first."
            )
        
        from data_processing.factor_models import analyze_ticker_factors
        
        results = analyze_ticker_factors(
            data_path=str(daily_path),
            ticker=ticker,
            model=model,
            cache_dir=str(cache_dir),
        )
        
        if 'error' in results:
            raise HTTPException(status_code=400, detail=results['error'])
        
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in factor exposure analysis: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error analyzing factors: {str(e)}")


@app.get("/api/factor-comparison/{ticker}")
def get_factor_comparison(
    ticker: str,
    market_ticker: str = Query(default="SPY", description="Market benchmark ticker"),
):
    """
    Compare CAPM vs Fama-French models for a ticker.
    
    Returns side-by-side comparison of model results.
    """
    try:
        daily_path = Path("data/cache/daily_2025.parquet")
        cache_dir = Path("data/cache")
        
        if not daily_path.exists():
            raise HTTPException(status_code=404, detail="Daily data not found")
        
        from data_processing.factor_models import compare_factor_models
        
        results = compare_factor_models(
            data_path=str(daily_path),
            ticker=ticker,
            market_ticker=market_ticker,
            cache_dir=str(cache_dir),
        )
        
        return results
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in factor comparison: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Model Training Endpoints
# ============================================

class LinearModelRequest(BaseModel):
    model_type: str = "linear"  # linear, logistic
    regularization: Optional[str] = None  # ridge, lasso, elasticnet
    alpha: float = 1.0
    tune_alpha: bool = False
    n_splits: int = 5
    test_size: int = 60


@app.post("/api/train-linear-model")
def train_linear_model(request: LinearModelRequest):
    """
    Train a linear model for return prediction.
    
    Returns model metrics, coefficients, and statistical significance.
    """
    try:
        features_path = Path("data/cache/technical_features.parquet")
        
        if not features_path.exists():
            raise HTTPException(
                status_code=404,
                detail="Technical features not found. Run: python data_processing/build_technical_features.py"
            )
        
        from data_processing.train_ml_model import train_and_evaluate
        
        results = train_and_evaluate(
            data_path=str(features_path),
            model_type=request.model_type,
            n_splits=request.n_splits,
            test_size=request.test_size,
            compute_shap=False,  # No SHAP for linear models
            regularization=request.regularization,
            alpha=request.alpha,
            tune_alpha=request.tune_alpha,
        )
        
        # Remove model object (not serializable)
        response = {k: v for k, v in results.items() if k not in ['model', 'scaler']}
        
        return response
        
    except Exception as e:
        logger.error(f"Error training linear model: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


"""
DuckDB-based fast scoring for ticker ranking.

This module provides sub-second scoring queries on 636M+ rows by using
DuckDB's columnar query engine instead of Python loops.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta

try:
    import duckdb
except ImportError:
    duckdb = None

logger = logging.getLogger(__name__)

# Path to DuckDB database
PROJECT_ROOT = Path(__file__).parent.parent
DUCKDB_PATH = PROJECT_ROOT / "data" / "cache" / "stonks.duckdb"


def is_duckdb_available() -> bool:
    """Check if DuckDB is available and database exists."""
    if duckdb is None:
        return False
    return DUCKDB_PATH.exists()


def get_connection(read_only: bool = True) -> "duckdb.DuckDBPyConnection":
    """Get a DuckDB connection."""
    if duckdb is None:
        raise ImportError("DuckDB not installed. Run: pip install duckdb")
    if not DUCKDB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB database not found at {DUCKDB_PATH}. "
            "Run: python data_processing/build_duckdb_index.py"
        )
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def score_tickers_fast(
    months_back: int = 12,
    min_days: int = 60,
    min_avg_volume: int = 100000,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    criteria: Optional[List[Dict[str, Any]]] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """
    Score and rank tickers using DuckDB for fast computation.
    
    This replaces the slow Python loop with a single SQL query that
    computes all metrics in parallel using DuckDB's vectorized engine.
    
    Args:
        months_back: How many months of history to analyze
        min_days: Minimum trading days required
        min_avg_volume: Minimum average daily volume
        min_price: Minimum current price filter
        max_price: Maximum current price filter
        criteria: Scoring criteria with weights
        limit: Maximum results to return
        
    Returns:
        Dict with ranked_tickers and total count
    """
    if not is_duckdb_available():
        raise RuntimeError("DuckDB not available. Run: python data_processing/build_duckdb_index.py")
    
    conn = get_connection()
    
    # Calculate date cutoff
    cutoff_date = (datetime.now() - timedelta(days=months_back * 30)).date()
    
    # Build price filter clause
    price_filters = []
    if min_price is not None:
        price_filters.append(f"latest_close >= {min_price}")
    if max_price is not None:
        price_filters.append(f"latest_close <= {max_price}")
    price_filter_clause = " AND ".join(price_filters) if price_filters else "TRUE"
    
    # Main query: compute all metrics in one pass using SQL aggregations
    # This is MUCH faster than Python loops because DuckDB uses vectorized execution
    query = f"""
    WITH recent_data AS (
        -- Filter to recent data only (much faster than scanning full history)
        SELECT * FROM daily
        WHERE date >= '{cutoff_date}'
    ),
    ticker_metrics AS (
        SELECT 
            ticker,
            
            -- Basic counts
            COUNT(*) as days_analyzed,
            
            -- Price metrics (use last value)
            LAST(close ORDER BY date) as latest_close,
            FIRST(close ORDER BY date) as first_close,
            MIN(close) as min_price,
            MAX(close) as max_price,
            AVG(close) as avg_price,
            STDDEV(close) as price_std,
            
            -- Volume metrics
            AVG(volume) as avg_volume,
            MIN(volume) as min_volume,
            MAX(volume) as max_volume,
            
            -- Daily range metrics (in dollars)
            AVG(high - low) as avg_daily_range_dollars,
            STDDEV(high - low) as daily_range_std_dollars,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY high - low) as median_daily_range_dollars,
            MIN(high - low) as min_daily_range_dollars,
            MAX(high - low) as max_daily_range_dollars,
            
            -- Sweet spot range days ($0.20-$0.60)
            SUM(CASE WHEN (high - low) BETWEEN 0.20 AND 0.60 THEN 1 ELSE 0 END) as sweet_spot_range_days,
            
            -- Return metrics (if ret_d column exists)
            CASE WHEN MAX(CASE WHEN ret_d IS NOT NULL THEN 1 ELSE 0 END) = 1 
                 THEN AVG(ret_d) ELSE NULL END as avg_return,
            CASE WHEN MAX(CASE WHEN ret_d IS NOT NULL THEN 1 ELSE 0 END) = 1 
                 THEN STDDEV(ret_d) ELSE NULL END as volatility,
            
            -- Absolute return metrics
            CASE WHEN MAX(CASE WHEN abs_ret_d IS NOT NULL THEN 1 ELSE 0 END) = 1 
                 THEN AVG(abs_ret_d) ELSE NULL END as avg_daily_swing_pct,
            CASE WHEN MAX(CASE WHEN abs_ret_d IS NOT NULL THEN 1 ELSE 0 END) = 1 
                 THEN MAX(abs_ret_d) ELSE NULL END as max_daily_swing_pct,
            
            -- Large swing days (>= 20% move)
            CASE WHEN MAX(CASE WHEN abs_ret_d IS NOT NULL THEN 1 ELSE 0 END) = 1 
                 THEN SUM(CASE WHEN abs_ret_d >= 0.20 THEN 1 ELSE 0 END) ELSE 0 END as large_swing_days
            
        FROM recent_data
        GROUP BY ticker
        HAVING COUNT(*) >= {min_days}
           AND AVG(volume) >= {min_avg_volume}
    ),
    scored_tickers AS (
        SELECT 
            *,
            
            -- Derived metrics
            CASE WHEN first_close > 0 
                 THEN ((latest_close / first_close) - 1) * 100 
                 ELSE 0 END as total_return_pct,
            
            -- Sweet spot percentage
            (sweet_spot_range_days * 100.0 / days_analyzed) as sweet_spot_range_pct,
            
            -- Liquidity multiple (how many times 10k shares fits in daily volume)
            (avg_volume / 10000.0) as liquidity_multiple,
            
            -- Position as % of daily volume
            CASE WHEN avg_volume > 0 
                 THEN (10000.0 / avg_volume) * 100 
                 ELSE 100 END as position_pct_of_volume,
            
            -- Daily range coefficient of variation (lower = more consistent)
            CASE WHEN avg_daily_range_dollars > 0 
                 THEN daily_range_std_dollars / avg_daily_range_dollars 
                 ELSE 1.0 END as daily_range_cv,
            
            -- Large swing frequency
            CASE WHEN days_analyzed > 0 
                 THEN (large_swing_days * 1.0 / days_analyzed) 
                 ELSE 0 END as large_swing_frequency,
                 
            -- Profit potential score
            CASE WHEN avg_volume > 0 
                 THEN avg_daily_range_dollars * LEAST(avg_volume / 10000.0, 10) 
                 ELSE 0 END as profit_potential_score
            
        FROM ticker_metrics
        WHERE {price_filter_clause}
    ),
    final_scored AS (
        SELECT 
            *,
            
            -- Tradability score (0-100)
            -- 1. Liquidity score (max 40 points)
            LEAST(liquidity_multiple / 10.0, 1.0) * 40 +
            
            -- 2. Range score (max 40 points)
            CASE 
                WHEN avg_daily_range_dollars BETWEEN 0.20 AND 0.60 THEN 40
                WHEN avg_daily_range_dollars BETWEEN 0.10 AND 0.20 
                     OR avg_daily_range_dollars BETWEEN 0.60 AND 1.00 THEN 25
                WHEN avg_daily_range_dollars < 0.10 THEN 5
                ELSE 10 
            END +
            
            -- 3. Consistency score (max 20 points)
            CASE 
                WHEN daily_range_cv < 0.3 THEN 20
                WHEN daily_range_cv < 0.5 THEN 15
                WHEN daily_range_cv < 0.7 THEN 10
                ELSE 5 
            END as tradability_score
            
        FROM scored_tickers
    )
    SELECT * FROM final_scored
    ORDER BY tradability_score DESC, avg_volume DESC
    LIMIT {limit}
    """
    
    try:
        start_time = datetime.now()
        result_df = conn.execute(query).fetchdf()
        query_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"DuckDB scoring query completed in {query_time:.3f}s, returned {len(result_df)} tickers")
        
        # Convert to response format
        ranked_tickers = []
        
        # Default criteria if not provided
        if criteria is None:
            criteria = [
                {"name": "tradability_score", "weight": 3.0, "min_value": 0, "max_value": 100, "invert": False},
                {"name": "avg_daily_range_dollars", "weight": 2.5, "min_value": 0.10, "max_value": 1.00, "invert": False},
                {"name": "liquidity_multiple", "weight": 2.0, "min_value": 0, "max_value": 100, "invert": False},
                {"name": "sweet_spot_range_pct", "weight": 1.5, "min_value": 0, "max_value": 100, "invert": False},
                {"name": "daily_range_cv", "weight": 1.0, "min_value": 0, "max_value": 1.0, "invert": True},
                {"name": "latest_close", "weight": 10.0, "min_value": 0, "max_value": 8, "invert": False},
            ]
        
        for _, row in result_df.iterrows():
            # Build metrics dict
            metrics = {
                "days_analyzed": int(row["days_analyzed"]),
                "current_price": float(row["latest_close"]) if row["latest_close"] else 0,
                "min_price": float(row["min_price"]) if row["min_price"] else 0,
                "max_price": float(row["max_price"]) if row["max_price"] else 0,
                "avg_volume": int(row["avg_volume"]) if row["avg_volume"] else 0,
                "min_volume": int(row["min_volume"]) if row["min_volume"] else 0,
                "max_volume": int(row["max_volume"]) if row["max_volume"] else 0,
                "avg_daily_range_dollars": float(row["avg_daily_range_dollars"]) if row["avg_daily_range_dollars"] else 0,
                "median_daily_range_dollars": float(row["median_daily_range_dollars"]) if row["median_daily_range_dollars"] else 0,
                "daily_range_std_dollars": float(row["daily_range_std_dollars"]) if row["daily_range_std_dollars"] else 0,
                "daily_range_cv": float(row["daily_range_cv"]) if row["daily_range_cv"] else 1.0,
                "sweet_spot_range_days": int(row["sweet_spot_range_days"]) if row["sweet_spot_range_days"] else 0,
                "sweet_spot_range_pct": float(row["sweet_spot_range_pct"]) if row["sweet_spot_range_pct"] else 0,
                "liquidity_multiple": float(row["liquidity_multiple"]) if row["liquidity_multiple"] else 0,
                "position_pct_of_volume": float(row["position_pct_of_volume"]) if row["position_pct_of_volume"] else 100,
                "profit_potential_score": float(row["profit_potential_score"]) if row["profit_potential_score"] else 0,
                "tradability_score": float(row["tradability_score"]) if row["tradability_score"] else 0,
                "total_return_pct": float(row["total_return_pct"]) if row["total_return_pct"] else 0,
            }
            
            # Add optional metrics if available
            if row["volatility"] is not None:
                metrics["volatility"] = float(row["volatility"])
            if row["avg_daily_swing_pct"] is not None:
                metrics["avg_daily_swing_pct"] = float(row["avg_daily_swing_pct"])
            if row["max_daily_swing_pct"] is not None:
                metrics["max_daily_swing_pct"] = float(row["max_daily_swing_pct"])
            if row["large_swing_days"] is not None:
                metrics["large_swing_days"] = int(row["large_swing_days"])
            if row["large_swing_frequency"] is not None:
                metrics["large_swing_frequency"] = float(row["large_swing_frequency"])
            
            # Compute weighted score based on criteria
            total_score = 0.0
            criterion_scores = {}
            
            for criterion in criteria:
                # Map criterion name to actual column
                col_name = criterion["name"]
                if col_name == "current_price":
                    col_name = "latest_close"
                
                value = metrics.get(col_name if col_name != "latest_close" else "current_price")
                if value is None:
                    continue
                
                # Normalize value to 0-1 range
                min_val = criterion.get("min_value", 0)
                max_val = criterion.get("max_value", 100)
                
                if max_val > min_val:
                    normalized = (value - min_val) / (max_val - min_val)
                    normalized = max(0.0, min(1.0, normalized))
                else:
                    normalized = 0.5
                
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
            
            ranked_tickers.append({
                "ticker": row["ticker"],
                "total_score": total_score,
                "metrics": metrics,
                "criterion_scores": criterion_scores,
            })
        
        # Sort by total score
        ranked_tickers.sort(key=lambda x: x["total_score"], reverse=True)
        
        conn.close()
        
        return {
            "ranked_tickers": ranked_tickers,
            "total": len(ranked_tickers),
            "query_time_seconds": query_time,
            "source": "duckdb",
        }
        
    except Exception as e:
        conn.close()
        logger.error(f"DuckDB scoring query failed: {e}")
        raise


def get_ticker_details_fast(ticker: str, months_back: int = 12) -> Dict[str, Any]:
    """
    Get detailed metrics for a specific ticker using DuckDB.
    
    Args:
        ticker: Stock ticker symbol
        months_back: Months of history to analyze
        
    Returns:
        Dict with detailed metrics
    """
    if not is_duckdb_available():
        raise RuntimeError("DuckDB not available")
    
    conn = get_connection()
    cutoff_date = (datetime.now() - timedelta(days=months_back * 30)).date()
    
    query = f"""
    SELECT 
        ticker,
        date,
        open,
        high,
        low,
        close,
        volume,
        (high - low) as daily_range
    FROM daily
    WHERE ticker = '{ticker}'
      AND date >= '{cutoff_date}'
    ORDER BY date DESC
    LIMIT 500
    """
    
    result_df = conn.execute(query).fetchdf()
    conn.close()
    
    if len(result_df) == 0:
        return {"error": f"No data found for ticker {ticker}"}
    
    return {
        "ticker": ticker,
        "data_points": len(result_df),
        "latest_date": result_df["date"].iloc[0].isoformat() if len(result_df) > 0 else None,
        "latest_close": float(result_df["close"].iloc[0]) if len(result_df) > 0 else None,
        "price_history": result_df[["date", "close", "volume"]].head(30).to_dict("records"),
    }


def search_tickers_fast(
    query: str,
    min_volume: int = 0,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fast ticker search using DuckDB.
    
    Args:
        query: Search query (ticker pattern)
        min_volume: Minimum average volume
        min_price: Minimum price
        max_price: Maximum price
        limit: Max results
        
    Returns:
        List of matching tickers with basic info
    """
    if not is_duckdb_available():
        raise RuntimeError("DuckDB not available")
    
    conn = get_connection()
    
    # Build filters
    filters = [f"ticker LIKE '{query.upper()}%'"]
    if min_volume > 0:
        filters.append(f"avg_volume >= {min_volume}")
    if min_price is not None:
        filters.append(f"latest_close >= {min_price}")
    if max_price is not None:
        filters.append(f"latest_close <= {max_price}")
    
    filter_clause = " AND ".join(filters)
    
    sql = f"""
    WITH recent AS (
        SELECT * FROM daily
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
    )
    SELECT 
        ticker,
        LAST(close ORDER BY date) as latest_close,
        AVG(volume) as avg_volume,
        COUNT(*) as days
    FROM recent
    GROUP BY ticker
    HAVING {filter_clause}
    ORDER BY avg_volume DESC
    LIMIT {limit}
    """
    
    result_df = conn.execute(sql).fetchdf()
    conn.close()
    
    return [
        {
            "ticker": row["ticker"],
            "latest_close": float(row["latest_close"]) if row["latest_close"] else 0,
            "avg_volume": int(row["avg_volume"]) if row["avg_volume"] else 0,
        }
        for _, row in result_df.iterrows()
    ]

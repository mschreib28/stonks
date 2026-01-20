#!/usr/bin/env python3
"""
Build a DuckDB database from Parquet files for fast indexed queries.

DuckDB is a columnar database optimized for analytical queries (OLAP).
It works directly with Parquet files and provides:
- Fast aggregate queries on billions of rows
- SQL interface with proper indexing
- Memory-efficient streaming for large datasets
- Direct Parquet file reading without loading into memory

Usage:
    python build_duckdb_index.py                    # Build from default datasets
    python build_duckdb_index.py --rebuild          # Force rebuild
    python build_duckdb_index.py --query "SELECT * FROM daily LIMIT 10"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta

try:
    import duckdb
except ImportError:
    print("DuckDB not installed. Install with: pip install duckdb")
    print("Or: uv add duckdb")
    sys.exit(1)

import polars as pl


# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_CACHE = PROJECT_ROOT / "data" / "cache"
DUCKDB_PATH = DATA_CACHE / "stonks.duckdb"


def get_connection(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Get a DuckDB connection."""
    return duckdb.connect(str(DUCKDB_PATH), read_only=read_only)


def build_database(force: bool = False) -> None:
    """Build DuckDB database from Parquet files."""
    
    if DUCKDB_PATH.exists() and not force:
        print(f"Database already exists: {DUCKDB_PATH}")
        print("Use --rebuild to force rebuild")
        return
    
    print(f"Building DuckDB database: {DUCKDB_PATH}")
    
    # Create fresh database
    if DUCKDB_PATH.exists():
        DUCKDB_PATH.unlink()
    
    conn = get_connection()
    
    # Define datasets to index
    datasets = [
        ("daily", "daily_2025.parquet"),
        ("weekly", "weekly.parquet"),
        ("filtered", "daily_filtered_1_9.99.parquet"),
        ("filtered_scored", "daily_filtered_scored.parquet"),
        ("minute_features", "minute_features.parquet"),
        ("scoring_index", "scoring_index_default.parquet"),
    ]
    
    for table_name, filename in datasets:
        parquet_path = DATA_CACHE / filename
        if not parquet_path.exists():
            print(f"  ⏭️  Skipping {table_name}: {filename} not found")
            continue
        
        print(f"  📦 Creating table '{table_name}' from {filename}...")
        
        # Create table from Parquet - DuckDB handles this efficiently
        conn.execute(f"""
            CREATE TABLE {table_name} AS 
            SELECT * FROM read_parquet('{parquet_path}')
        """)
        
        # Get row count
        result = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        row_count = result[0] if result else 0
        
        print(f"     ✅ {row_count:,} rows loaded")
    
    # Create indexes for common query patterns
    print("\n📊 Creating indexes...")
    
    # Index on ticker for fast lookups
    for table in ["daily", "weekly", "filtered", "filtered_scored"]:
        try:
            conn.execute(f"CREATE INDEX idx_{table}_ticker ON {table}(ticker)")
            print(f"  ✅ Index on {table}(ticker)")
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  ⚠️  Could not create index on {table}(ticker): {e}")
    
    # Composite index on ticker + date for time-series queries
    for table in ["daily", "weekly", "filtered"]:
        try:
            if table == "weekly":
                conn.execute(f"CREATE INDEX idx_{table}_ticker_date ON {table}(ticker, week_start)")
            else:
                conn.execute(f"CREATE INDEX idx_{table}_ticker_date ON {table}(ticker, date)")
            print(f"  ✅ Composite index on {table}(ticker, date)")
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"  ⚠️  Could not create composite index on {table}: {e}")
    
    # Create pre-aggregated views for common queries
    print("\n📈 Creating pre-aggregated views...")
    
    # Ticker summary view with latest data
    try:
        conn.execute("""
            CREATE OR REPLACE VIEW ticker_summary AS
            SELECT 
                ticker,
                COUNT(*) as total_days,
                MIN(date) as first_date,
                MAX(date) as last_date,
                AVG(volume) as avg_volume,
                AVG(close) as avg_price,
                MIN(close) as min_price,
                MAX(close) as max_price,
                STDDEV(close) as price_stddev
            FROM daily
            GROUP BY ticker
        """)
        print("  ✅ Created ticker_summary view")
    except Exception as e:
        print(f"  ⚠️  Could not create ticker_summary view: {e}")
    
    # Recent data view (last 3 months)
    try:
        conn.execute("""
            CREATE OR REPLACE VIEW recent_daily AS
            SELECT * FROM daily
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
        """)
        print("  ✅ Created recent_daily view")
    except Exception as e:
        print(f"  ⚠️  Could not create recent_daily view: {e}")
    
    # High volume tickers view
    try:
        conn.execute("""
            CREATE OR REPLACE VIEW high_volume_tickers AS
            SELECT 
                ticker,
                AVG(volume) as avg_volume,
                COUNT(*) as days_count
            FROM daily
            WHERE date >= CURRENT_DATE - INTERVAL '90 days'
            GROUP BY ticker
            HAVING AVG(volume) >= 100000
            ORDER BY avg_volume DESC
        """)
        print("  ✅ Created high_volume_tickers view")
    except Exception as e:
        print(f"  ⚠️  Could not create high_volume_tickers view: {e}")
    
    conn.close()
    
    # Print database size
    db_size = DUCKDB_PATH.stat().st_size / (1024 * 1024)
    print(f"\n✅ Database built successfully!")
    print(f"   Path: {DUCKDB_PATH}")
    print(f"   Size: {db_size:.1f} MB")


def run_query(query: str) -> None:
    """Run a SQL query and print results."""
    if not DUCKDB_PATH.exists():
        print("Database not found. Run without --query first to build it.")
        return
    
    conn = get_connection(read_only=True)
    
    print(f"Query: {query}\n")
    
    start = datetime.now()
    result = conn.execute(query).fetchdf()
    elapsed = (datetime.now() - start).total_seconds()
    
    print(result.to_string())
    print(f"\n⏱️  Query executed in {elapsed:.3f} seconds")
    print(f"📊 {len(result)} rows returned")
    
    conn.close()


def demo_queries() -> None:
    """Run demo queries to show DuckDB capabilities."""
    if not DUCKDB_PATH.exists():
        print("Database not found. Building first...")
        build_database()
    
    conn = get_connection(read_only=True)
    
    print("=" * 60)
    print("DuckDB Demo Queries")
    print("=" * 60)
    
    queries = [
        ("Count total rows in daily table", 
         "SELECT COUNT(*) as total_rows FROM daily"),
        
        ("Get ticker count",
         "SELECT COUNT(DISTINCT ticker) as unique_tickers FROM daily"),
        
        ("Top 10 tickers by average volume (last 90 days)",
         """
         SELECT ticker, 
                CAST(AVG(volume) AS BIGINT) as avg_volume,
                COUNT(*) as days
         FROM daily 
         WHERE date >= CURRENT_DATE - INTERVAL '90 days'
         GROUP BY ticker 
         HAVING COUNT(*) >= 40
         ORDER BY avg_volume DESC 
         LIMIT 10
         """),
        
        ("Tickers with price between $1-$10 and volume > 500k",
         """
         SELECT ticker,
                ROUND(AVG(close), 2) as avg_price,
                CAST(AVG(volume) AS BIGINT) as avg_volume,
                COUNT(*) as days
         FROM daily
         WHERE date >= CURRENT_DATE - INTERVAL '90 days'
         GROUP BY ticker
         HAVING AVG(close) BETWEEN 1 AND 10
            AND AVG(volume) > 500000
            AND COUNT(*) >= 40
         ORDER BY avg_volume DESC
         LIMIT 20
         """),
        
        ("Quick scoring candidates (sub-second query!)",
         """
         SELECT 
            ticker,
            ROUND(AVG(close), 2) as avg_price,
            CAST(AVG(volume) AS BIGINT) as avg_volume,
            ROUND(AVG(high - low), 3) as avg_daily_range,
            COUNT(*) as days
         FROM daily
         WHERE date >= CURRENT_DATE - INTERVAL '90 days'
         GROUP BY ticker
         HAVING AVG(close) BETWEEN 1 AND 10
            AND AVG(volume) > 100000
            AND COUNT(*) >= 40
         ORDER BY AVG(high - low) / AVG(close) DESC
         LIMIT 50
         """),
    ]
    
    for title, query in queries:
        print(f"\n{'─' * 60}")
        print(f"📌 {title}")
        print(f"{'─' * 60}")
        
        start = datetime.now()
        try:
            result = conn.execute(query).fetchdf()
            elapsed = (datetime.now() - start).total_seconds()
            
            print(result.to_string())
            print(f"\n⏱️  {elapsed:.3f}s | {len(result)} rows")
        except Exception as e:
            print(f"Error: {e}")
    
    conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and query DuckDB database for fast stock data analysis"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of database even if it exists",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Run a SQL query against the database",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo queries to showcase DuckDB speed",
    )
    
    args = parser.parse_args()
    
    if args.query:
        run_query(args.query)
    elif args.demo:
        demo_queries()
    else:
        build_database(force=args.rebuild)


if __name__ == "__main__":
    main()

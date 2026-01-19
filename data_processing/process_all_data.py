#!/usr/bin/env python3
"""
Load and process all data with recommended defaults.

This script runs incrementally by default - it only processes new data that
hasn't been processed yet, based on date comparisons between input files and
existing cache files.

Processing steps:
1. Processes daily/weekly aggregates (incremental - checks dates)
2. Processes minute-level features (incremental - checks dates)
3. Processes MACD features (incremental - skips existing dates)
4. Joins MACD with minute features
5. Processes technical indicator features (incremental - checks dates)
6. Builds scoring index for default criteria (incremental - checks dates)

Use --force to reprocess everything regardless of existing outputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# Default paths - cache is shared across all years, raw data is partitioned by year
CACHE_DIR = Path("data/cache")

def get_available_years() -> list[int]:
    """Get all available years from data directories."""
    years = []
    data_root = Path("data")
    if data_root.exists():
        for year_dir in sorted(data_root.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                years.append(int(year_dir.name))
    return sorted(years)

def get_daily_output_filename() -> str:
    """Get dynamic daily output filename based on available years."""
    years = get_available_years()
    if not years:
        # Fallback to current year if no data directories found
        from datetime import datetime
        current_year = datetime.now().year
        return f"daily_{current_year}.parquet"
    
    if len(years) == 1:
        return f"daily_{years[0]}.parquet"
    else:
        # Multiple years: use range or "all"
        min_year = min(years)
        max_year = max(years)
        # If years span more than 2, use "all"
        if max_year - min_year > 1:
            return "daily_all.parquet"
        else:
            return f"daily_{min_year}_{max_year}.parquet"

def get_all_daily_inputs() -> list[Path]:
    """Get all daily input directories from all year directories."""
    inputs = []
    data_root = Path("data")
    if data_root.exists():
        for year_dir in sorted(data_root.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                daily_dir = year_dir / "polygon_day_aggs"
                if daily_dir.exists():
                    inputs.append(daily_dir)
    return inputs

def get_all_minute_inputs() -> list[Path]:
    """Get all minute input directories from all year directories."""
    inputs = []
    data_root = Path("data")
    if data_root.exists():
        for year_dir in sorted(data_root.iterdir()):
            if year_dir.is_dir() and year_dir.name.isdigit():
                minute_dir = year_dir / "polygon_minute_aggs"
                if minute_dir.exists():
                    inputs.append(minute_dir)
    return inputs

# Output files
# Dynamic output filename based on available years
# Will be set in process_daily_weekly() based on actual data
DAILY_OUTPUT = CACHE_DIR / "daily_all.parquet"  # Default, will be updated dynamically
WEEKLY_OUTPUT = CACHE_DIR / "weekly.parquet"
FILTERED_OUTPUT = CACHE_DIR / "daily_filtered_1_9.99.parquet"  # Legacy price-filtered
FILTERED_SCORED_OUTPUT = CACHE_DIR / "daily_filtered_scored.parquet"  # New scored-filtered
MINUTE_FEATURES_OUTPUT = CACHE_DIR / "minute_features.parquet"
MACD_INC_DIR = CACHE_DIR / "macd_day_features_inc" / "mode=all"
MINUTE_MACD_OUTPUT = CACHE_DIR / "minute_features_plus_macd.parquet"
TECHNICAL_FEATURES_OUTPUT = CACHE_DIR / "technical_features.parquet"
SCORING_INDEX_OUTPUT = CACHE_DIR / "scoring_index_default.parquet"


def check_input_files(input_dir: Path, file_type: str) -> tuple[bool, int]:
    """Check if input CSV files exist."""
    if not input_dir.exists():
        return False, 0
    
    csv_files = list(input_dir.rglob("*.csv.gz"))
    count = len(csv_files)
    exists = count > 0
    
    return exists, count


def check_output_exists(output_path: Path) -> bool:
    """Check if output file/directory exists."""
    if output_path.is_dir():
        # For directories, check if it has parquet files
        parquet_files = list(output_path.rglob("*.parquet"))
        return len(parquet_files) > 0
    return output_path.exists()

def get_latest_date_in_cache(cache_path: Path) -> str | None:
    """Get the latest date from an existing cache file."""
    if not cache_path.exists() or not HAS_POLARS:
        return None
    
    try:
        df = pl.scan_parquet(cache_path)
        schema_names = df.collect_schema().names()
        if "date" in schema_names:
            max_date = df.select(pl.col("date").max()).collect()
            if max_date.height > 0 and max_date[0, 0] is not None:
                return str(max_date[0, 0])
    except Exception:
        pass
    return None

def get_latest_date_in_inputs(input_dirs: list[Path]) -> str | None:
    """Get the latest date from input CSV files."""
    latest_date = None
    for input_dir in input_dirs:
        for csv_file in input_dir.rglob("*.csv.gz"):
            # Extract date from filename: YYYY-MM-DD.csv.gz
            date_str = csv_file.stem.replace(".csv", "")
            if len(date_str) == 10:  # YYYY-MM-DD format
                if latest_date is None or date_str > latest_date:
                    latest_date = date_str
    return latest_date

def needs_rebuild(cache_path: Path, input_dirs: list[Path]) -> bool:
    """Check if cache needs to be rebuilt based on latest dates."""
    cache_latest = get_latest_date_in_cache(cache_path)
    input_latest = get_latest_date_in_inputs(input_dirs)
    
    if cache_latest is None:
        return True  # No cache exists
    
    if input_latest is None:
        return False  # No input files
    
    # Rebuild if input has newer dates
    return input_latest > cache_latest


def run_command(cmd: list[str], description: str, skip_if_exists: Optional[Path] = None) -> bool:
    """Run a command, skipping if output already exists."""
    if skip_if_exists and check_output_exists(skip_if_exists):
        print(f"⏭️  Skipping {description} (output already exists: {skip_if_exists})")
        return True
    
    print(f"🔄 Running: {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        print(f"✅ Completed: {description}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}")
        print(f"   Error: {e.stderr}")
        return False


def process_daily_weekly(force: bool = False) -> bool:
    """Process daily and weekly aggregates from all year directories."""
    print("=" * 60)
    print("Step 1: Processing Daily and Weekly Aggregates")
    print("=" * 60)
    
    # Check inputs from all years
    daily_inputs = get_all_daily_inputs()
    if not daily_inputs:
        print(f"⚠️  Warning: No daily CSV files found in any year directory")
        print("   Please download data first or check the path.")
        return False
    
    total_count = 0
    for input_dir in daily_inputs:
        _, count = check_input_files(input_dir, "daily")
        total_count += count
        print(f"✓ Found {count} daily CSV files in {input_dir}")
    
    print(f"✓ Total: {total_count} daily CSV files\n")
    
    # Determine output filename based on available years
    global DAILY_OUTPUT
    daily_output_filename = get_daily_output_filename()
    DAILY_OUTPUT = CACHE_DIR / daily_output_filename
    print(f"✓ Output file: {DAILY_OUTPUT}")
    
    # Process daily/weekly - combine data from all year directories
    # The build_polygon_cache.py script uses input_root/**/*.csv.gz glob pattern
    # By pointing to the parent 'data/' directory, the glob will find files in
    # data/2025/polygon_day_aggs/**/*.csv.gz, data/2026/polygon_day_aggs/**/*.csv.gz, etc.
    # Note: --prefilter-min-avg-volume and --prefilter-min-days now have defaults in build_polygon_cache.py
    # matching the Swing Trading Default preset (750k volume, 60 days)
    data_root = Path("data")
    cmd = [
        sys.executable,
        "data_processing/build_polygon_cache.py",
        "--input-root", str(data_root),
        "--out-daily", str(DAILY_OUTPUT),
        "--add-returns",
        "--build-weekly",
        # Defaults are now in build_polygon_cache.py: min_avg_volume=750000, min_days=60
    ]
    
    if len(daily_inputs) > 1:
        years = [d.parent.name for d in daily_inputs]
        print(f"✓ Processing data from {len(daily_inputs)} year directories: {years}")
    
    if not force:
        # Check if we need to rebuild based on dates
        cache_latest = get_latest_date_in_cache(DAILY_OUTPUT)
        input_latest = get_latest_date_in_inputs(daily_inputs)
        
        if cache_latest and input_latest:
            print(f"   Cache latest date: {cache_latest}")
            print(f"   Input latest date: {input_latest}")
            if input_latest <= cache_latest:
                print(f"⏭️  Skipping daily/weekly processing (cache is up to date)")
                return True
            else:
                print(f"🔄 Cache needs update (new data available)")
        elif check_output_exists(DAILY_OUTPUT) and check_output_exists(WEEKLY_OUTPUT) and check_output_exists(FILTERED_OUTPUT):
            print(f"⏭️  Skipping daily/weekly processing (outputs already exist)")
            return True
    
    return run_command(cmd, "Daily and Weekly Aggregates")


def process_minute_features(force: bool = False) -> bool:
    """Process minute-level features."""
    print("=" * 60)
    print("Step 2: Processing Minute-Level Features")
    print("=" * 60)
    
    # Check inputs from all years
    daily_inputs = get_all_daily_inputs()
    minute_inputs = get_all_minute_inputs()
    
    if not daily_inputs:
        print(f"⚠️  Warning: No daily CSV files found in any year directory")
        return False
    
    if not minute_inputs:
        print(f"⚠️  Warning: No minute CSV files found in any year directory")
        print("   Skipping minute features processing.")
        return False
    
    daily_count = sum(check_input_files(d, "daily")[1] for d in daily_inputs)
    minute_count = sum(check_input_files(m, "minute")[1] for m in minute_inputs)
    
    print(f"✓ Found {daily_count} daily CSV files")
    print(f"✓ Found {minute_count} minute CSV files\n")
    
    # Process minute features - combine data from all year directories
    # Point to parent data/ directory so glob finds files from all years
    data_root = Path("data")
    cmd = [
        sys.executable,
        "data_processing/build_polygon_cache.py",
        "--input-root", str(data_root),
        "--input-minute-root", str(data_root),
        "--add-returns",
        "--build-minute-features",
    ]
    
    if len(daily_inputs) > 1 or len(minute_inputs) > 1:
        print(f"✓ Processing minute features from {len(minute_inputs)} year directories: {[m.parent.name for m in minute_inputs]}")
    
    if not force:
        # Check if we need to rebuild based on dates
        cache_latest = get_latest_date_in_cache(MINUTE_FEATURES_OUTPUT)
        input_latest = get_latest_date_in_inputs(minute_inputs)
        
        if cache_latest and input_latest:
            print(f"   Cache latest date: {cache_latest}")
            print(f"   Input latest date: {input_latest}")
            if input_latest <= cache_latest:
                print(f"⏭️  Skipping minute features processing (cache is up to date)")
                return True
            else:
                print(f"🔄 Cache needs update (new data available)")
        elif check_output_exists(MINUTE_FEATURES_OUTPUT):
            print(f"⏭️  Skipping minute features processing (output already exists)")
            return True
    
    return run_command(cmd, "Minute-Level Features", skip_if_exists=MINUTE_FEATURES_OUTPUT if not force else None)


def process_macd_features(force: bool = False) -> bool:
    """Process MACD features incrementally."""
    print("=" * 60)
    print("Step 3: Processing MACD Features (Incremental)")
    print("=" * 60)
    
    # Check inputs from all years
    minute_inputs = get_all_minute_inputs()
    if not minute_inputs:
        print(f"⚠️  Warning: No minute CSV files found in any year directory")
        print("   Skipping MACD features processing.")
        return False
    
    minute_count = sum(check_input_files(m, "minute")[1] for m in minute_inputs)
    print(f"✓ Found {minute_count} minute CSV files across {len(minute_inputs)} year directories\n")
    
    # Check existing files once before processing
    if not force and check_output_exists(MACD_INC_DIR):
        parquet_count = len(list(MACD_INC_DIR.rglob("*.parquet")))
        print(f"✓ Found {parquet_count} existing MACD feature files")
        print(f"   Running incremental update (will skip existing dates)...\n")
    
    # Process MACD features incrementally - process each year separately
    # The incremental script processes files one by one and skips existing files,
    # so we can safely run it for each year directory
    success = True
    for minute_input in minute_inputs:
        year = minute_input.parent.name
        print(f"Processing MACD features for year {year}...")
        
        cmd = [
            sys.executable,
            "data_processing/build_macd_day_features_incremental.py",
            "--input-root", str(minute_input),
            "--out-root", str(CACHE_DIR / "macd_day_features_inc"),
            "--mode", "all",
        ]
        
        result = run_command(cmd, f"MACD Features for {year} (Incremental)")
        if not result:
            success = False
            print(f"⚠️  Warning: MACD processing failed for year {year}")
        print()  # Add blank line between years
    
    return success


def join_macd_with_minute_features(force: bool = False) -> bool:
    """Join MACD features with minute features."""
    print("=" * 60)
    print("Step 4: Joining MACD with Minute Features")
    print("=" * 60)
    
    # Check prerequisites
    if not check_output_exists(MINUTE_FEATURES_OUTPUT):
        print(f"⚠️  Error: Minute features not found: {MINUTE_FEATURES_OUTPUT}")
        print("   Please run minute features processing first.")
        return False
    
    if not check_output_exists(MACD_INC_DIR):
        print(f"⚠️  Error: MACD features not found: {MACD_INC_DIR}")
        print("   Please run MACD features processing first.")
        return False
    
    print(f"✓ Minute features found: {MINUTE_FEATURES_OUTPUT}")
    print(f"✓ MACD features found: {MACD_INC_DIR}\n")
    
    # Join MACD with minute features
    cmd = [
        sys.executable,
        "data_processing/join_macd_with_volitility_dataset.py",
    ]
    
    if not force and check_output_exists(MINUTE_MACD_OUTPUT):
        print(f"⏭️  Skipping join (output already exists)")
        return True
    
    return run_command(cmd, "Join MACD with Minute Features", skip_if_exists=MINUTE_MACD_OUTPUT if not force else None)


def process_technical_features(force: bool = False) -> bool:
    """Process technical indicator features from daily data."""
    print("=" * 60)
    print("Step 5: Processing Technical Indicator Features")
    print("=" * 60)
    
    # Check prerequisite
    if not check_output_exists(DAILY_OUTPUT):
        print(f"⚠️  Error: Daily data not found: {DAILY_OUTPUT}")
        print("   Please run daily processing first.")
        return False
    
    print(f"✓ Daily data found: {DAILY_OUTPUT}\n")
    
    # Process technical features
    cmd = [
        sys.executable,
        "data_processing/build_technical_features.py",
        "--daily-path", str(DAILY_OUTPUT),
        "--output", str(TECHNICAL_FEATURES_OUTPUT),
        "--n-jobs", "8",
    ]
    
    if not force:
        # Check if we need to rebuild based on dates
        # Technical features depend on daily data, so check if daily data is newer
        cache_latest = get_latest_date_in_cache(TECHNICAL_FEATURES_OUTPUT)
        daily_latest = get_latest_date_in_cache(DAILY_OUTPUT)
        
        if cache_latest and daily_latest:
            print(f"   Technical features cache latest date: {cache_latest}")
            print(f"   Daily data latest date: {daily_latest}")
            if daily_latest <= cache_latest:
                print(f"⏭️  Skipping technical features processing (cache is up to date)")
                return True
            else:
                print(f"🔄 Cache needs update (daily data has newer dates)")
        elif check_output_exists(TECHNICAL_FEATURES_OUTPUT):
            print(f"⏭️  Skipping technical features (output already exists)")
            return True
    
    return run_command(cmd, "Technical Indicator Features", skip_if_exists=TECHNICAL_FEATURES_OUTPUT if not force else None)


def build_scoring_index(force: bool = False) -> bool:
    """Build precomputed scoring index using default criteria."""
    print("=" * 60)
    print("Step 6: Building Scoring Index")
    print("=" * 60)
    
    # Check prerequisite
    if not check_output_exists(FILTERED_OUTPUT):
        print(f"⚠️  Error: Filtered dataset not found: {FILTERED_OUTPUT}")
        print("   Please run daily processing first.")
        return False
    
    print(f"✓ Filtered dataset found: {FILTERED_OUTPUT}\n")
    
    # Build scoring index
    cmd = [
        sys.executable,
        "data_processing/build_scoring_index.py",
        "--dataset", str(FILTERED_OUTPUT),
        "--output", str(SCORING_INDEX_OUTPUT),
    ]
    
    if not force:
        # Check if we need to rebuild based on dates
        cache_latest = get_latest_date_in_cache(SCORING_INDEX_OUTPUT)
        filtered_latest = get_latest_date_in_cache(FILTERED_OUTPUT)
        
        if cache_latest and filtered_latest:
            print(f"   Scoring index latest date: {cache_latest}")
            print(f"   Filtered dataset latest date: {filtered_latest}")
            if filtered_latest <= cache_latest:
                print(f"⏭️  Skipping scoring index build (index is up to date)")
                return True
            else:
                print(f"🔄 Index needs update (filtered dataset has newer dates)")
        elif check_output_exists(SCORING_INDEX_OUTPUT):
            print(f"⏭️  Skipping scoring index build (index already exists)")
            return True
    
    return run_command(cmd, "Scoring Index", skip_if_exists=SCORING_INDEX_OUTPUT if not force else None)


def build_scored_filtered_dataset(force: bool = False) -> bool:
    """Build filtered dataset from scoring results on daily dataset."""
    print("=" * 60)
    print("Step 7: Building Scored Filtered Dataset")
    print("=" * 60)
    
    # Find the actual daily output file (may have been set dynamically)
    # If DAILY_OUTPUT doesn't exist, try to find it
    daily_path = DAILY_OUTPUT
    if not daily_path.exists():
        cache_dir = Path("data/cache")
        daily_files = list(cache_dir.glob("daily_*.parquet"))
        if daily_files:
            # Prefer "daily_all.parquet" or most recent
            preferred = [f for f in daily_files if f.name == "daily_all.parquet"]
            if preferred:
                daily_path = preferred[0]
            else:
                daily_path = max(daily_files, key=lambda p: p.stat().st_mtime)
    
    if not check_output_exists(daily_path):
        print(f"⚠️  Error: Daily dataset not found: {daily_path}")
        print("   Please run daily processing first.")
        return False
    
    print(f"✓ Daily dataset found: {daily_path}\n")
    
    # Build scored filtered dataset
    cmd = [
        sys.executable,
        "data_processing/build_filtered_from_scoring.py",
        "--daily-path", str(daily_path),
        "--output", str(FILTERED_SCORED_OUTPUT),
        "--min-daily-range", "0.05",  # Filter out flat stocks
        "--min-score", "0.0",  # No minimum score, but other filters apply
        "--n-jobs", "8",  # Use 8 parallel workers (similar to technical features)
    ]
    
    if not force:
        # Check if we need to rebuild based on dates
        if check_output_exists(FILTERED_SCORED_OUTPUT):
            cache_latest = get_latest_date_in_cache(FILTERED_SCORED_OUTPUT)
            daily_latest = get_latest_date_in_cache(daily_path)
            
            if cache_latest and daily_latest:
                print(f"   Scored filtered dataset latest date: {cache_latest}")
                print(f"   Daily dataset latest date: {daily_latest}")
                if daily_latest <= cache_latest:
                    print(f"⏭️  Skipping scored filtered dataset build (dataset is up to date)")
                    return True
                else:
                    print(f"🔄 Dataset needs update (daily dataset has newer dates)")
            else:
                print(f"⏭️  Skipping scored filtered dataset build (dataset already exists)")
                return True
    
    return run_command(cmd, "Scored Filtered Dataset", skip_if_exists=FILTERED_SCORED_OUTPUT if not force else None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and process all data with recommended defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all data (skips existing outputs)
  python process_all_data.py
  
  # Force reprocess everything
  python process_all_data.py --force
  
  # Process only specific steps
  python process_all_data.py --steps daily minute technical
        """
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if outputs exist",
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["daily", "minute", "macd", "join", "technical", "scoring_index", "scored_filtered"],
        help="Process only specific steps (default: all)",
    )
    parser.add_argument(
        "--skip-download-check",
        action="store_true",
        help="Skip checking for input CSV files",
    )
    
    args = parser.parse_args()
    
    # Determine which steps to run
    if args.steps:
        steps_to_run = set(args.steps)
    else:
        steps_to_run = {"daily", "minute", "macd", "join", "technical", "scoring_index", "scored_filtered"}
    
    print("=" * 60)
    print("Stonks Data Processing Pipeline")
    print("=" * 60)
    print(f"Cache dir: {CACHE_DIR}")
    print(f"Steps to run: {', '.join(sorted(steps_to_run))}")
    print(f"Force mode: {args.force}")
    print("=" * 60)
    print()
    
    # Ensure cache directory exists
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    # Step 1: Daily and Weekly
    if "daily" in steps_to_run:
        if not process_daily_weekly(force=args.force):
            success = False
            if not args.force:
                print("⚠️  Daily/weekly processing failed. Continuing with other steps...\n")
    
    # Step 2: Minute Features
    if "minute" in steps_to_run:
        if not process_minute_features(force=args.force):
            success = False
            if not args.force:
                print("⚠️  Minute features processing failed. Continuing with other steps...\n")
    
    # Step 3: MACD Features
    if "macd" in steps_to_run:
        if not process_macd_features(force=args.force):
            success = False
            if not args.force:
                print("⚠️  MACD features processing failed. Continuing with other steps...\n")
    
    # Step 4: Join MACD with Minute Features
    if "join" in steps_to_run:
        if not join_macd_with_minute_features(force=args.force):
            success = False
    
    # Step 5: Technical Features
    if "technical" in steps_to_run:
        if not process_technical_features(force=args.force):
            success = False
            if not args.force:
                print("⚠️  Technical features processing failed. Continuing with other steps...\n")
    
    # Step 6: Scoring Index
    if "scoring_index" in steps_to_run:
        if not build_scoring_index(force=args.force):
            success = False
            if not args.force:
                print("⚠️  Scoring index build failed. Continuing...\n")
    
    # Step 7: Scored Filtered Dataset
    if "scored_filtered" in steps_to_run:
        if not build_scored_filtered_dataset(force=args.force):
            success = False
            if not args.force:
                print("⚠️  Scored filtered dataset build failed. Continuing...\n")
    
    # Summary
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    outputs = {
        "Daily": DAILY_OUTPUT,
        "Weekly": WEEKLY_OUTPUT,
        "Filtered (Legacy)": FILTERED_OUTPUT,
        "Filtered (Scored)": FILTERED_SCORED_OUTPUT,
        "Minute Features": MINUTE_FEATURES_OUTPUT,
        "MACD Features": MACD_INC_DIR,
        "Minute + MACD": MINUTE_MACD_OUTPUT,
        "Technical Features": TECHNICAL_FEATURES_OUTPUT,
        "Scoring Index": SCORING_INDEX_OUTPUT,
    }
    
    for name, path in outputs.items():
        exists = check_output_exists(path)
        status = "✅" if exists else "❌"
        print(f"{status} {name:20} {path}")
    
    print()
    if success:
        print("✅ All requested processing steps completed successfully!")
    else:
        print("⚠️  Some processing steps failed. Check the output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

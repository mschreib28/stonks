#!/usr/bin/env python3
"""
Load and process all data with recommended defaults.

This script:
1. Checks if input CSV files exist (warns if missing)
2. Processes daily/weekly aggregates
3. Processes minute-level features
4. Processes MACD features (incremental)
5. Joins MACD with minute features

Skips processing if output files already exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional


# Default paths
DATA_ROOT = Path("data/2025")
CACHE_DIR = DATA_ROOT / "cache"
DAILY_INPUT = DATA_ROOT / "polygon_day_aggs"
MINUTE_INPUT = DATA_ROOT / "polygon_minute_aggs"

# Output files
DAILY_OUTPUT = CACHE_DIR / "daily_2025.parquet"
WEEKLY_OUTPUT = CACHE_DIR / "weekly.parquet"
FILTERED_OUTPUT = CACHE_DIR / "daily_filtered_1_9.99.parquet"
MINUTE_FEATURES_OUTPUT = CACHE_DIR / "minute_features.parquet"
MACD_INC_DIR = CACHE_DIR / "macd_day_features_inc" / "mode=all"
MINUTE_MACD_OUTPUT = CACHE_DIR / "minute_features_plus_macd.parquet"


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
    """Process daily and weekly aggregates."""
    print("=" * 60)
    print("Step 1: Processing Daily and Weekly Aggregates")
    print("=" * 60)
    
    # Check input
    has_daily, count = check_input_files(DAILY_INPUT, "daily")
    if not has_daily:
        print(f"⚠️  Warning: No daily CSV files found in {DAILY_INPUT}")
        print("   Please download data first or check the path.")
        return False
    print(f"✓ Found {count} daily CSV files\n")
    
    # Process daily/weekly
    cmd = [
        sys.executable,
        "data_processing/build_polygon_cache.py",
        "--input-root", str(DAILY_INPUT),
        "--add-returns",
        "--build-weekly",
    ]
    
    if not force:
        # Check if outputs exist
        if (check_output_exists(DAILY_OUTPUT) and 
            check_output_exists(WEEKLY_OUTPUT) and 
            check_output_exists(FILTERED_OUTPUT)):
            print(f"⏭️  Skipping daily/weekly processing (outputs already exist)")
            return True
    
    return run_command(cmd, "Daily and Weekly Aggregates")


def process_minute_features(force: bool = False) -> bool:
    """Process minute-level features."""
    print("=" * 60)
    print("Step 2: Processing Minute-Level Features")
    print("=" * 60)
    
    # Check inputs
    has_daily, daily_count = check_input_files(DAILY_INPUT, "daily")
    has_minute, minute_count = check_input_files(MINUTE_INPUT, "minute")
    
    if not has_daily:
        print(f"⚠️  Warning: No daily CSV files found in {DAILY_INPUT}")
        return False
    
    if not has_minute:
        print(f"⚠️  Warning: No minute CSV files found in {MINUTE_INPUT}")
        print("   Skipping minute features processing.")
        return False
    
    print(f"✓ Found {daily_count} daily CSV files")
    print(f"✓ Found {minute_count} minute CSV files\n")
    
    # Process minute features
    cmd = [
        sys.executable,
        "data_processing/build_polygon_cache.py",
        "--input-root", str(DAILY_INPUT),
        "--input-minute-root", str(MINUTE_INPUT),
        "--add-returns",
        "--build-minute-features",
    ]
    
    if not force and check_output_exists(MINUTE_FEATURES_OUTPUT):
        print(f"⏭️  Skipping minute features processing (output already exists)")
        return True
    
    return run_command(cmd, "Minute-Level Features", skip_if_exists=MINUTE_FEATURES_OUTPUT if not force else None)


def process_macd_features(force: bool = False) -> bool:
    """Process MACD features incrementally."""
    print("=" * 60)
    print("Step 3: Processing MACD Features (Incremental)")
    print("=" * 60)
    
    # Check input
    has_minute, minute_count = check_input_files(MINUTE_INPUT, "minute")
    if not has_minute:
        print(f"⚠️  Warning: No minute CSV files found in {MINUTE_INPUT}")
        print("   Skipping MACD features processing.")
        return False
    
    print(f"✓ Found {minute_count} minute CSV files\n")
    
    # Process MACD features incrementally
    cmd = [
        sys.executable,
        "data_processing/build_macd_day_features_incremental.py",
        "--input-root", str(MINUTE_INPUT),
        "--out-root", str(CACHE_DIR / "macd_day_features_inc"),
        "--mode", "all",
    ]
    
    # For incremental processing, we check if the directory has files
    # but we still run it since it's incremental (only processes missing days)
    if not force and check_output_exists(MACD_INC_DIR):
        parquet_count = len(list(MACD_INC_DIR.rglob("*.parquet")))
        print(f"✓ Found {parquet_count} existing MACD feature files")
        print(f"   Running incremental update (will skip existing dates)...\n")
    
    return run_command(cmd, "MACD Features (Incremental)")


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
  python process_all_data.py --steps daily minute
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
        choices=["daily", "minute", "macd", "join"],
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
        steps_to_run = {"daily", "minute", "macd", "join"}
    
    print("=" * 60)
    print("Stonks Data Processing Pipeline")
    print("=" * 60)
    print(f"Data root: {DATA_ROOT}")
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
    
    # Summary
    print("=" * 60)
    print("Processing Summary")
    print("=" * 60)
    
    outputs = {
        "Daily": DAILY_OUTPUT,
        "Weekly": WEEKLY_OUTPUT,
        "Filtered": FILTERED_OUTPUT,
        "Minute Features": MINUTE_FEATURES_OUTPUT,
        "MACD Features": MACD_INC_DIR,
        "Minute + MACD": MINUTE_MACD_OUTPUT,
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

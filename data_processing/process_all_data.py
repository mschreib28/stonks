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


# Default paths - cache is shared across all years, raw data is partitioned by year
CACHE_DIR = Path("data/cache")

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
    
    # Process daily/weekly - combine data from all year directories
    # The build_polygon_cache.py script uses input_root/**/*.csv.gz glob
    # We need to point it to a directory that contains all year subdirectories
    # Since the structure is data/YYYY/polygon_day_aggs/MM/file.csv.gz,
    # we can't use a single input-root. Instead, we'll need to process each year
    # and combine, or use a workaround.
    # For now, let's process the primary year (2025) and note that 2026 needs separate handling
    # TODO: Update build_polygon_cache.py to accept multiple input roots or combine years
    
    # Use the first (oldest) year directory as primary, but note we should combine all
    if not daily_inputs:
        raise ValueError("No daily input directories found. Please download data first.")
    primary_input = daily_inputs[0]
    cmd = [
        sys.executable,
        "data_processing/build_polygon_cache.py",
        "--input-root", str(primary_input),
        "--add-returns",
        "--build-weekly",
    ]
    
    # If we have multiple years, warn that we need to combine them
    if len(daily_inputs) > 1:
        print(f"⚠️  Note: Found data in {len(daily_inputs)} year directories.")
        print(f"   Currently processing from: {primary_input}")
        print(f"   To include all years, you may need to manually combine or update the processing script.")
    
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
    
    # Process minute features - use primary year for now
    # TODO: Update to handle multiple years
    primary_daily = daily_inputs[0]
    primary_minute = minute_inputs[0]
    
    cmd = [
        sys.executable,
        "data_processing/build_polygon_cache.py",
        "--input-root", str(primary_daily),
        "--input-minute-root", str(primary_minute),
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

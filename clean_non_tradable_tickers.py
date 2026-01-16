#!/usr/bin/env python3
"""
Clean non-tradable tickers from dataset files.

Removes tickers that are not regular tradable stocks:
- Warrants (tickers ending in W, WS, or containing .W)
- Units (tickers ending in U or containing .U)
- Rights (tickers ending in R or containing .R)
- Preferred shares (tickers with P before last char, or ending in P + letter/number)
- Test symbols (containing TEST)
- Special securities with unusual characters (., -, ^, +, =)
- Very long tickers (7+ characters) which are typically special securities
- Tickers with embedded numbers (except trailing numbers for share classes)
- Short sale volume tickers and other non-stock securities
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from utils.ticker_filter import is_non_tradable_ticker


def clean_dataset(input_path: Path, output_path: Path | None = None, dry_run: bool = False) -> dict:
    """
    Clean a single dataset file by removing non-tradable tickers.
    
    Args:
        input_path: Path to input parquet file
        output_path: Path to output file (defaults to overwriting input)
        dry_run: If True, don't write changes, just report what would be removed
    
    Returns:
        Dictionary with statistics about the cleaning
    """
    if output_path is None:
        output_path = input_path
    
    print(f"\nProcessing: {input_path}")
    
    # Read the dataset
    df = pl.read_parquet(input_path)
    original_rows = len(df)
    
    if "ticker" not in df.columns:
        print(f"  Warning: No 'ticker' column found, skipping")
        return {"file": str(input_path), "skipped": True}
    
    # Get unique tickers before cleaning
    original_tickers = set(df["ticker"].unique().to_list())
    
    # Find non-tradable tickers
    non_tradable = {t for t in original_tickers if is_non_tradable_ticker(t)}
    tradable = original_tickers - non_tradable
    
    print(f"  Original tickers: {len(original_tickers):,}")
    print(f"  Non-tradable: {len(non_tradable):,}")
    print(f"  Tradable: {len(tradable):,}")
    
    if non_tradable:
        # Show sample of removed tickers
        sample = sorted(non_tradable)[:20]
        print(f"  Sample removed: {', '.join(sample)}")
        if len(non_tradable) > 20:
            print(f"  ... and {len(non_tradable) - 20} more")
    
    # Filter the dataframe
    df_clean = df.filter(~pl.col("ticker").is_in(list(non_tradable)))
    cleaned_rows = len(df_clean)
    
    print(f"  Original rows: {original_rows:,}")
    print(f"  Cleaned rows: {cleaned_rows:,}")
    print(f"  Removed rows: {original_rows - cleaned_rows:,}")
    
    if not dry_run and len(non_tradable) > 0:
        # Write the cleaned dataset
        df_clean.write_parquet(output_path)
        print(f"  Written to: {output_path}")
    elif dry_run:
        print(f"  (Dry run - no changes written)")
    
    return {
        "file": str(input_path),
        "original_tickers": len(original_tickers),
        "removed_tickers": len(non_tradable),
        "remaining_tickers": len(tradable),
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "removed_rows": original_rows - cleaned_rows,
        "sample_removed": sorted(non_tradable)[:50],
    }


def main():
    parser = argparse.ArgumentParser(description="Clean non-tradable tickers from datasets")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/2025/cache",
        help="Directory containing parquet cache files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without making changes",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Clean a specific file only",
    )
    args = parser.parse_args()
    
    cache_dir = Path(args.cache_dir)
    
    if args.file:
        # Clean specific file
        file_path = Path(args.file)
        if not file_path.exists():
            file_path = cache_dir / args.file
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
        
        clean_dataset(file_path, dry_run=args.dry_run)
    else:
        # Clean all main dataset files
        dataset_files = [
            "daily_2025.parquet",
            "daily_filtered_1_9.99.parquet",
            "weekly.parquet",
            "minute_features.parquet",
            "minute_features_plus_macd.parquet",
        ]
        
        results = []
        for filename in dataset_files:
            file_path = cache_dir / filename
            if file_path.exists():
                result = clean_dataset(file_path, dry_run=args.dry_run)
                results.append(result)
            else:
                print(f"\nSkipping (not found): {file_path}")
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        
        total_removed_tickers = 0
        total_removed_rows = 0
        
        for r in results:
            if not r.get("skipped"):
                total_removed_tickers += r.get("removed_tickers", 0)
                total_removed_rows += r.get("removed_rows", 0)
        
        print(f"Total unique non-tradable tickers removed: ~{total_removed_tickers}")
        print(f"Total rows removed across all files: {total_removed_rows:,}")
        
        if args.dry_run:
            print("\n(This was a dry run - no changes were made)")
            print("Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()

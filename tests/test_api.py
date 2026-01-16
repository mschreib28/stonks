#!/usr/bin/env python3
"""Quick test to verify API server can start and dependencies are available."""

import sys

def test_imports():
    """Test if all required packages are installed."""
    try:
        import fastapi
        import uvicorn
        import polars
        import pydantic
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nPlease install dependencies with: uv sync")
        return False

def test_cache_dir():
    """Check if cache directory exists."""
    from pathlib import Path
    cache_dir = Path("cache")
    if cache_dir.exists():
        print(f"✓ Cache directory exists: {cache_dir}")
        parquet_files = list(cache_dir.glob("*.parquet"))
        if parquet_files:
            print(f"  Found {len(parquet_files)} Parquet file(s):")
            for f in parquet_files:
                print(f"    - {f.name}")
        else:
            print("  ⚠ No Parquet files found. Run: python data_processing/process_all_data.py")
        return True
    else:
        print(f"⚠ Cache directory does not exist: {cache_dir}")
        print("  Run: python data_processing/process_all_data.py to generate cache files")
        return False

if __name__ == "__main__":
    print("Testing API server setup...\n")
    
    imports_ok = test_imports()
    print()
    cache_ok = test_cache_dir()
    
    print()
    if imports_ok:
        print("✓ API server should be able to start")
        sys.exit(0)
    else:
        print("✗ API server cannot start - missing dependencies")
        sys.exit(1)



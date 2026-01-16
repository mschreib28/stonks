#!/usr/bin/env python3
"""
Download latest data from Polygon S3 bucket and process it.

This script:
1. Checks what dates are already downloaded locally
2. Determines the latest available date (up to today)
3. Downloads missing daily and minute data from S3
4. Organizes files into the correct directory structure
5. Runs process_all_data.py to process the updated data

Requirements:
- AWS CLI configured with credentials, OR
- boto3 installed (pip install boto3)
- Access to s3://flatfiles bucket
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Try to import boto3, fall back to AWS CLI if not available
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

# Default AWS profile for this project
DEFAULT_AWS_PROFILE = "massive"


# S3 bucket and paths
S3_BUCKET = "flatfiles"
S3_DAY_PREFIX = "us_stocks_sip/day_aggs_v1"
S3_MINUTE_PREFIX = "us_stocks_sip/minute_aggs_v1"
# Custom endpoint for Massive S3
S3_ENDPOINT_URL = "https://files.massive.com"

# Local paths - will be determined dynamically based on year
def get_data_root(year: int) -> Path:
    """Get the data root path for a given year."""
    return Path(f"data/{year}")

def get_daily_input(year: int) -> Path:
    """Get the daily input path for a given year."""
    return get_data_root(year) / "polygon_day_aggs"

def get_minute_input(year: int) -> Path:
    """Get the minute input path for a given year."""
    return get_data_root(year) / "polygon_minute_aggs"


def get_local_dates(input_dir: Path) -> set[str]:
    """Get set of dates (YYYY-MM-DD) that are already downloaded locally."""
    if not input_dir.exists():
        return set()
    
    dates = set()
    for csv_file in input_dir.rglob("*.csv.gz"):
        # Extract date from filename: YYYY-MM-DD.csv.gz
        date_str = csv_file.stem.replace(".csv", "")
        if len(date_str) == 10:  # YYYY-MM-DD format
            dates.add(date_str)
    
    return dates

def get_all_local_dates() -> set[str]:
    """Get all dates from all year directories."""
    dates = set()
    data_root = Path("data")
    if not data_root.exists():
        return dates
    
    # Check all year directories
    for year_dir in data_root.iterdir():
        if year_dir.is_dir() and year_dir.name.isdigit():
            daily_dir = year_dir / "polygon_day_aggs"
            dates.update(get_local_dates(daily_dir))
    
    return dates


def get_s3_dates_boto3(bucket: str, prefix: str, year: int, profile: Optional[str] = None) -> set[str]:
    """Get set of available dates from S3 using boto3."""
    # Prefer environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) over profile
    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if aws_key_id and aws_secret:
        print(f"🔑 Using AWS credentials from environment variables")
        print(f"   Access Key ID: {aws_key_id[:8]}...{aws_key_id[-4:] if len(aws_key_id) > 12 else '***'}")
        print(f"   Secret Key: {'*' * 20}...{aws_secret[-4:] if len(aws_secret) > 24 else '***'}")
        session = boto3.Session()  # Will use credentials from environment
    elif profile is not None:
        print(f"🔑 Using AWS profile: {profile}")
        session = boto3.Session(profile_name=profile)
        # Try to get credentials from the session to verify
        try:
            credentials = session.get_credentials()
            if credentials:
                access_key = credentials.access_key
                print(f"   Access Key ID from profile: {access_key[:8]}...{access_key[-4:] if len(access_key) > 12 else '***'}")
        except Exception as e:
            print(f"   ⚠️  Could not retrieve credentials from profile: {e}")
    else:
        print(f"🔑 Using default AWS credentials")
        session = boto3.Session()
        try:
            credentials = session.get_credentials()
            if credentials:
                access_key = credentials.access_key
                print(f"   Access Key ID: {access_key[:8]}...{access_key[-4:] if len(access_key) > 12 else '***'}")
        except Exception as e:
            print(f"   ⚠️  Could not retrieve credentials: {e}")
    
    s3 = session.client('s3', endpoint_url=S3_ENDPOINT_URL)
    dates = set()
    
    try:
        # List objects in the year/month structure
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/{year}/")
        
        for page in pages:
            if 'Contents' not in page:
                continue
            for obj in page['Contents']:
                key = obj['Key']
                # Extract date from key: prefix/YYYY/MM/YYYY-MM-DD.csv.gz
                filename = Path(key).name
                if filename.endswith('.csv.gz'):
                    date_str = filename.replace('.csv.gz', '')
                    if len(date_str) == 10:  # YYYY-MM-DD format
                        dates.add(date_str)
    except NoCredentialsError:
        print("❌ Error: AWS credentials not found. Please configure AWS CLI or set credentials.")
        sys.exit(1)
    except ClientError as e:
        print(f"❌ Error accessing S3: {e}")
        sys.exit(1)
    
    return dates


def get_s3_dates_aws_cli(bucket: str, prefix: str, year: int, profile: Optional[str] = None) -> set[str]:
    """Get set of available dates from S3 using AWS CLI."""
    dates = set()
    
    try:
        # Use AWS CLI to list objects
        cmd = [
            "aws", "s3", "ls",
            f"s3://{bucket}/{prefix}/{year}/",
            "--recursive",
            "--endpoint-url", S3_ENDPOINT_URL
        ]
        if profile:
            cmd.extend(["--profile", profile])
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 4:
                key = parts[-1]  # Last part is the full key
                filename = Path(key).name
                if filename.endswith('.csv.gz'):
                    date_str = filename.replace('.csv.gz', '')
                    if len(date_str) == 10:  # YYYY-MM-DD format
                        dates.add(date_str)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running AWS CLI: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: AWS CLI not found. Please install AWS CLI or install boto3 (pip install boto3)")
        sys.exit(1)
    
    return dates


def download_from_s3_boto3(bucket: str, s3_key: str, local_path: Path, profile: Optional[str] = None) -> bool:
    """Download a file from S3 using boto3."""
    # Prefer environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) over profile
    aws_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret = os.getenv("AWS_SECRET_ACCESS_KEY")
    
    if aws_key_id and aws_secret:
        session = boto3.Session()  # Will use credentials from environment
    elif profile is not None:
        session = boto3.Session(profile_name=profile)
    else:
        session = boto3.Session()
    s3 = session.client('s3', endpoint_url=S3_ENDPOINT_URL)
    
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, s3_key, str(local_path))
        return True
    except ClientError as e:
        print(f"   ⚠️  Failed to download {s3_key}: {e}")
        return False


def download_from_s3_aws_cli(bucket: str, s3_key: str, local_path: Path, profile: Optional[str] = None) -> bool:
    """Download a file from S3 using AWS CLI."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "aws", "s3", "cp",
            f"s3://{bucket}/{s3_key}",
            str(local_path),
            "--endpoint-url", S3_ENDPOINT_URL
        ]
        if profile:
            cmd.extend(["--profile", profile])
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"   ⚠️  Failed to download {s3_key}: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        return False


def download_missing_dates(
    dates_to_download: set[str],
    prefix: str,
    use_boto3: bool,
    profile: Optional[str] = None
) -> int:
    """Download missing dates from S3, saving to the correct year directory."""
    if not dates_to_download:
        return 0
    
    print(f"\n📥 Downloading {len(dates_to_download)} files from {prefix}...")
    
    downloaded = 0
    for date_str in sorted(dates_to_download):
        # Parse date to get year and month
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        year = date_obj.year
        month = f"{date_obj.month:02d}"
        
        # S3 key: prefix/YYYY/MM/YYYY-MM-DD.csv.gz
        s3_key = f"{prefix}/{year}/{month}/{date_str}.csv.gz"
        
        # Local path: data/YYYY/polygon_day_aggs/MM/YYYY-MM-DD.csv.gz or data/YYYY/polygon_minute_aggs/MM/YYYY-MM-DD.csv.gz
        # Determine output directory based on prefix
        if "day_aggs" in prefix:
            output_dir = get_daily_input(year)
        else:
            output_dir = get_minute_input(year)
        
        local_path = output_dir / month / f"{date_str}.csv.gz"
        
        # Skip if already exists
        if local_path.exists():
            print(f"   ⏭️  {date_str} already exists, skipping")
            continue
        
        # Ensure directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"   📥 Downloading {date_str}...", end=" ", flush=True)
        
        if use_boto3:
            success = download_from_s3_boto3(S3_BUCKET, s3_key, local_path, profile)
        else:
            success = download_from_s3_aws_cli(S3_BUCKET, s3_key, local_path, profile)
        
        if success:
            print("✅")
            downloaded += 1
        else:
            print("❌")
    
    return downloaded


def get_dates_to_download(
    local_dates: set[str],
    s3_dates: set[str],
    end_date: Optional[datetime] = None
) -> set[str]:
    """Determine which dates need to be downloaded."""
    if end_date is None:
        end_date = datetime.now()
    
    # Only consider dates up to end_date (exclude future dates)
    valid_s3_dates = {
        d for d in s3_dates
        if datetime.strptime(d, "%Y-%m-%d") <= end_date
    }
    
    # Return dates that are in S3 but not local
    return valid_s3_dates - local_dates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download latest Polygon data from S3 and process it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download latest data and process
  python update_and_process_data.py
  
  # Download only (don't process)
  python update_and_process_data.py --no-process
  
  # Download up to a specific date
  python update_and_process_data.py --end-date 2025-01-31
  
  # Force reprocess after download
  python update_and_process_data.py --force-process
        """
    )
    parser.add_argument(
        "--no-process",
        action="store_true",
        help="Download data but don't run process_all_data.py",
    )
    parser.add_argument(
        "--force-process",
        action="store_true",
        help="Force reprocess all data after download (passes --force to process_all_data.py)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Only download data up to this date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Starting year to download data for (default: 2026). Will also check subsequent years up to end-date.",
    )
    parser.add_argument(
        "--use-aws-cli",
        action="store_true",
        help="Force use AWS CLI instead of boto3",
    )
    parser.add_argument(
        "--skip-daily",
        action="store_true",
        help="Skip downloading daily data",
    )
    parser.add_argument(
        "--skip-minute",
        action="store_true",
        help="Skip downloading minute data",
    )
    parser.add_argument(
        "--aws-profile",
        type=str,
        default=DEFAULT_AWS_PROFILE,
        help=f"AWS profile to use (default: {DEFAULT_AWS_PROFILE})",
    )
    
    args = parser.parse_args()
    
    # Set AWS profile environment variable for subprocess calls
    aws_profile = args.aws_profile or DEFAULT_AWS_PROFILE
    os.environ["AWS_PROFILE"] = aws_profile
    
    # Determine which method to use
    use_boto3 = HAS_BOTO3 and not args.use_aws_cli
    if not use_boto3 and not args.use_aws_cli:
        print("⚠️  Warning: boto3 not available, falling back to AWS CLI")
        print("   Install boto3 with: pip install boto3")
        use_boto3 = False
    
    # Parse end date
    if args.end_date:
        try:
            end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Error: Invalid date format '{args.end_date}'. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        end_date = datetime.now()
    
    # Determine which years to check (from 2025 to end_date.year, or args.year to end_date.year if later)
    start_year = min(2025, args.year)  # Always check from at least 2025
    years_to_check = list(range(start_year, end_date.year + 1))
    
    print("=" * 60)
    print("Polygon Data Update Script")
    print("=" * 60)
    print(f"Years to check: {years_to_check}")
    print(f"End date: {end_date.strftime('%Y-%m-%d')}")
    print(f"Method: {'boto3' if use_boto3 else 'AWS CLI'}")
    print(f"AWS Profile: {aws_profile}")
    print("=" * 60)
    
    # Get local dates from all year directories
    print("\n📂 Checking local files...")
    local_daily = get_all_local_dates()
    local_minute = get_all_local_dates()  # Same function works for both
    print(f"   Daily files: {len(local_daily)}")
    print(f"   Minute files: {len(local_minute)}")
    
    # Get S3 dates for all years
    print("\n☁️  Checking S3 for available dates...")
    s3_daily = set()
    s3_minute = set()
    
    for year in years_to_check:
        print(f"   Checking year {year}...")
        if use_boto3:
            year_daily = get_s3_dates_boto3(S3_BUCKET, S3_DAY_PREFIX, year, aws_profile)
            year_minute = get_s3_dates_boto3(S3_BUCKET, S3_MINUTE_PREFIX, year, aws_profile)
        else:
            year_daily = get_s3_dates_aws_cli(S3_BUCKET, S3_DAY_PREFIX, year, aws_profile)
            year_minute = get_s3_dates_aws_cli(S3_BUCKET, S3_MINUTE_PREFIX, year, aws_profile)
        s3_daily.update(year_daily)
        s3_minute.update(year_minute)
        print(f"      Year {year}: {len(year_daily)} daily, {len(year_minute)} minute files")
    
    print(f"   Total daily files available: {len(s3_daily)}")
    print(f"   Total minute files available: {len(s3_minute)}")
    
    # Determine what to download
    if not args.skip_daily:
        daily_to_download = get_dates_to_download(local_daily, s3_daily, end_date)
        print(f"\n📊 Daily data: {len(daily_to_download)} files to download")
    else:
        daily_to_download = set()
        print("\n⏭️  Skipping daily data download")
    
    if not args.skip_minute:
        minute_to_download = get_dates_to_download(local_minute, s3_minute, end_date)
        print(f"📊 Minute data: {len(minute_to_download)} files to download")
    else:
        minute_to_download = set()
        print("⏭️  Skipping minute data download")
    
    # Download missing files
    total_downloaded = 0
    
    if daily_to_download:
        downloaded = download_missing_dates(
            daily_to_download, S3_DAY_PREFIX, use_boto3, aws_profile
        )
        total_downloaded += downloaded
        print(f"\n✅ Downloaded {downloaded} daily files")
    
    if minute_to_download:
        downloaded = download_missing_dates(
            minute_to_download, S3_MINUTE_PREFIX, use_boto3, aws_profile
        )
        total_downloaded += downloaded
        print(f"\n✅ Downloaded {downloaded} minute files")
    
    if total_downloaded == 0:
        print("\n✅ All data is up to date!")
    
    # Run process_all_data.py
    if not args.no_process:
        print("\n" + "=" * 60)
        print("Processing Data")
        print("=" * 60)
        
        cmd = [sys.executable, "data_processing/process_all_data.py"]
        if args.force_process:
            cmd.append("--force")
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("\n✅ Data processing completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Data processing failed with exit code {e.returncode}")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping data processing (--no-process flag set)")
        print("   Run manually with: python data_processing/process_all_data.py")


if __name__ == "__main__":
    main()

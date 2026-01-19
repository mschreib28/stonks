> **Disclaimer**: This software and documentation are for informational and educational purposes only. Nothing contained herein should be construed as investment advice, a recommendation to buy, sell, or hold any security or financial instrument, or as an offer or solicitation of an offer to buy or sell any security. Trading securities involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. You should consult with a qualified financial advisor or other professional before making any investment decisions. The authors and contributors of this project are not registered investment advisors, financial planners, or securities brokers. Use of this software is at your own risk.

# Stonks - Stock Data Processing and Exploration

A Python-based stock data processing pipeline with a React web interface for exploring and analyzing market data from Polygon.io.

## Features

### Data Processing
- **Daily/Weekly Aggregates**: Process Polygon CSV files into efficient Parquet caches with computed returns and filtering columns
- **Minute-Level Features**: Aggregate minute-by-minute data to ticker-day level features for fast screening
- **MACD Indicators**: Compute MACD(12,26,9) on 1-minute and 5-minute timeframes
- **Pre-filtered Datasets**: Pre-filter stocks by price range (default: $1-9.99) for faster analysis

### Web Interface
- **Dynamic Filtering**: Filter by any column with range, boolean, or text filters
- **Sortable Tables**: Click column headers to sort data
- **Multiple Datasets**: Switch between daily, weekly, filtered, minute features, and MACD-enhanced datasets
- **Interactive Charts**: View price, volume, and returns charts for any ticker
- **Real-time Stats**: See min/max/mean/std for numeric columns

## Setup

### Prerequisites
- Python 3.13+
- Node.js 18+ and npm
- `uv` package manager (for Python dependencies)

### Python Backend Setup

1. **Install dependencies using `uv`:**
```bash
uv sync
```

2. **Verify installation:**
```bash
python --version  # Should be 3.13+
uv --version
```

### React Frontend Setup

1. **Navigate to the frontend directory:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Verify installation:**
```bash
npm --version
```

## Data Download and Structure

### Folder Structure

The project uses the following folder structure:

```
.
├── data/
│   └── 2025/
│       ├── cache/                          # Processed Parquet files
│       │   ├── daily_2025.parquet
│       │   ├── daily_filtered_1_9.99.parquet
│       │   ├── weekly.parquet
│       │   ├── minute_features.parquet
│       │   ├── minute_features_plus_macd.parquet
│       │   └── macd_day_features_inc/      # Incremental MACD features
│       ├── polygon_day_aggs/               # Daily CSV files (input)
│       │   └── MM/                         # Month folders (01-12)
│       │       └── YYYY-MM-DD.csv.gz
│       └── polygon_minute_aggs/            # Minute CSV files (input)
│           └── MM/                         # Month folders (01-12)
│               └── YYYY-MM-DD.csv.gz
```

### Downloading Data

Download Polygon data files and organize them in the structure above:

1. **Daily aggregates**: Place in `data/2025/polygon_day_aggs/MM/YYYY-MM-DD.csv.gz`
2. **Minute aggregates**: Place in `data/2025/polygon_minute_aggs/MM/YYYY-MM-DD.csv.gz`

**Expected CSV columns:**
- `ticker` (or `symbol`)
- `window_start` (or `timestamp` or `t`) - epoch milliseconds for daily, nanoseconds for minute
- `open`, `high`, `low`, `close` (Float64)
- `volume` (Int64)

## Data Processing

### Quick Start: Process All Data

**Recommended approach:** Use the helper script to process all data with recommended defaults:

```bash
python data_processing/process_all_data.py
```

This script:
- ✅ Checks if input CSV files exist
- ✅ Skips processing if output files already exist
- ✅ Processes all steps in order: daily/weekly → minute features → MACD → join
- ✅ Uses recommended defaults
- ✅ Provides clear status messages

**Options:**
```bash
# Force reprocess everything (ignore existing outputs)
python data_processing/process_all_data.py --force

# Process only specific steps
python data_processing/process_all_data.py --steps daily minute

# Available steps: daily, minute, macd, join
```

### Manual Processing (Step by Step)

If you prefer to run steps individually or customize options:

### 1. Daily and Weekly Aggregates

Process daily data and create weekly aggregates:

```bash
python data_processing/build_polygon_cache.py \
  --input-root data/2025/polygon_day_aggs \
  --add-returns \
  --build-weekly
```

**Output files:**
- `data/2025/cache/daily_2025.parquet` - Daily data with returns and computed columns
- `data/2025/cache/daily_filtered_1_9.99.parquet` - Pre-filtered $1-9.99 price range
- `data/2025/cache/weekly.parquet` - Weekly aggregates

**Options:**
- `--add-returns`: Add daily return calculations (required for most features)
- `--build-weekly`: Build weekly aggregates
- `--prefilter-price-range MIN MAX`: Custom price range (default: 1.0 9.99)
- `--partitioned`: Write partitioned Parquet (recommended for large datasets)
- `--compression`: Compression codec (default: zstd)

### 2. Minute-Level Features

Process minute-by-minute data into ticker-day level features:

```bash
python data_processing/build_polygon_cache.py \
  --input-root data/2025/polygon_day_aggs \
  --input-minute-root data/2025/polygon_minute_aggs \
  --add-returns \
  --build-minute-features
```

**Output file:**
- `data/2025/cache/minute_features.parquet`

**Features computed:**
- `day_range_pct`: Intraday volatility `(high - low) / open`
- `max_1m_ret`, `max_5m_ret`: Maximum 1-minute and 5-minute returns
- `max_drawdown_5m`: Maximum drawdown within rolling 5-minute windows
- `vol_spike`: Volume vs 20-day average
- Opening range features (5m and 15m):
  - `first_5m_high`, `first_5m_low`, `first_15m_high`, `first_15m_low`
  - `breakout_5m_up`, `breakout_5m_down`, `breakout_15m_up`, `breakout_15m_down`
  - `range_5m_pct`, `range_15m_pct`

**Note:** Minute features are automatically filtered to the same price range as the pre-filtered daily dataset (default: $1-9.99).

**Options:**
- `--vol-spike-lookback N`: Days for volume spike comparison (default: 20)

### 3. MACD Features

#### Option A: Full MACD Cache (All Data at Once)

Process all minute data to create MACD features:

```bash
python data_processing/build_macd_cache.py \
  --input-root data/2025/polygon_minute_aggs \
  --out-root data/2025/cache/macd_day_features \
  --partitioned
```

**Output file:**
- `data/2025/cache/macd_day_features/macd_day_features.parquet`

#### Option B: Incremental MACD Features (Recommended)

Process day-by-day for incremental updates:

```bash
python data_processing/build_macd_day_features_incremental.py \
  --input-root data/2025/polygon_minute_aggs \
  --out-root data/2025/cache/macd_day_features_inc \
  --mode all
```

**Output directory:**
- `data/2025/cache/macd_day_features_inc/mode=all/date=YYYY-MM-DD.parquet`

**Options:**
- `--mode all`: Use all trading hours (default)
- `--mode rth`: Use regular trading hours only (9:30 AM - 4:00 PM ET)

**MACD Features:**
- Computed on 1-minute and 5-minute timeframes
- Per ticker-day: `macd_last`, `signal_last`, `hist_last`, `hist_min`, `hist_max`
- Zero-crossing counts: `hist_zero_cross_up_count`, `hist_zero_cross_down_count`
- Columns suffixed with `__1m` and `__5m` for each timeframe

### 4. Join MACD with Minute Features

Combine minute features with MACD data:

```bash
python data_processing/join_macd_with_volitility_dataset.py
```

**Output file:**
- `data/2025/cache/minute_features_plus_macd.parquet`

This creates an enriched dataset combining volatility features with MACD indicators.

### Complete Processing Pipeline

**Option 1: Use the helper script (recommended)**
```bash
python data_processing/process_all_data.py
```

**Option 2: Manual step-by-step processing**

```bash
# 1. Daily and weekly aggregates
python data_processing/build_polygon_cache.py \
  --input-root data/2025/polygon_day_aggs \
  --add-returns \
  --build-weekly

# 2. Minute features
python data_processing/build_polygon_cache.py \
  --input-root data/2025/polygon_day_aggs \
  --input-minute-root data/2025/polygon_minute_aggs \
  --add-returns \
  --build-minute-features

# 3. MACD features (incremental)
python data_processing/build_macd_day_features_incremental.py \
  --input-root data/2025/polygon_minute_aggs \
  --out-root data/2025/cache/macd_day_features_inc \
  --mode all

# 4. Join MACD with minute features
python data_processing/join_macd_with_volitility_dataset.py
```

## Running the Application

### Option 1: Convenience Script (Recommended)

Use the provided script to start both servers:

```bash
./start_dev.sh
```

This starts:
- API server on http://localhost:8000
- Frontend dev server on http://localhost:3000

### Option 2: Manual Startup

**Terminal 1 - API Server:**
```bash
python backend/api_server.py
```

Or with auto-reload:
```bash
uvicorn backend.api_server:app --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Using the Web Interface

1. Open http://localhost:3000 in your browser
2. Select a dataset from the dropdown:
   - `daily`: Full daily dataset
   - `weekly`: Weekly aggregates
   - `filtered`: Pre-filtered $1-9.99 price range
   - `minute_features`: Minute-level features
   - `minute_features_plus_macd`: Minute features with MACD indicators
3. Use the filter panel to filter data by any column
4. Click column headers to sort
5. Click on any ticker to view detailed charts

## Project Structure

```
.
├── data_processing/                    # Data processing scripts
│   ├── process_all_data.py             # Helper script: process all data with defaults
│   ├── build_polygon_cache.py          # Main data processing pipeline
│   ├── build_macd_cache.py             # MACD feature generation (full)
│   ├── build_macd_day_features_incremental.py  # MACD features (incremental)
│   ├── join_macd_with_volitility_dataset.py   # Join MACD with minute features
│   ├── build_training_dataset.py       # Build ML training dataset
│   ├── build_event_labels.py           # Build event labels
│   ├── build_range_expansion_labels.py # Build range expansion labels
│   ├── update_and_process_data.py      # Download and process data
│   ├── clean_non_tradable_tickers.py   # Clean non-tradable tickers
│   └── check_missing.py                # Check for missing data files
├── backend/
│   └── api_server.py                   # FastAPI backend server
├── tests/
│   └── test_api.py                     # API testing script
├── start_dev.sh                        # Convenience startup script
├── data/                               # Data directory
│   └── 2025/
│       ├── cache/                      # Processed Parquet files
│       ├── polygon_day_aggs/           # Daily CSV input files
│       └── polygon_minute_aggs/        # Minute CSV input files
└── frontend/                           # React frontend application
    ├── src/
    │   ├── components/                 # React components
    │   │   ├── DataTable.tsx
    │   │   ├── FilterPanel.tsx
    │   │   ├── ChartPanel.tsx
    │   │   └── DatasetSelector.tsx
    │   ├── App.tsx                     # Main app component
    │   ├── api.ts                      # API client
    │   └── types.ts                    # TypeScript types
    └── package.json
```

## API Endpoints

The FastAPI server provides the following endpoints:

- `GET /health` - Health check
- `GET /api/datasets` - List available datasets
- `GET /api/dataset/{dataset}/columns` - Get column information for a dataset
- `POST /api/query` - Query data with filters, sorting, and pagination
- `GET /api/dataset/{dataset}/stats` - Get statistical summary of a dataset
- `GET /api/dataset/{dataset}/ticker/{ticker}` - Get all data for a specific ticker

### Example API Usage

```bash
# List available datasets
curl http://localhost:8000/api/datasets

# Get column info
curl http://localhost:8000/api/dataset/daily/columns

# Query with filters
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "dataset": "filtered",
    "filters": {"close": {"min": 2.0, "max": 5.0}},
    "sort_by": "volume",
    "sort_desc": true,
    "limit": 100
  }'
```

## Data Formats

### Daily Dataset (`daily_2025.parquet`)

**Columns:**
- `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`
- `ret_d`: Daily return percentage `(close / prev_close) - 1` (if `--add-returns`)
- `close_in_band`: Boolean indicating if close is in price range (default: $1-9.99)
- `abs_ret_d`: Absolute daily return percentage `|ret_d|`
- `abs_change_d`: Absolute daily price change in dollars `|close - prev_close|`
- `year`, `month`: Partition keys

### Weekly Dataset (`weekly.parquet`)

**Columns:**
- `week_start`: Start date of trading week (Monday)
- `ticker`, `close_w`, `ret_w`, `year`, `week`
- `abs_ret_w`: Absolute weekly return percentage `|ret_w|`
- `abs_change_w`: Absolute weekly price change in dollars `|close_w - prev_close_w|`

### Minute Features Dataset (`minute_features.parquet`)

**Columns:**
- `ticker`, `date`
- `day_range_pct`: Intraday volatility `(high - low) / open`
- `max_1m_ret`, `max_5m_ret`: Maximum returns over 1-minute and 5-minute periods
- `max_drawdown_5m`: Maximum drawdown within rolling 5-minute windows
- `vol_spike`: Volume ratio vs 20-day average
- Opening range (5m): `first_5m_high`, `first_5m_low`, `range_5m_pct`, `breakout_5m_up`, `breakout_5m_down`
- Opening range (15m): `first_15m_high`, `first_15m_low`, `range_15m_pct`, `breakout_15m_up`, `breakout_15m_down`

### MACD Features Dataset

**Columns (per timeframe, suffixed with `__1m` or `__5m`):**
- `close_last__{tf}`: Last close price for the day
- `macd_last__{tf}`, `signal_last__{tf}`, `hist_last__{tf}`: Final MACD values
- `hist_min__{tf}`, `hist_max__{tf}`: Min/max histogram values for the day
- `hist_zero_cross_up_count__{tf}`, `hist_zero_cross_down_count__{tf}`: Zero-crossing counts

### Minute Features + MACD (`minute_features_plus_macd.parquet`)

Combines all columns from `minute_features` with all MACD columns.

## Troubleshooting

### API Server Issues

**API server won't start:**
```bash
# Test setup
python tests/test_api.py

# Install missing packages
uv sync

# Check if cache files exist
ls data/2025/cache/*.parquet
```

**Frontend can't connect to API (ECONNREFUSED):**
```bash
# Check if API is running
curl http://localhost:8000/health

# Check API server logs
tail -f api_server.log  # if using start_dev.sh

# Verify API server started successfully
# Look for "Application startup complete" in logs
```

### Data Processing Issues

**No data showing:**
```bash
# Ensure cache files are built
python data_processing/process_all_data.py
# Or manually:
python data_processing/build_polygon_cache.py --input-root data/2025/polygon_day_aggs --add-returns
```

# Check which datasets are available
curl http://localhost:8000/api/datasets

# Verify file paths
ls -lh data/2025/cache/*.parquet
```

**Minute features have many null values:**
- This is expected for some features:
  - Opening range features: null if no trading data in first 5/15 minutes
  - Volume spike: null if insufficient prior data (first 20 days)
  - Return/drawdown metrics: null if too few minute bars to calculate
- Breakout flags are set to `False` (not null) when opening range data is missing

**Processing is slow:**
- Use `--partitioned` flag for large datasets
- Minute feature processing filters by price range automatically (reduces data size)
- Consider processing incrementally for MACD features

### Frontend Issues

**Build errors:**
```bash
cd frontend
rm -rf node_modules package-lock.json
npm install
```

**Type errors:**
```bash
cd frontend
npm run build  # Check for TypeScript errors
```

## Performance Tips

1. **Use partitioned Parquet** for large datasets: `--partitioned`
2. **Filter early**: Minute features are pre-filtered by price range
3. **Incremental processing**: Use `data_processing/build_macd_day_features_incremental.py` for daily updates
4. **Compression**: Default `zstd` compression provides good balance of size and speed

## License

This project is dual-licensed:

### Non-Commercial Use

For non-commercial use, this software is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

You are free to:
- Share and redistribute the software
- Adapt and modify the software
- Use the software for personal, educational, or research purposes

Under the following terms:
- **Attribution**: You must give appropriate credit
- **NonCommercial**: You may not use the software for commercial purposes

See the [LICENSE](LICENSE) file for the full license text.

### Commercial Use

For commercial use, a separate proprietary license is required. Commercial use includes any use in for-profit products, services, or organizations that generate revenue.

To obtain a commercial license, please contact the licensor. See [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) for more information.

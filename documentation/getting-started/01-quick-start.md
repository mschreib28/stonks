# Quick Start Guide

Get Stonks running and find your first tradeable stocks in 5 minutes.

## Prerequisites

- Python 3.10+
- Node.js 18+
- Market data (Polygon.io CSV files or S3 access)

## Step 1: Install Dependencies

```bash
# Backend dependencies
uv sync  # or pip install -e .

# Frontend dependencies
cd frontend && npm install
```

## Step 2: Get Market Data

**Option A: If you have Polygon S3 access:**
```bash
# Download latest data from S3 and process it
uv run python data_processing/update_and_process_data.py
```

**Option B: If you have local CSV files:**
Place your files in the expected directory structure:
```
data/
├── 2025/
│   ├── polygon_day_aggs/
│   │   └── 01/
│   │       └── 2025-01-02.csv.gz
│   └── polygon_minute_aggs/
│       └── 01/
│           └── 2025-01-02.csv.gz
└── 2026/
    └── ...
```

## Step 3: Process the Data

Run the master pipeline script to build all caches:

```bash
# Process all data (smart caching - skips existing outputs)
uv run python data_processing/process_all_data.py
```

This will:
- Build daily and weekly Parquet caches
- Compute minute-level features (volatility, opening range breakouts)
- Compute MACD features on 1m and 5m timeframes
- Build technical indicators (RSI, ADX, Bollinger Bands, etc.)

See [Data Pipeline Guide](02-data-pipeline.md) for details on each step.

## Step 4: Start the Application

Open two terminal windows:

**Terminal 1: Backend API**
```bash
uv run python api_server.py
```

**Terminal 2: Frontend**
```bash
cd frontend && npm run dev
```

## Step 5: Find Tradeable Stocks

1. Open http://localhost:5173 in your browser
2. Go to the **Scoring** tab
3. Click **Load Swing Trading Default** preset
4. Click **Score Stocks**
5. Review the ranked results

## What You'll See

The scoring panel ranks stocks based on:

| Metric | What It Measures |
|--------|-----------------|
| **Tradability Score** | Overall suitability for swing trading (0-100) |
| **Avg Daily Range ($)** | How much the stock moves in dollars |
| **Liquidity Multiple** | How easily you can trade 10k shares |
| **Sweet Spot Days %** | Days with ideal $0.20-$0.60 range |
| **Range Consistency** | How predictable the daily range is |
| **Price** | Filtered to $1-$8 for small-cap focus |

## Understanding the Results

### Top 10 Stocks Table

The scored stocks are ranked by a weighted combination of all criteria. Higher scores = better trading candidates.

### What Makes a Good Score?

- **Tradability Score > 70**: Excellent candidate
- **Tradability Score 50-70**: Good candidate, verify other metrics
- **Tradability Score < 50**: May have issues with liquidity or range

### Key Filters Applied

The default preset filters for:
- Price between $0 and $8
- Minimum average volume (ensures liquidity)
- At least 20 trading days of data

## Next Steps

Now that you have the system running:

### Explore the Data
1. **Charts**: Click any ticker to see price history and technical indicators
2. **Factor Evaluation**: Go to the Factor Evaluation tab to see which indicators predict returns

### Train ML Models
```bash
# Train a LightGBM classifier for direction prediction
uv run python data_processing/train_ml_model.py --model-type classifier

# Or train a linear model with Ridge regularization
uv run python data_processing/train_ml_model.py --model-type linear --regularization ridge
```

### Evaluate Factors
```bash
# Evaluate all technical indicators
uv run python data_processing/evaluate_factors.py --factor-column all
```

### Backtest Strategies
```bash
# Test MACD crossover strategy
uv run python data_processing/backtest_vectorbt.py --strategy macd --ticker AAPL
```

### Learn More
- [Data Pipeline Guide](02-data-pipeline.md) - Complete script reference and execution order
- [Stock Selection Criteria](../strategy-guide/02-stock-selection-criteria.md) - Trading criteria explained
- [Feature Engineering](../tier1-features/01-feature-engineering.md) - Technical indicator details
- [ML Models](../tier1-features/02-linear-models.md) - Model training approaches

## Troubleshooting

### "No data found"
- Check that data files exist in `data/YYYY/polygon_day_aggs/`
- Run `uv run python data_processing/process_all_data.py` to build caches

### "API server not responding"
- Check that `api_server.py` is running on port 8000
- Look for error messages in the terminal

### "Frontend won't load"
- Ensure npm dependencies are installed
- Check that the dev server is running on port 5173

### "Missing parquet files"
- Run the pipeline: `uv run python data_processing/process_all_data.py`
- Check output files exist in `data/cache/`

---

*Continue to: [Data Pipeline Guide](02-data-pipeline.md)*

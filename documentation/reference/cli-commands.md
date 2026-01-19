# CLI Commands Reference

Complete reference for all command-line tools in the Stonks project.

## Data Processing

### build_polygon_cache.py

Download and cache market data from Polygon.io.

```bash
uv run python data_processing/build_polygon_cache.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input-root` | data/2025/polygon_day_aggs | Folder containing month subfolders with *.csv.gz files |
| `--out-daily` | data/cache/daily_2025.parquet | Output Parquet path for daily data |
| `--add-returns` | False | Add per-ticker daily returns column ret_d |
| `--partitioned` | False | Write partitioned Parquet dataset |
| `--build-weekly` | False | Also build weekly close-to-close returns |
| `--build-minute-features` | False | Build minute-level feature cache |

**Examples**:
```bash
# Process daily data with returns
uv run python data_processing/build_polygon_cache.py --add-returns

# Process with weekly aggregates
uv run python data_processing/build_polygon_cache.py --add-returns --build-weekly

# Process minute features
uv run python data_processing/build_polygon_cache.py --build-minute-features

# For incremental updates, use process_all_data.py instead:
uv run python data_processing/process_all_data.py
```

---

### build_technical_features.py

Generate technical indicators from cached price data.

```bash
uv run python data_processing/build_technical_features.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input-path` | data/cache/daily_*.parquet | Input price data |
| `--output-path` | data/cache/technical_features.parquet | Output features |
| `--features` | all | Comma-separated feature list |

**Examples**:
```bash
# Build all features
uv run python data_processing/build_technical_features.py

# Build specific features only
uv run python data_processing/build_technical_features.py --features rsi_14,atr_14,bb_pct
```

**Generated Features**:
- `rsi_14` - Relative Strength Index
- `stoch_k`, `stoch_d` - Stochastic Oscillator
- `roc_10` - Rate of Change
- `adx_14` - Average Directional Index
- `bb_pct` - Bollinger Band %B
- `sma_cross_20_50` - SMA Crossover
- `atr_14` - Average True Range
- `hist_vol_20` - Historical Volatility
- Binary signals: `rsi_overbought`, `rsi_oversold`, `above_bb_upper`, `below_bb_lower`, `strong_trend`

---

## Machine Learning

### train_ml_model.py

Train LightGBM models for return prediction.

```bash
uv run python data_processing/train_ml_model.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-path` | data/cache/technical_features.parquet | Feature data |
| `--model-type` | classifier | `classifier` or `regressor` |
| `--n-splits` | 5 | Number of CV folds |
| `--test-size` | 60 | Test period days per fold |
| `--output-dir` | data/cache/models | Model output directory |
| `--num-leaves` | 31 | LightGBM num_leaves |
| `--learning-rate` | 0.05 | LightGBM learning_rate |
| `--n-estimators` | 100 | Max number of trees |

**Examples**:
```bash
# Train classifier for direction prediction
uv run python data_processing/train_ml_model.py --model-type classifier

# Train regressor with custom parameters
uv run python data_processing/train_ml_model.py \
    --model-type regressor \
    --num-leaves 15 \
    --learning-rate 0.03 \
    --n-estimators 200

# Train with more CV folds
uv run python data_processing/train_ml_model.py \
    --model-type classifier \
    --n-splits 10 \
    --test-size 30
```

---

## Factor Evaluation

### evaluate_factors.py

Compute Information Coefficient and quantile returns for factors.

```bash
uv run python data_processing/evaluate_factors.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--factor-path` | data/cache/technical_features.parquet | Factor data |
| `--factor-column` | rsi_14 | Factor to evaluate (or `all`) |
| `--price-path` | data/cache/daily_*.parquet | Price data |
| `--periods` | 1,5,10 | Forward return periods |
| `--quantiles` | 5 | Number of quantile groups |
| `--output` | None | Output directory |
| `--start-date` | None | Start date filter (YYYY-MM-DD) |
| `--end-date` | None | End date filter |

**Examples**:
```bash
# Evaluate single factor
uv run python data_processing/evaluate_factors.py \
    --factor-column rsi_14 \
    --periods 1,5,10

# Evaluate all factors
uv run python data_processing/evaluate_factors.py \
    --factor-column all \
    --output data/cache/factor_evaluation/

# Evaluate on specific date range
uv run python data_processing/evaluate_factors.py \
    --factor-column momentum \
    --start-date 2025-01-01 \
    --end-date 2025-06-30
```

**Output**:
```
Factor: rsi_14
================================================================================
Period  Mean IC    Std IC    t-stat    Positive%
1D      -0.032     0.145     -4.21     42%
5D      -0.028     0.120     -3.50     44%
10D     -0.022     0.095     -2.85     46%

Quantile Returns (1D)
Q1: +0.35%  Q2: +0.15%  Q3: +0.05%  Q4: -0.10%  Q5: -0.25%
Spread: -0.60%
```

---

## Backtesting

### backtest_vectorbt.py

Run strategy backtests using VectorBT.

```bash
uv run python data_processing/backtest_vectorbt.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-path` | data/cache/daily_*.parquet | Price data |
| `--strategy` | macd | `macd`, `rsi`, or `custom` |
| `--ticker` | ALL | Ticker to backtest |
| `--initial-cash` | 100000 | Starting capital |
| `--fees` | 0.001 | Transaction fee (0.1%) |
| `--slippage` | 0.001 | Slippage (0.1%) |
| `--output` | None | Output JSON path |

**MACD Strategy Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--fast-period` | 12 | Fast EMA period |
| `--slow-period` | 26 | Slow EMA period |
| `--signal-period` | 9 | Signal line period |

**RSI Strategy Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--rsi-period` | 14 | RSI calculation period |
| `--oversold` | 30 | Oversold threshold |
| `--overbought` | 70 | Overbought threshold |

**Examples**:
```bash
# MACD backtest on single ticker
uv run python data_processing/backtest_vectorbt.py \
    --strategy macd \
    --ticker AAPL \
    --output results/aapl_macd.json

# RSI backtest with custom parameters
uv run python data_processing/backtest_vectorbt.py \
    --strategy rsi \
    --ticker MSFT \
    --rsi-period 14 \
    --oversold 25 \
    --overbought 75

# MACD with custom periods and higher slippage
uv run python data_processing/backtest_vectorbt.py \
    --strategy macd \
    --fast-period 8 \
    --slow-period 21 \
    --signal-period 5 \
    --slippage 0.005  # 0.5% for small-caps

# Backtest on all tickers
uv run python data_processing/backtest_vectorbt.py \
    --strategy rsi \
    --ticker ALL \
    --output results/portfolio_rsi.json
```

**Output**:
```
Strategy: MACD Crossover
Ticker: AAPL
Period: 2024-01-01 to 2025-01-01
================================================================================
Performance:
  Total Return:    25.3%
  CAGR:            25.3%
  Sharpe Ratio:    1.24
  Sortino Ratio:   1.85
  Max Drawdown:    -12.5%
  
Trades:
  Total:           42
  Win Rate:        54.8%
  Profit Factor:   1.65
  Avg Trade:       0.6%
```

---

## Performance Analysis

### performance_analysis.py

Compute comprehensive performance metrics.

```bash
uv run python data_processing/performance_analysis.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--data-path` | data/cache/daily_*.parquet | Price data |
| `--ticker` | None | Single ticker to analyze |
| `--benchmark` | None | Benchmark ticker (e.g., SPY) |
| `--output` | None | Output JSON path |
| `--start-date` | None | Start date |
| `--end-date` | None | End date |

**Examples**:
```bash
# Analyze single stock
uv run python data_processing/performance_analysis.py \
    --ticker AAPL \
    --benchmark SPY \
    --output results/aapl_performance.json

# Analyze all stocks (portfolio)
uv run python data_processing/performance_analysis.py \
    --output results/portfolio_performance.json

# Analyze specific date range
uv run python data_processing/performance_analysis.py \
    --ticker TSLA \
    --start-date 2024-01-01 \
    --end-date 2024-12-31
```

**Output**:
```
Performance Analysis: AAPL
================================================================================
Summary:
  Total Return:         35.2%
  Annualized Return:    35.2%
  Annualized Volatility: 28.5%
  Trading Days:         252

Risk-Adjusted:
  Sharpe Ratio:    1.23
  Sortino Ratio:   1.85
  Calmar Ratio:    2.45

Drawdown:
  Max Drawdown:    -14.3%
  Max Duration:    45 days

Risk:
  VaR (95%):       -2.8%
  CVaR (95%):      -4.2%

Trade Metrics:
  Win Rate:        54.5%
  Profit Factor:   1.45
  Best Day:        +5.2%
  Worst Day:       -4.8%

Benchmark Comparison (SPY):
  Beta:            1.15
  Alpha:           8.5%
  Information Ratio: 0.65
  Correlation:     0.78
```

---

## Server

### api_server.py

Start the FastAPI backend server.

```bash
uv run python backend/api_server.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | 0.0.0.0 | Host to bind |
| `--port` | 8000 | Port number |
| `--reload` | False | Auto-reload on changes |

**Examples**:
```bash
# Start production server
uv run python backend/api_server.py

# Start development server with reload
uv run python backend/api_server.py --reload

# Start on different port
uv run python backend/api_server.py --port 8080
```

---

## Common Workflows

### Morning Data Update

```bash
# Update data and rebuild features (incremental - checks dates automatically)
uv run python data_processing/process_all_data.py

# Or manually update specific components:
uv run python data_processing/build_polygon_cache.py --add-returns && \
uv run python data_processing/build_technical_features.py
```

### Weekly Factor Review

```bash
# Evaluate all factors and save results
uv run python data_processing/evaluate_factors.py \
    --factor-column all \
    --output data/cache/factor_evaluation/weekly_$(date +%Y%m%d)/
```

### Full Backtest Suite

```bash
# Test both strategies on all tickers
uv run python data_processing/backtest_vectorbt.py \
    --strategy macd \
    --ticker ALL \
    --output results/macd_all.json

uv run python data_processing/backtest_vectorbt.py \
    --strategy rsi \
    --ticker ALL \
    --output results/rsi_all.json
```

### Retrain Models

```bash
# Retrain classifier with latest data
uv run python data_processing/train_ml_model.py \
    --model-type classifier \
    --output-dir data/cache/models/$(date +%Y%m%d)/
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `POLYGON_API_KEY` | Yes | Polygon.io API key |

**Setting Environment Variables**:

```bash
# Temporary (current session)
export POLYGON_API_KEY=your_key_here

# Permanent (add to .bashrc or .zshrc)
echo 'export POLYGON_API_KEY=your_key_here' >> ~/.zshrc

# Or use .env file in project root
echo 'POLYGON_API_KEY=your_key_here' > .env
```

---

## Troubleshooting

### Common Issues

**"Module not found"**:
```bash
# Ensure you're in the project directory and environment is activated
cd /path/to/Stonks
source .venv/bin/activate  # or uv sync
```

**"No data found"**:
```bash
# Check if data files exist
ls -la data/cache/

# Rebuild cache if needed (use --force flag with process_all_data.py)
uv run python data_processing/process_all_data.py --force
```

**"API rate limit"**:
```bash
# Polygon has rate limits; wait and retry
sleep 60 && uv run python data_processing/process_all_data.py
```

**"Out of memory"**:
```bash
# Process data in smaller batches by limiting input directories
# Or use partitioned output to reduce memory usage
uv run python data_processing/build_polygon_cache.py --partitioned
```

---

## Getting Help

```bash
# Most scripts support --help
uv run python data_processing/build_polygon_cache.py --help
uv run python data_processing/train_ml_model.py --help
uv run python data_processing/backtest_vectorbt.py --help
```

---

*Return to: [Documentation Home](../README.md)*

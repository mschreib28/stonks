# Data Pipeline Guide

This guide documents all scripts in the `data_processing/` folder, organized by their purpose and the order in which they should be executed to build a trained dataset from scratch.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     STAGE 1: DATA ACQUISITION                        │
├─────────────────────────────────────────────────────────────────────┤
│  update_and_process_data.py (downloads from S3, triggers pipeline)   │
│                              ↓                                       │
│  build_polygon_cache.py (converts CSV → Parquet, filters tickers)    │
│       ↓ daily_2025.parquet    ↓ weekly.parquet    ↓ minute_features │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 2: FEATURE ENGINEERING                      │
├─────────────────────────────────────────────────────────────────────┤
│  build_technical_features.py → RSI, ADX, Bollinger, SMA crossovers   │
│  build_macd_day_features_incremental.py → MACD day summaries         │
│  join_macd_with_volitility_dataset.py → combined minute+MACD         │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: LABELING (Optional)                      │
├─────────────────────────────────────────────────────────────────────┤
│  build_event_labels.py → MACD cross event labels                     │
│  build_range_expansion_labels.py → next-day range expansion labels   │
│  build_training_dataset.py → combines labels with features           │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: MODEL TRAINING                           │
├─────────────────────────────────────────────────────────────────────┤
│  train_ml_model.py → LightGBM, Linear, Logistic models               │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    STAGE 5: ANALYSIS & BACKTESTING                   │
├─────────────────────────────────────────────────────────────────────┤
│  evaluate_factors.py → Information Coefficient analysis              │
│  factor_models.py → CAPM / Fama-French analysis                      │
│  backtest_vectorbt.py → strategy backtesting                         │
│  performance_analysis.py → Sharpe, drawdowns, risk metrics           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start (Recommended Path)

For most users, run the master orchestration script:

```bash
# Process all data with smart caching (skips existing outputs)
python data_processing/process_all_data.py

# Force full reprocessing
python data_processing/process_all_data.py --force

# Process specific steps only
python data_processing/process_all_data.py --steps daily technical
```

This runs Stage 1 and Stage 2 automatically. Continue to Stage 4 for model training.

---

## Stage 1: Data Acquisition

### `update_and_process_data.py` - Download Latest Data

Downloads daily and minute data from Polygon's S3 bucket and triggers processing.

```bash
# Download latest data and process
python data_processing/update_and_process_data.py

# Download only (don't process)
python data_processing/update_and_process_data.py --no-process

# Download specific year
python data_processing/update_and_process_data.py --year 2025

# Skip minute data (faster)
python data_processing/update_and_process_data.py --skip-minute
```

**Prerequisites:**
- AWS credentials configured (profile `massive` by default)
- Access to `s3://flatfiles` bucket

**Outputs:** Raw CSV files in `data/YYYY/polygon_day_aggs/` and `data/YYYY/polygon_minute_aggs/`

---

### `build_polygon_cache.py` - Build Parquet Caches

Converts raw CSV files to efficient Parquet format with filtering and feature computation.

```bash
# Build daily and weekly caches with returns
python data_processing/build_polygon_cache.py \
    --input-root data \
    --add-returns \
    --build-weekly

# Build minute features (requires minute data)
python data_processing/build_polygon_cache.py \
    --input-root data \
    --input-minute-root data \
    --build-minute-features
```

**Key Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--prefilter-price-range` | `1.0 9.99` | Price range filter for swing trading |
| `--prefilter-min-avg-volume` | `750000` | Minimum average daily volume |
| `--prefilter-min-days` | `60` | Minimum trading days required |
| `--vol-spike-lookback` | `20` | Days for volume spike calculation |

**Outputs:**
| File | Description |
|------|-------------|
| `data/cache/daily_2025.parquet` | Full daily OHLCV with returns |
| `data/cache/weekly.parquet` | Weekly aggregates |
| `data/cache/daily_filtered_1_9.99.parquet` | Price-filtered daily data |
| `data/cache/minute_features.parquet` | Minute-level aggregated features |

---

## Stage 2: Feature Engineering

### `build_technical_features.py` - Technical Indicators

Computes technical indicators from daily OHLCV data.

```bash
python data_processing/build_technical_features.py \
    --daily-path data/cache/daily_2025.parquet \
    --output data/cache/technical_features.parquet
```

**Features Generated:**

| Category | Indicators |
|----------|------------|
| **Momentum** | RSI(14), Stochastic %K/%D, ROC(10) |
| **Trend** | ADX(14), SMA crossovers (20/50, 50/200), SMAs |
| **Volatility** | ATR(14), Historical Volatility(20), Bollinger Bands |
| **Volume** | OBV, VWAP approximation |
| **Binary Signals** | RSI overbought/oversold, BB breakouts, trend strength |

**Output:** `data/cache/technical_features.parquet`

---

### `build_macd_day_features_incremental.py` - MACD Features (Recommended)

Computes MACD features incrementally per day. More memory-efficient than the batch version.

```bash
python data_processing/build_macd_day_features_incremental.py \
    --input-root data/2025/polygon_minute_aggs \
    --out-root data/cache/macd_day_features_inc \
    --mode all  # or 'rth' for regular trading hours only
```

**Features Generated:**
- `hist_last__1m`, `hist_last__5m` - Last histogram value
- `hist_min__1m`, `hist_max__1m` - Histogram extremes
- `hist_zero_cross_up_count__1m` - Bullish cross count
- `hist_zero_cross_down_count__1m` - Bearish cross count

**Output:** `data/cache/macd_day_features_inc/mode=all/*.parquet`

---

### `build_macd_cache.py` - MACD Features (Batch)

Alternative batch version of MACD feature computation. Loads all data at once.

```bash
python data_processing/build_macd_cache.py \
    --input-root data/2025/polygon_minute_aggs \
    --out-root data/cache/macd_day_features \
    --write-bar-level  # Optional: also write bar-level data (large)
```

**Use `build_macd_day_features_incremental.py` instead** unless you need bar-level MACD data.

---

### `join_macd_with_volitility_dataset.py` - Combine Features

Joins minute features with MACD day features.

```bash
python data_processing/join_macd_with_volitility_dataset.py
```

**Inputs:**
- `data/cache/minute_features.parquet`
- `data/cache/macd_day_features_inc/mode=all/*.parquet`

**Output:** `data/cache/minute_features_plus_macd.parquet`

---

## Stage 3: Labeling (For ML Training)

### `build_event_labels.py` - MACD Cross Event Labels

Creates labeled events from MACD histogram crosses for supervised learning.

```bash
python data_processing/build_event_labels.py \
    --input-root data/2025/polygon_minute_aggs \
    --out-root data/cache/training \
    --target-move 0.008 \
    --time-window 30 \
    --mode all
```

**Key Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--target-move` | `0.008` | Target move (0.8% = 80 basis points) |
| `--time-window` | `30` | Forward-looking bars to check |
| `--long-only` | - | Only cross-up (bullish) events |
| `--short-only` | - | Only cross-down (bearish) events |

**Output:** `data/cache/training/events_YYYY-MM-DD.parquet`

---

### `build_range_expansion_labels.py` - Range Expansion Labels

Creates labels based on next-day range expansion.

```bash
python data_processing/build_range_expansion_labels.py \
    --input data/cache/daily_2025.parquet \
    --output data/cache/range_expansion_labels.parquet \
    --threshold 0.08  # 8% range expansion
```

**Output:** `data/cache/range_expansion_labels.parquet`

---

### `build_training_dataset.py` - Combine Features & Labels

Combines event labels with MACD and volatility features for model training.

```bash
python data_processing/build_training_dataset.py \
    --events-glob "data/cache/training/events_*.parquet" \
    --macd-glob "data/cache/macd_day_features_inc/mode=all/*.parquet" \
    --vol-path data/cache/minute_features.parquet \
    --daily-path data/cache/daily_filtered_1_9.99.parquet \
    --output data/cache/training/ml_dataset.parquet
```

**Output:** `data/cache/training/ml_dataset.parquet`

---

## Stage 4: Model Training

### `train_ml_model.py` - Train Predictive Models

Trains ML models using time-series cross-validation.

```bash
# LightGBM classifier (direction prediction)
python data_processing/train_ml_model.py \
    --data-path data/cache/technical_features.parquet \
    --model-type classifier \
    --output-dir data/cache/models

# LightGBM regressor (return magnitude)
python data_processing/train_ml_model.py \
    --model-type regressor

# Linear regression with Ridge regularization
python data_processing/train_ml_model.py \
    --model-type linear \
    --regularization ridge \
    --tune-alpha

# Lasso for feature selection
python data_processing/train_ml_model.py \
    --model-type linear \
    --regularization lasso \
    --alpha 0.01

# Logistic regression for direction
python data_processing/train_ml_model.py \
    --model-type logistic \
    --regularization l2
```

**Model Types:**
| Type | Purpose | Output |
|------|---------|--------|
| `classifier` | LightGBM for up/down prediction | Accuracy, AUC, F1 |
| `regressor` | LightGBM for return magnitude | RMSE, R², Direction accuracy |
| `linear` | OLS/Ridge/Lasso/ElasticNet regression | Coefficients, t-stats, p-values |
| `logistic` | Logistic regression for direction | Accuracy, AUC, coefficients |

**Outputs:**
- `data/cache/models/*.joblib` - Trained model files
- `data/cache/models/*_results.json` - CV metrics and feature importance

---

## Stage 5: Analysis & Backtesting

### `evaluate_factors.py` - Factor Quality Analysis

Evaluates predictive power of features using Information Coefficient.

```bash
# Evaluate single factor
python data_processing/evaluate_factors.py \
    --factor-path data/cache/technical_features.parquet \
    --factor-column rsi_14 \
    --price-path data/cache/daily_2025.parquet

# Evaluate all factors
python data_processing/evaluate_factors.py \
    --factor-column all \
    --output data/cache/factor_evaluation
```

**Metrics Computed:**
- Mean IC (Information Coefficient)
- IC t-statistic and stability
- Quantile returns (Q1 vs Q5 spread)

---

### `factor_models.py` - CAPM & Fama-French Analysis

Computes beta, alpha, and factor exposures.

```bash
# CAPM analysis
python data_processing/factor_models.py \
    --ticker AAPL \
    --analysis capm

# Fama-French 5-factor model
python data_processing/factor_models.py \
    --ticker AAPL \
    --analysis ff \
    --model 5-factor

# Compare all models
python data_processing/factor_models.py \
    --ticker AAPL \
    --analysis compare
```

---

### `backtest_vectorbt.py` - Strategy Backtesting

Backtests trading strategies using VectorBT.

```bash
# MACD crossover strategy
python data_processing/backtest_vectorbt.py \
    --data-path data/cache/daily_2025.parquet \
    --strategy macd \
    --ticker AAPL

# RSI mean reversion
python data_processing/backtest_vectorbt.py \
    --strategy rsi \
    --rsi-period 14 \
    --oversold 30 \
    --overbought 70
```

**Strategies Available:**
| Strategy | Entry Signal | Exit Signal |
|----------|--------------|-------------|
| MACD | Histogram crosses above zero | Histogram crosses below zero |
| RSI | RSI crosses below oversold | RSI crosses above overbought |

---

### `performance_analysis.py` - Performance Metrics

Comprehensive strategy/portfolio performance analysis.

```bash
# Single ticker analysis
python data_processing/performance_analysis.py \
    --data-path data/cache/daily_2025.parquet \
    --ticker AAPL \
    --benchmark SPY

# Portfolio analysis (equal weight all tickers)
python data_processing/performance_analysis.py \
    --data-path data/cache/daily_2025.parquet
```

**Metrics Computed:**
- Returns: Total, annualized, volatility
- Risk-adjusted: Sharpe, Sortino, Calmar ratios
- Risk: VaR, CVaR, max drawdown
- Benchmark: Alpha, beta, information ratio

---

## Utility Scripts

### `process_all_data.py` - Master Orchestration

Runs the complete pipeline with smart caching.

```bash
python data_processing/process_all_data.py
python data_processing/process_all_data.py --force  # Rebuild everything
python data_processing/process_all_data.py --steps daily minute macd
```

---

### `check_missing.py` - Verify Data Completeness

Compares local files against expected S3 listing.

```bash
python data_processing/check_missing.py
```

**Requires:** `data/files_list.txt` with S3 listing

---

### `clean_non_tradable_tickers.py` - Remove Non-Stock Tickers

Removes warrants, units, rights, preferred shares, etc.

```bash
# Preview what would be removed
python data_processing/clean_non_tradable_tickers.py --dry-run

# Actually clean
python data_processing/clean_non_tradable_tickers.py

# Clean specific file
python data_processing/clean_non_tradable_tickers.py --file daily_2025.parquet
```

**Removes tickers matching:**
- Warrants (ending in W, WS)
- Units (ending in U)
- Rights (ending in R)
- Preferred shares (P patterns)
- Test symbols, special characters

---

## Complete Pipeline Example

Here's the recommended sequence for building a trained dataset from scratch:

```bash
# 1. Download latest data (if not already present)
python data_processing/update_and_process_data.py --no-process

# 2. Run the core pipeline (daily, minute, MACD, technical features)
python data_processing/process_all_data.py

# 3. (Optional) Build event labels for ML training
python data_processing/build_event_labels.py

# 4. (Optional) Build training dataset
python data_processing/build_training_dataset.py

# 5. Train a model
python data_processing/train_ml_model.py --model-type classifier

# 6. Evaluate factor quality
python data_processing/evaluate_factors.py --factor-column all

# 7. Backtest a strategy
python data_processing/backtest_vectorbt.py --strategy macd
```

---

## Output Files Summary

| File | Size Est. | Description |
|------|-----------|-------------|
| `daily_2025.parquet` | ~200 MB | Full daily OHLCV with returns |
| `weekly.parquet` | ~50 MB | Weekly aggregates |
| `daily_filtered_1_9.99.parquet` | ~50 MB | Price-filtered daily data |
| `minute_features.parquet` | ~100 MB | Minute-level aggregated features |
| `minute_features_plus_macd.parquet` | ~150 MB | Combined minute + MACD |
| `technical_features.parquet` | ~200 MB | All technical indicators |
| `macd_day_features_inc/` | ~100 MB | MACD day summaries |
| `training/ml_dataset.parquet` | ~50 MB | ML-ready training data |
| `models/*.joblib` | ~10 MB | Trained models |

---

*Previous: [Quick Start](01-quick-start.md) | Next: [Using the Frontend](03-frontend-guide.md)*

# Stonks Documentation

Welcome to the Stonks project documentation. This guide will help you understand and effectively use the platform for **short-term trading of small to mid-cap stocks**.

## Who This Is For

This project is designed for traders who want to:
- Find stocks with optimal characteristics for **swing trading** (holding 1-10 days)
- Screen for stocks with good **daily price ranges** and **liquidity**
- Apply **machine learning** to identify predictive patterns
- **Backtest** strategies before risking real capital
- Evaluate **factor effectiveness** using quantitative methods

## Documentation Structure

### 1. Getting Started
Start here if you're new to the project.

| Document | Description |
|----------|-------------|
| [Quick Start](getting-started/01-quick-start.md) | Get the project running in 5 minutes |
| [Understanding the Data Pipeline](getting-started/02-data-pipeline.md) | How data flows through the system |
| [Using the Frontend](getting-started/03-frontend-guide.md) | Navigate the web interface |

### 2. Strategy Guide
Understand the trading approach this project is built around.

| Document | Description |
|----------|-------------|
| [Swing Trading Philosophy](strategy-guide/01-swing-trading-philosophy.md) | The core strategy and why it works |
| [Stock Selection Criteria](strategy-guide/02-stock-selection-criteria.md) | How to find ideal stocks for short-term trading |
| [Risk Management](strategy-guide/03-risk-management.md) | Position sizing and protecting capital |

### 3. Tier 1 Features (Core Tools)
These are the production-ready features you should master first.

| Document | Description | Status |
|----------|-------------|--------|
| [Feature Engineering](tier1-features/01-feature-engineering.md) | Technical indicators that drive predictions | ✅ Complete |
| [Linear Models & Cross-Validation](tier1-features/02-linear-models.md) | Interpretable baselines and proper testing | 🟡 Partial |
| [LightGBM Predictions](tier1-features/03-lightgbm-models.md) | Machine learning for return prediction | ✅ Complete |
| [VectorBT Backtesting](tier1-features/04-vectorbt-backtesting.md) | Test strategies on historical data | ✅ Complete |
| [Alphalens Factor Evaluation](tier1-features/05-alphalens-factors.md) | Evaluate if your factors actually predict returns | ✅ Complete |
| [Performance Analysis](tier1-features/06-performance-analysis.md) | Comprehensive strategy metrics | ✅ Complete |

### 4. Workflows
Step-by-step guides for common tasks.

| Document | Description |
|----------|-------------|
| [Daily Screening Workflow](workflows/01-daily-screening.md) | Find tradeable stocks each morning |
| [Factor Research Workflow](workflows/02-factor-research.md) | Discover new predictive factors |
| [Strategy Development Workflow](workflows/03-strategy-development.md) | Build and test a new trading strategy |

### 5. Reference
Technical details and API documentation.

| Document | Description |
|----------|-------------|
| [Scoring Criteria Reference](reference/scoring-criteria.md) | All available metrics for stock screening |
| [CLI Commands Reference](reference/cli-commands.md) | Command-line tool usage |

---

## Quick Overview: How the System Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                   │
│  Polygon API → build_polygon_cache.py → Parquet Files               │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                             │
│  build_technical_features.py → RSI, ATR, MACD, Bollinger Bands      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                       ANALYSIS LAYER                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  ML Models  │  │ Backtesting │  │   Factor    │                  │
│  │ (LightGBM)  │  │ (VectorBT)  │  │ Evaluation  │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND                                      │
│  React App → Stock Scoring → Charts → Factor Analysis               │
└─────────────────────────────────────────────────────────────────────┘
```

## The Default Trading Strategy

The project comes configured for **swing trading small-cap stocks** with these default criteria:

| Criteria | Target | Why |
|----------|--------|-----|
| **Price Range** | $1 - $8 | Small-cap stocks with room to move |
| **Daily Range** | $0.10 - $1.00 | Enough movement to profit, not too volatile |
| **Sweet Spot Days** | High % | Days with $0.20-$0.60 range are ideal |
| **Liquidity Multiple** | High | Can easily enter/exit 10k share positions |
| **Range Consistency** | Low CV | Predictable daily ranges reduce risk |
| **Tradability Score** | High | Combined metric of all factors |

---

## Next Steps

1. **New users**: Start with [Quick Start](getting-started/01-quick-start.md)
2. **Understand the strategy**: Read [Swing Trading Philosophy](strategy-guide/01-swing-trading-philosophy.md)
3. **Master the tools**: Work through the [Tier 1 Features](tier1-features/01-feature-engineering.md)
4. **Apply daily**: Follow the [Daily Screening Workflow](workflows/01-daily-screening.md)

---

*Last Updated: January 2026*

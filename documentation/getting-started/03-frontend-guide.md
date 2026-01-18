# Using the Frontend

This guide walks you through the Stonks web interface and how to use each panel effectively.

## Accessing the Frontend

1. Start the backend: `python api_server.py`
2. Start the frontend: `cd frontend && npm run dev`
3. Open http://localhost:5173

## Main Interface Layout

The frontend has several main sections accessible via tabs:

```
┌────────────────────────────────────────────────────────┐
│  [Charts]  [Scoring]  [Factor Eval]  [Performance]     │
├────────────────────────────────────────────────────────┤
│                                                        │
│                   Main Content Area                    │
│                                                        │
└────────────────────────────────────────────────────────┘
```

## Charts Panel

### Purpose
View price history and technical indicators for individual stocks.

### Features
- **Candlestick charts**: OHLC price data
- **Volume bars**: Trading activity
- **Technical overlays**: Moving averages, Bollinger Bands
- **Indicator subplots**: RSI, MACD, etc.

### How to Use
1. Enter a ticker symbol (e.g., "AAPL")
2. Select date range
3. Toggle indicators on/off
4. Zoom and pan to analyze specific periods

### Tips for Short-Term Trading
- Look for RSI divergences (price makes new low, RSI makes higher low)
- Watch for MACD histogram crossovers
- Note Bollinger Band squeezes (low volatility → potential breakout)

## Scoring Panel

### Purpose
Find the best stocks for swing trading based on multiple criteria.

### The Default Swing Trading Preset

The "Swing Trading Default" preset is optimized for:
- **10,000 share position sizes**
- **$0.20-$0.60 daily profit targets**
- **Stocks priced $1-$8**

### Criteria Explained

| Criteria | Weight | Target | Why |
|----------|--------|--------|-----|
| **Tradability Score** | 3.0 | 0-100 (higher better) | Combined metric of all trading factors |
| **Avg Daily Range ($)** | 2.5 | $0.10-$1.00 | Enough movement to profit from |
| **Liquidity Multiple** | 2.0 | High | Can trade 10k shares without moving price |
| **Sweet Spot Days %** | 1.5 | High | Consistent ideal range days |
| **Range Consistency (CV)** | 1.0 | Low (inverted) | Predictable = less risk |
| **Current Price** | 10.0 | $0-$8 | Small-cap focus |

### How Scoring Works

1. Each stock is evaluated on all criteria
2. Values are normalized to 0-100 scale
3. Weighted sum produces final score
4. Stocks ranked highest to lowest

### Using the Scoring Panel

1. **Load a preset** or create custom criteria
2. **Set filters**:
   - Months of data to analyze
   - Minimum trading days
   - Minimum average volume
   - Price range
3. **Click "Score Stocks"**
4. **Review results** in the table below

### Interpreting Results

| Score Range | Interpretation |
|-------------|----------------|
| 80-100 | Excellent - Prime trading candidate |
| 60-80 | Good - Worth detailed analysis |
| 40-60 | Fair - May have one weak area |
| Below 40 | Poor - Likely has liquidity or range issues |

### Creating Custom Presets

1. Add/remove criteria using the dropdown
2. Adjust weights (higher = more important)
3. Set min/max value ranges
4. Toggle "Invert" for metrics where lower is better
5. Save as a named preset for future use

## Factor Evaluation Panel

### Purpose
Understand which technical indicators actually predict future returns.

### Key Metrics

| Metric | What It Means |
|--------|---------------|
| **Information Coefficient (IC)** | Correlation between factor and future returns (-1 to +1) |
| **Mean IC** | Average IC over all time periods |
| **t-statistic** | Statistical significance (>2 is significant) |
| **Positive %** | How often IC is positive |
| **Quantile Spread** | Return difference between high and low factor values |

### Reading the IC Chart

- **Positive IC**: High factor values predict positive returns
- **Negative IC**: High factor values predict negative returns
- **IC near 0**: Factor doesn't predict returns

### Quantile Returns Chart

Shows average returns for each factor quintile:
- **Q1**: Lowest 20% of factor values
- **Q5**: Highest 20% of factor values
- **Ideal pattern**: Monotonic increase from Q1 to Q5 (or decrease for inverted factors)

### Using Factor Evaluation

1. Select a factor (e.g., RSI, momentum)
2. Choose forward periods (1-day, 5-day, 10-day)
3. Review IC time series and quantile returns
4. Compare multiple factors to find the strongest

### For Short-Term Trading

Focus on:
- **1-day and 5-day IC**: Most relevant for swing trading
- **High t-stat factors**: Statistically reliable signals
- **Consistent positive %**: Works in different market conditions

## Performance Panel

### Purpose
Analyze historical performance of strategies or individual stocks.

### Metrics Displayed

| Category | Metrics |
|----------|---------|
| **Returns** | Total return, Annualized return |
| **Risk-Adjusted** | Sharpe ratio, Sortino ratio, Calmar ratio |
| **Drawdown** | Maximum drawdown, Drawdown duration |
| **Risk** | VaR (95%), CVaR (95%) |
| **Trade Quality** | Win rate, Profit factor, Best/worst day |

### Charts

- **Equity Curve**: Cumulative returns over time
- **Drawdown Chart**: Underwater plot showing losses from peak
- **Rolling Sharpe**: Risk-adjusted performance over time

### Interpreting Performance

| Metric | Good Value | Excellent Value |
|--------|------------|-----------------|
| Sharpe Ratio | > 1.0 | > 2.0 |
| Sortino Ratio | > 1.5 | > 3.0 |
| Max Drawdown | < 20% | < 10% |
| Win Rate | > 50% | > 60% |
| Profit Factor | > 1.5 | > 2.0 |

## Tips for Effective Use

### Morning Routine
1. Open Scoring panel, run with default preset
2. Review top 10-20 stocks
3. Check Charts for interesting candidates
4. Note any with strong factor signals

### Before Trading
1. Verify liquidity on Charts panel (volume bars)
2. Check recent price action (any news events?)
3. Review Performance metrics for the stock

### Research Mode
1. Use Factor Evaluation to discover new signals
2. Test hypotheses with custom scoring criteria
3. Backtest strategies on promising stocks

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Enter` | Submit current form |
| `Esc` | Close modal/dropdown |
| `Tab` | Navigate fields |

## Mobile Usage

The frontend is responsive but optimized for desktop. For mobile:
- Use landscape orientation
- Charts may require horizontal scrolling
- Scoring panel works well

---

*Continue to: [Swing Trading Philosophy](../strategy-guide/01-swing-trading-philosophy.md)*

# Performance Analysis (Tier 1)

**Status**: ✅ Complete  
**Module**: `data_processing/performance_analysis.py`  
**Book Reference**: Python for Algorithmic Trading Cookbook, Chapters 39-43

## Why This Matters for Short-Term Trading

Performance analysis answers: **How well is my strategy actually doing?**

Beyond simple returns, you need to understand:
1. **Risk-adjusted performance**: Are returns worth the risk taken?
2. **Drawdown behavior**: What's the worst you can expect to lose?
3. **Consistency**: Does it work in different market conditions?
4. **Benchmark comparison**: Are you beating buy-and-hold?

This information helps you:
- Size positions appropriately
- Set realistic expectations
- Know when to stop trading a strategy
- Explain results to yourself (or investors)

## Implemented Metrics

### Return Metrics

#### Total Return
```python
total_return = (final_value / initial_value) - 1
```
Simple measure of how much money you made.

#### Annualized Return (CAGR)
```python
cagr = (1 + total_return) ** (252 / trading_days) - 1
```
Standardizes to annual basis for comparison.

#### Annualized Volatility
```python
volatility = daily_returns.std() * sqrt(252)
```
Measures how much returns vary year-over-year.

### Risk-Adjusted Metrics

#### Sharpe Ratio
```python
sharpe = (annualized_return - risk_free_rate) / annualized_volatility
```

**The most important metric.** Measures return per unit of risk.

| Sharpe | Interpretation |
|--------|----------------|
| < 0 | Losing money |
| 0-0.5 | Below average |
| 0.5-1.0 | Acceptable |
| 1.0-2.0 | Good |
| 2.0-3.0 | Excellent |
| > 3.0 | Suspicious (check for errors) |

#### Sortino Ratio
```python
sortino = (annualized_return - risk_free_rate) / downside_deviation
```

Like Sharpe, but only penalizes **downside** volatility. Upside volatility is good!

Higher Sortino than Sharpe means positive skew (big winners, small losers).

#### Calmar Ratio
```python
calmar = annualized_return / abs(max_drawdown)
```

Return relative to worst loss. Good for understanding risk/reward tradeoff.

### Drawdown Metrics

#### Maximum Drawdown
```python
def compute_max_drawdown(returns):
    """
    Maximum peak-to-trough decline.
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()
```

**Critical metric.** This is what you'll experience in your worst period.

#### Maximum Drawdown Duration

How many days until you recovered from the worst drawdown.

Important for psychology: Can you handle 60 days underwater?

### Risk Metrics

#### Value at Risk (VaR)
```python
var_95 = returns.quantile(0.05)  # 5th percentile
```

95% VaR: "On 95% of days, you won't lose more than X%"

#### Conditional VaR (CVaR / Expected Shortfall)
```python
cvar_95 = returns[returns <= var_95].mean()
```

Average loss when you *do* exceed VaR. Measures tail risk.

### Trade Metrics

#### Win Rate
```python
win_rate = positive_days / total_days
```

Percentage of days with positive returns.

#### Profit Factor
```python
profit_factor = sum(positive_returns) / abs(sum(negative_returns))
```

How much bigger are your wins than your losses?

| PF | Interpretation |
|----|----------------|
| < 1.0 | Losing money |
| 1.0-1.2 | Breakeven after costs |
| 1.2-1.5 | Decent |
| 1.5-2.0 | Good |
| > 2.0 | Excellent |

### Rolling Metrics

#### Rolling Sharpe
```python
rolling_sharpe = returns.rolling(60).apply(sharpe_function)
```

Sharpe ratio over sliding 60-day windows. Shows consistency over time.

#### Rolling Volatility
```python
rolling_vol = returns.rolling(20).std() * sqrt(252)
```

How volatility changes over time. Important for position sizing.

## Using the Module

### Analyze Single Stock

```bash
uv run python data_processing/performance_analysis.py \
    --data-path data/cache/daily_2025.parquet \
    --ticker AAPL \
    --benchmark SPY \
    --output data/cache/performance/aapl_analysis.json
```

### Analyze Portfolio

```bash
uv run python data_processing/performance_analysis.py \
    --data-path data/cache/daily_2025.parquet \
    --output data/cache/performance/portfolio_analysis.json
```

## Understanding Output

### Full Output Structure

```python
{
    'summary': {
        'total_return': 0.15,
        'total_return_pct': 15.0,
        'annualized_return': 0.12,
        'annualized_return_pct': 12.0,
        'annualized_volatility': 0.18,
        'annualized_volatility_pct': 18.0,
        'trading_days': 252,
    },
    'risk_adjusted': {
        'sharpe_ratio': 0.67,
        'sortino_ratio': 0.95,
        'calmar_ratio': 0.85,
    },
    'drawdown': {
        'max_drawdown_pct': -14.2,
        'max_drawdown_duration_days': 45,
    },
    'risk': {
        'var_95_pct': -2.1,
        'cvar_95_pct': -3.2,
    },
    'trade_metrics': {
        'win_rate_pct': 54.2,
        'profit_factor': 1.35,
        'best_day_pct': 4.5,
        'worst_day_pct': -5.2,
        'avg_daily_return_pct': 0.05,
    },
    'time_series': {
        'cumulative_returns': {...},
        'rolling_sharpe_60d': {...},
        'rolling_volatility_20d': {...},
        'drawdown': {...},
    },
    'benchmark_comparison': {  # If benchmark provided
        'beta': 1.12,
        'alpha': 0.03,
        'alpha_pct': 3.0,
        'information_ratio': 0.45,
        'correlation': 0.85,
        'benchmark_sharpe': 0.55,
    },
}
```

## Benchmark Comparison

### Why Compare to Benchmark?

A 20% return sounds great, but if the market returned 30%, you underperformed!

### Benchmark Metrics

#### Beta
```python
beta = cov(strategy_returns, market_returns) / var(market_returns)
```

Measures market sensitivity:
- β = 1: Moves with market
- β > 1: More volatile than market
- β < 1: Less volatile than market
- β < 0: Inverse to market

#### Alpha (Jensen's Alpha)
```python
alpha = strategy_return - (risk_free + beta * (market_return - risk_free))
```

Return in excess of what beta predicts. **This is your skill**.

#### Information Ratio
```python
ir = (strategy_return - benchmark_return) / tracking_error
```

Excess return per unit of tracking error. Measures consistency of outperformance.

| IR | Interpretation |
|----|----------------|
| < 0 | Underperforming benchmark |
| 0-0.5 | Slight outperformance |
| 0.5-1.0 | Good outperformance |
| > 1.0 | Excellent |

## Interpreting Performance

### Good Swing Trading Performance

| Metric | Target | Excellent |
|--------|--------|-----------|
| Annualized Return | 15-30% | 40%+ |
| Sharpe Ratio | 1.0+ | 2.0+ |
| Sortino Ratio | 1.5+ | 2.5+ |
| Max Drawdown | < 15% | < 10% |
| Win Rate | > 50% | > 60% |
| Profit Factor | > 1.5 | > 2.0 |

### Warning Signs

| Sign | Possible Issue |
|------|----------------|
| Sharpe > 3.0 | Overfitting or calculation error |
| Very high win rate (>80%) | Taking profits too early |
| Very low win rate (<35%) | Not cutting losses fast enough |
| Increasing drawdowns | Strategy may be breaking down |
| Declining rolling Sharpe | Regime change or alpha decay |

## Using Charts

### Equity Curve

Shows cumulative returns over time.

**Look for**:
- Steady upward slope (consistent returns)
- Limited drawdowns (dips from peaks)
- No long flat periods

**Warning signs**:
- Sharp drawdowns
- Long periods without new highs
- Recent underperformance

### Drawdown Chart

Shows loss from peak over time.

**Look for**:
- Shallow drawdowns (< 10%)
- Quick recoveries
- No drawdown exceeding your risk tolerance

**Use for**: Setting maximum position sizes and stop-losses.

### Rolling Sharpe

Shows 60-day Sharpe ratio over time.

**Look for**:
- Consistently positive
- Not highly variable
- No sustained negative periods

**Warning signs**:
- Sharpe going negative
- High variance (inconsistent)
- Declining trend

## Practical Application

### Position Sizing from Volatility

```python
# Size position to target 1% daily portfolio volatility
target_vol = 0.01
stock_vol = rolling_vol / sqrt(252)  # Daily vol
position_size = target_vol / stock_vol
```

### Drawdown-Based Stop

```python
# If strategy drawdown exceeds 2x historical max, stop trading
if current_drawdown < 2 * historical_max_drawdown:
    pause_trading()
```

### Performance-Based Position Adjustment

```python
# Reduce size when rolling Sharpe is negative
if rolling_sharpe[-1] < 0:
    position_size *= 0.5
```

## API Integration

### Endpoint

```python
@app.get("/api/performance/{ticker}")
async def get_performance(ticker: str, benchmark: str = None):
    """Get performance analysis for a ticker."""
```

### Frontend Display

The Performance Panel shows:
- Equity curve chart
- Drawdown chart
- Rolling Sharpe chart
- Key metrics table
- Benchmark comparison (if available)

## Common Questions

### "My Sharpe is low but I'm making money"

Sharpe penalizes volatility. If you're capturing big moves, your returns may be high but volatile. Check Sortino — if it's significantly higher, you have positive skew (good).

### "Max drawdown seems too high"

Small-caps are volatile. 15-20% drawdowns are normal. The question is:
1. Does it recover?
2. Can you handle it emotionally?
3. Is position sizing appropriate?

### "Rolling Sharpe is declining"

Could be:
- Market regime change
- Strategy alpha decaying
- Seasonal effect

**Action**: Investigate and consider reducing size until stabilization.

### "My benchmark comparison shows negative alpha"

You're underperforming risk-adjusted. Options:
1. Reduce fees/slippage
2. Improve signal quality
3. Consider just buying the benchmark

## Best Practices

1. **Look at multiple metrics**: No single metric tells the whole story
2. **Consider time period**: Performance varies by market regime
3. **Include transaction costs**: Gross vs net performance can differ significantly
4. **Use appropriate benchmark**: Compare small-cap strategy to small-cap index
5. **Monitor over time**: Check rolling metrics regularly
6. **Set realistic expectations**: 1.0-1.5 Sharpe is good, 2.0+ is excellent

---

*Continue to: [Daily Screening Workflow](../workflows/01-daily-screening.md)*

# Strategy Development Workflow

A complete guide to developing, testing, and deploying a new trading strategy using Stonks.

## Overview

Building a trading strategy is an iterative process:

```
Idea → Hypothesis → Backtest → Analyze → Refine → Deploy → Monitor
```

This workflow ensures you:
1. Start with a clear hypothesis
2. Test rigorously before risking capital
3. Understand expected performance
4. Have criteria for when to stop trading the strategy

## Phase 1: Strategy Design

### Define Your Edge

What market inefficiency are you exploiting?

**Mean Reversion Example**:
> Small-cap stocks that become oversold (RSI < 30) while in non-trending 
> markets (ADX < 25) tend to bounce within 3-5 days.

**Momentum Example**:
> Stocks breaking out above 20-day highs with volume confirmation 
> continue trending for 5-10 days on average.

### Specify Entry Rules

Be precise and unambiguous:

```
ENTRY CONDITIONS (all must be true):
1. RSI(14) < 30
2. ADX(14) < 25
3. Price > 20-day SMA (uptrend context)
4. Volume > 0.8x 20-day average (not dying)
5. Price between $1-$8 (liquidity screen)
6. Liquidity multiple > 20x
```

### Specify Exit Rules

```
EXIT CONDITIONS (any triggers exit):
1. RSI(14) > 50 (mean reversion complete)
2. Price drops 2x ATR from entry (stop loss)
3. 5 days elapsed without target (time stop)
4. Price > entry + 3x ATR (profit target)
```

### Document Risk Parameters

```
RISK PARAMETERS:
- Max risk per trade: 1% of account
- Position sizing: Based on 2x ATR stop
- Max positions: 5 concurrent
- Max sector exposure: 2 positions per sector
- Max daily trades: 3 new entries
```

## Phase 2: Initial Backtest

### Implement Entry/Exit Signals

Create signals in Python:

```python
def generate_signals(df):
    """Generate entry and exit signals for mean reversion strategy."""
    
    # Entry conditions
    entries = (
        (df['rsi_14'] < 30) &
        (df['adx_14'] < 25) &
        (df['close'] > df['sma_20']) &
        (df['volume'] > df['volume'].rolling(20).mean() * 0.8)
    )
    
    # Exit conditions
    exits = (
        (df['rsi_14'] > 50) |
        (df['close'] < df['close'].shift(1) - 2 * df['atr_14']) |
        # Time stop and profit target handled in backtest
    )
    
    return entries, exits
```

### Run Backtest

```bash
uv run python data_processing/backtest_vectorbt.py \
    --strategy custom \
    --signal-module my_strategy \
    --initial-cash 100000 \
    --fees 0.001 \
    --slippage 0.005 \
    --output data/cache/backtest/mean_reversion_v1.json
```

### Review Initial Results

```python
import json
with open('data/cache/backtest/mean_reversion_v1.json') as f:
    results = json.load(f)

print(f"Total Return: {results['performance']['total_return']:.1%}")
print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {results['performance']['max_drawdown']:.1%}")
print(f"Win Rate: {results['performance']['win_rate']:.1%}")
print(f"Total Trades: {results['trades']['total_trades']}")
```

## Phase 3: Analyze & Refine

### Check for Red Flags

| Red Flag | Possible Cause | Solution |
|----------|----------------|----------|
| Sharpe > 3.0 | Overfitting | Simplify rules |
| < 30 trades | Not enough data | Expand universe/time |
| Win rate > 80% | Taking profits too early | Review exit rules |
| Win rate < 40% | Bad entries or no stops | Review entry rules |
| Profit factor < 1.2 | Edge too small | Better entry timing |

### Examine Trade Distribution

```python
# Plot trade returns histogram
import matplotlib.pyplot as plt
trade_returns = results['trade_returns']
plt.hist(trade_returns, bins=30)
plt.title('Trade Return Distribution')
plt.xlabel('Return')
plt.ylabel('Frequency')
plt.show()
```

**Look for**:
- Fat right tail (big winners)
- Thin left tail (cut losers)
- No cluster at -100% (stop loss working)

### Review Equity Curve

```python
# Plot equity curve
equity = results['equity_curve']
plt.plot(equity['dates'], equity['values'])
plt.title('Equity Curve')
plt.xlabel('Date')
plt.ylabel('Portfolio Value')
plt.show()
```

**Look for**:
- Steady upward slope
- Limited drawdowns
- No long flat periods
- Recent performance similar to historical

### Iterate on Rules

Based on analysis, adjust:

```python
# Version 2: Tighter entry, wider stop
entries = (
    (df['rsi_14'] < 25) &        # Was 30, now stricter
    (df['adx_14'] < 20) &        # Was 25, now stricter
    (df['close'] > df['sma_20']) &
    (df['volume'] > df['volume'].rolling(20).mean() * 0.9)  # Was 0.8
)

# Use 2.5x ATR stop instead of 2x
```

Re-run backtest, compare results.

## Phase 4: Walk-Forward Optimization

### Avoid Overfitting

Don't just optimize on full history. Use walk-forward:

```python
def walk_forward_backtest(df, train_months=6, test_months=1):
    """
    Train on 6 months, test on next 1 month, roll forward.
    """
    results = []
    
    for start_idx in range(0, len(df), test_months * 21):
        train_end = start_idx + train_months * 21
        test_end = train_end + test_months * 21
        
        train_df = df.iloc[start_idx:train_end]
        test_df = df.iloc[train_end:test_end]
        
        # Train on in-sample
        best_params = optimize_parameters(train_df)
        
        # Test on out-of-sample
        oos_result = backtest(test_df, best_params)
        results.append(oos_result)
    
    return aggregate_results(results)
```

### Compare In-Sample vs Out-of-Sample

| Metric | In-Sample | Out-of-Sample | Ratio |
|--------|-----------|---------------|-------|
| Sharpe | 1.8 | 1.1 | 61% |
| Return | 45% | 22% | 49% |
| Win Rate | 58% | 52% | 90% |

**Acceptable**: OOS is 50%+ of in-sample
**Concerning**: OOS is < 30% of in-sample

## Phase 5: Stress Testing

### Different Market Regimes

```python
# Test across market conditions
regimes = {
    'bull': ('2023-01-01', '2023-06-30'),
    'bear': ('2022-01-01', '2022-06-30'),
    'sideways': ('2023-07-01', '2023-12-31'),
    'volatile': ('2020-02-01', '2020-04-30'),
}

for regime, (start, end) in regimes.items():
    result = backtest(df, start_date=start, end_date=end)
    print(f"{regime}: Sharpe={result['sharpe']:.2f}")
```

**Good strategy**: Works in most regimes
**Concerning**: Only works in bull markets

### Monte Carlo Simulation

Bootstrap confidence intervals:

```python
def monte_carlo_backtest(df, n_simulations=1000):
    """
    Resample trades to get distribution of possible outcomes.
    """
    base_trades = backtest(df)['trade_returns']
    
    results = []
    for _ in range(n_simulations):
        # Randomly resample trades with replacement
        sampled_trades = np.random.choice(base_trades, size=len(base_trades))
        simulated_return = np.prod(1 + sampled_trades) - 1
        results.append(simulated_return)
    
    return {
        'median_return': np.median(results),
        'return_5pct': np.percentile(results, 5),
        'return_95pct': np.percentile(results, 95),
        'prob_positive': np.mean(np.array(results) > 0),
    }
```

**Use for**: Understanding range of possible outcomes

## Phase 6: Paper Trading

### Deploy to Paper Account

Before real money:
1. Run strategy with real-time data
2. Execute paper trades
3. Track for 1-3 months minimum

### Monitor Key Metrics

Daily checklist:
- [ ] Signals generated as expected?
- [ ] Paper execution prices realistic?
- [ ] Performance tracking historical?

### Compare to Backtest

| Metric | Backtest | Paper Trading |
|--------|----------|---------------|
| Trades/Month | 15 | 12 |
| Win Rate | 55% | 52% |
| Avg Win | 2.1% | 1.8% |
| Avg Loss | -1.2% | -1.4% |
| Sharpe (annualized) | 1.2 | 0.9 |

**Expected**: Some degradation (15-30%)
**Concerning**: Major degradation (>50%)

## Phase 7: Live Deployment

### Position Sizing

Start small:

```
Week 1-2: 25% of target position size
Week 3-4: 50% of target position size
Month 2: 75% of target position size
Month 3+: Full position size
```

### Risk Controls

Implement automatic stops:

```python
# Daily loss limit
if daily_pnl < -0.02 * account_value:
    close_all_positions()
    halt_new_entries_today()

# Drawdown limit
if drawdown < -0.10:
    reduce_position_sizes(0.5)
    
if drawdown < -0.15:
    halt_strategy()
    review_required()
```

### Performance Tracking

Track live vs expected:

```python
# Weekly review
def weekly_review(live_results, backtest_baseline):
    """Compare live performance to expectations."""
    
    metrics = {
        'live_sharpe': compute_sharpe(live_results),
        'expected_sharpe': backtest_baseline['sharpe'],
        'deviation': live_sharpe / expected_sharpe - 1,
    }
    
    if metrics['deviation'] < -0.5:
        print("WARNING: Performance 50%+ below expectations")
    
    return metrics
```

## Phase 8: Ongoing Monitoring

### Regular Reviews

**Daily**:
- Check positions and P&L
- Note any anomalies

**Weekly**:
- Compare to backtest expectations
- Review any losing trades
- Check rolling Sharpe

**Monthly**:
- Full performance review
- Factor IC analysis
- Consider strategy adjustments

### Decay Detection

Strategies degrade over time. Watch for:

```python
# Rolling Sharpe declining
rolling_sharpe = compute_rolling_sharpe(returns, 60)
if rolling_sharpe[-1] < 0.5 and rolling_sharpe[-60] > 1.0:
    print("ALERT: Sharpe degrading significantly")

# Win rate declining
rolling_win_rate = compute_rolling_win_rate(trades, 30)
if rolling_win_rate[-1] < 0.40:
    print("ALERT: Win rate below threshold")
```

### When to Stop

Clear criteria for halting:

| Condition | Action |
|-----------|--------|
| Drawdown > 15% | Reduce size 50%, review |
| Drawdown > 20% | Halt strategy |
| 3 months negative | Full strategy review |
| Rolling Sharpe < 0 for 2 months | Consider retiring |

## Strategy Documentation Template

Keep a strategy spec document:

```markdown
# Strategy: RSI Mean Reversion v2

## Overview
Mean reversion strategy for oversold small-caps.

## Rules
### Entry
- RSI(14) < 25
- ADX(14) < 20
- Price > SMA(20)
- Liquidity > 20x

### Exit
- RSI(14) > 50
- Stop: 2.5x ATR
- Time: 5 days max
- Target: 3x ATR

## Backtest Results
- Period: 2022-2025
- Sharpe: 1.2
- Max DD: 12%
- Win Rate: 54%
- Trades: 180

## Live Results
- Started: 2025-01-01
- Current Sharpe: 0.95
- Max DD: 8%

## Version History
- v1 (2024-06): Initial version
- v2 (2024-09): Tighter entry (RSI 25 vs 30)

## Review Schedule
- Weekly performance check
- Monthly full review
```

---

*Continue to: [Scoring Criteria Reference](../reference/scoring-criteria.md)*

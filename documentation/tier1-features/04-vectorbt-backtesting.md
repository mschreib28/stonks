# VectorBT Backtesting (Tier 1)

**Status**: ✅ Complete  
**Module**: `data_processing/backtest_vectorbt.py`  
**Book Reference**: Python for Algorithmic Trading Cookbook, Chapters 29-31

## Why This Matters for Short-Term Trading

Backtesting answers the critical question: **Would this strategy have made money?**

Before risking real capital:
1. **Test your ideas** on historical data
2. **Understand expected drawdowns** so you don't panic-sell
3. **Optimize parameters** without paying tuition to the market
4. **Compare strategies** objectively

VectorBT is particularly suited for swing trading because it's:
- **Fast**: Vectorized operations test thousands of parameter combinations
- **Realistic**: Models fees, slippage, and portfolio constraints
- **Flexible**: Works with any entry/exit signals

## Implemented Strategies

### MACD Crossover Strategy

Trades based on MACD histogram direction changes.

```python
def backtest_macd_strategy(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    initial_cash: float = 100000,
    fees: float = 0.001,        # 0.1% per trade
    slippage: float = 0.001,    # 0.1% slippage
) -> dict:
```

**Entry Signal**: MACD histogram crosses from negative to positive  
**Exit Signal**: MACD histogram crosses from positive to negative

**Best for**: Trend-following, momentum capture

### RSI Mean Reversion Strategy

Trades oversold/overbought conditions.

```python
def backtest_rsi_strategy(
    data: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    initial_cash: float = 100000,
    fees: float = 0.001,
    slippage: float = 0.001,
) -> dict:
```

**Entry Signal**: RSI crosses below oversold threshold (buy the dip)  
**Exit Signal**: RSI crosses above overbought threshold (sell the rally)

**Best for**: Range-bound markets, mean reversion

### Custom Signals Strategy

Test any entry/exit logic you define.

```python
def backtest_custom_signals(
    data: pd.DataFrame,
    entries: pd.Series,        # Boolean series: True = enter
    exits: pd.Series,          # Boolean series: True = exit
    initial_cash: float = 100000,
    fees: float = 0.001,
    slippage: float = 0.001,
    strategy_name: str = 'Custom Strategy',
) -> dict:
```

**Example custom signals**:
```python
# Mean reversion: RSI oversold AND below lower Bollinger Band
entries = (data['rsi_14'] < 30) & (data['bb_pct'] < 0)

# Exit: RSI back to neutral OR hit upper band
exits = (data['rsi_14'] > 50) | (data['bb_pct'] > 0.8)
```

## Walk-Forward Optimization

Avoid overfitting by testing parameters out-of-sample.

```python
def walk_forward_optimization(
    data: pd.DataFrame,
    strategy_func: callable,
    param_grid: dict,
    train_months: int = 6,
    test_months: int = 1,
    metric: str = 'sharpe_ratio',
) -> dict:
```

**How it works**:
1. Train on 6 months, find best parameters
2. Test those parameters on next 1 month (out-of-sample)
3. Roll forward, repeat
4. Report aggregate out-of-sample performance

**This is the closest simulation to live trading!**

## CLI Usage

### Basic MACD Backtest

```bash
uv run python data_processing/backtest_vectorbt.py \
    --data-path data/cache/daily_2025.parquet \
    --strategy macd \
    --ticker AAPL \
    --output data/cache/backtest_results.json
```

### RSI Backtest with Custom Parameters

```bash
uv run python data_processing/backtest_vectorbt.py \
    --strategy rsi \
    --ticker MSFT \
    --rsi-period 14 \
    --oversold 25 \
    --overbought 75
```

### MACD with Custom Periods

```bash
uv run python data_processing/backtest_vectorbt.py \
    --strategy macd \
    --fast-period 8 \
    --slow-period 21 \
    --signal-period 5
```

### Multi-Ticker Backtest

```bash
uv run python data_processing/backtest_vectorbt.py \
    --strategy macd \
    --ticker ALL \
    --output data/cache/portfolio_backtest.json
```

## Understanding Output

### Performance Metrics

```python
{
    'strategy': 'MACD Crossover',
    'parameters': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9
    },
    'performance': {
        'total_return': 0.25,           # 25% total return
        'cagr': 0.18,                   # 18% annualized
        'sharpe_ratio': 1.2,            # Risk-adjusted return
        'sortino_ratio': 1.8,           # Downside-adjusted return
        'max_drawdown': -0.12,          # -12% worst peak-to-trough
        'win_rate': 0.55,               # 55% of trades profitable
        'profit_factor': 1.6,           # Gross profit / gross loss
    },
    'trades': {
        'total_trades': 45,
        'avg_trade_return': 0.008,      # 0.8% average per trade
        'best_trade': 0.15,             # Best single trade: +15%
        'worst_trade': -0.08,           # Worst single trade: -8%
    },
    'equity_curve': {...},              # For plotting
    'drawdowns': {...},                 # Drawdown time series
}
```

### Key Metrics Explained

| Metric | Good Value | Great Value | Meaning |
|--------|------------|-------------|---------|
| **Sharpe Ratio** | >1.0 | >2.0 | Return per unit of risk |
| **Sortino Ratio** | >1.5 | >2.5 | Return per unit of downside risk |
| **Max Drawdown** | <15% | <10% | Worst loss from peak |
| **Win Rate** | >50% | >60% | Percentage of winning trades |
| **Profit Factor** | >1.5 | >2.0 | How much bigger wins are than losses |
| **CAGR** | >10% | >20% | Annual return |

### Interpreting Results

**Good backtest**:
- Sharpe > 1.0 (sustainable risk/reward)
- Max drawdown < 15% (survivable losses)
- Win rate reasonable for strategy (50%+ for mean reversion, 40%+ for trend)
- Consistent across different periods

**Warning signs**:
- Very high Sharpe (>3.0) — likely overfitting
- Too few trades (<20) — not enough data
- Win rate doesn't match strategy type
- Great early performance, poor recent performance

## Realistic Backtesting

### Modeling Costs

VectorBT includes transaction costs:

```python
fees=0.001,      # 0.1% commission (each way)
slippage=0.001,  # 0.1% slippage
```

**For small-caps, consider higher slippage**:
```python
# More realistic for illiquid stocks
fees=0.001,
slippage=0.005,  # 0.5% slippage for small-caps
```

### Position Sizing

Default is 100% allocation per trade. For realistic testing:

```python
# Limit position size
portfolio = vbt.Portfolio.from_signals(
    close=data['close'],
    entries=entries,
    exits=exits,
    size=0.2,  # 20% of capital per trade
    size_type='percent',
)
```

### Avoiding Common Mistakes

| Mistake | Why It's Wrong | Solution |
|---------|----------------|----------|
| **No transaction costs** | Real trades cost money | Include fees + slippage |
| **Look-ahead bias** | Using future data | Check feature timestamps |
| **Survivorship bias** | Only testing stocks that exist today | Include delisted stocks |
| **Overfitting** | Tuning to past data | Use walk-forward optimization |
| **Starting too recent** | Missing different market regimes | Test across multiple years |

## Walk-Forward Results

### Understanding Walk-Forward Output

```python
{
    'walk_forward_results': [
        {
            'train_period': '2024-01-01 to 2024-06-30',
            'test_period': '2024-07-01 to 2024-07-31',
            'best_params': {'fast': 10, 'slow': 22, 'signal': 7},
            'train_sharpe': 1.8,
            'test_sharpe': 0.9,  # Lower OOS is normal
        },
        # ... more windows
    ],
    'aggregate_metrics': {
        'mean_oos_sharpe': 1.05,
        'std_oos_sharpe': 0.35,
        'mean_oos_return': 0.08,
        'total_oos_return': 0.45,
    }
}
```

### Key Insight

**In-sample Sharpe will almost always be higher than out-of-sample**.

If your OOS Sharpe is:
- >50% of in-sample: Good, model generalizes
- 25-50% of in-sample: Acceptable, some overfitting
- <25% of in-sample: Too much overfitting, simplify

## Strategy Comparison

### Testing Multiple Strategies

```bash
# Run both strategies on same data
uv run python data_processing/backtest_vectorbt.py --strategy macd --ticker XYZ --output macd.json
uv run python data_processing/backtest_vectorbt.py --strategy rsi --ticker XYZ --output rsi.json
```

### Comparison Framework

| Metric | MACD | RSI | Winner |
|--------|------|-----|--------|
| Sharpe | 1.2 | 0.9 | MACD |
| Max DD | -12% | -8% | RSI |
| Win Rate | 45% | 58% | RSI |
| Profit Factor | 1.8 | 1.4 | MACD |
| Trades/Year | 15 | 30 | RSI (more opportunity) |

**Choose based on your priorities**:
- Want higher returns? MACD (higher Sharpe/PF)
- Want lower drawdowns? RSI (smaller losses)
- Want more trades? RSI (more active)

## Custom Strategy Examples

### Bollinger Band Breakout

```python
# Entry: Close above upper band with volume surge
entries = (
    (data['close'] > data['bb_upper']) & 
    (data['volume'] > data['volume'].rolling(20).mean() * 1.5)
)

# Exit: Close below middle band or trailing stop
exits = data['close'] < data['bb_middle']

results = backtest_custom_signals(data, entries, exits)
```

### RSI + ADX Combo

```python
# Entry: RSI oversold in non-trending market
entries = (
    (data['rsi_14'] < 30) & 
    (data['adx_14'] < 25)  # Weak trend = mean reversion works
)

# Exit: RSI back to 50 or ADX increases (trend starting)
exits = (data['rsi_14'] > 50) | (data['adx_14'] > 30)

results = backtest_custom_signals(data, entries, exits)
```

### MACD + Volume Confirmation

```python
# Entry: MACD crossover with volume confirmation
macd_cross_up = (
    (data['macd_hist'] > 0) & 
    (data['macd_hist'].shift(1) <= 0)
)
volume_surge = data['volume'] > data['volume'].rolling(10).mean()
entries = macd_cross_up & volume_surge

# Exit: MACD crosses down
exits = (data['macd_hist'] < 0) & (data['macd_hist'].shift(1) >= 0)

results = backtest_custom_signals(data, entries, exits)
```

## Best Practices

### 1. Test on Multiple Tickers
Don't just backtest on one stock. A strategy that works on AAPL might not work on small-caps.

### 2. Test Multiple Time Periods
Include bull markets, bear markets, and sideways markets.

### 3. Use Realistic Costs
Small-cap slippage can be 0.5-1%. Include it.

### 4. Walk-Forward for Parameter Selection
If you tune parameters, always validate out-of-sample.

### 5. Minimum Trade Count
Need at least 30+ trades for statistically meaningful results.

### 6. Check for Regime Dependence
A strategy that worked 2020-2023 might not work in 2024 (different Fed policy, etc).

---

*Continue to: [Alphalens Factor Evaluation](05-alphalens-factors.md)*

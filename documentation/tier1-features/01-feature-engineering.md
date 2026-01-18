# Feature Engineering (Tier 1)

**Status**: ✅ Complete  
**Module**: `data_processing/build_technical_features.py`  
**Book Reference**: Machine Learning for Algorithmic Trading, Chapter 2

## Why This Matters for Short-Term Trading

Feature engineering transforms raw price data into **predictive signals**. For swing trading small-caps, the right features help you:

1. **Identify overbought/oversold conditions** for mean reversion entries
2. **Detect trend strength** to ride momentum
3. **Measure volatility** for position sizing and stop placement
4. **Generate binary signals** for systematic strategies

Without good features, ML models and factor evaluation are useless—garbage in, garbage out.

## Implemented Features

### Momentum Indicators

#### RSI (Relative Strength Index)
```
Feature: rsi_14
Period: 14 days
Range: 0-100
```

**What it measures**: Ratio of average gains to average losses, scaled to 0-100.

**For swing trading**:
- RSI < 30: Oversold, potential bounce
- RSI > 70: Overbought, potential pullback
- RSI divergences: Powerful reversal signals

**How to use**:
- Factor evaluation: Check if RSI extremes predict reversals
- Entry trigger: Buy when RSI crosses above 30
- Exit trigger: Sell when RSI crosses below 70

#### Stochastic Oscillator
```
Features: stoch_k, stoch_d
Period: 14 days
Range: 0-100
```

**What it measures**: Current close relative to recent high-low range.

**For swing trading**:
- Faster than RSI, more signals
- Crossovers between %K and %D
- Works well in ranging markets

**How to use**:
- Buy: %K crosses above %D below 20
- Sell: %K crosses below %D above 80

#### Rate of Change (ROC)
```
Feature: roc_10
Period: 10 days
Range: Unbounded (typically -20% to +20%)
```

**What it measures**: Percentage price change over period.

**For swing trading**:
- Identifies momentum strength
- Extreme values suggest reversal
- Useful for momentum strategies

### Trend Indicators

#### ADX (Average Directional Index)
```
Feature: adx_14
Period: 14 days
Range: 0-100
```

**What it measures**: Trend strength regardless of direction.

**For swing trading**:
- ADX > 25: Strong trend (trade with trend)
- ADX < 20: Weak/no trend (mean reversion works)
- Rising ADX: Trend gaining strength

**How to use**:
- Filter: Only take momentum trades when ADX > 25
- Strategy switch: Mean reversion when ADX < 20

#### SMA Crossovers
```
Features: sma_cross_20_50, sma_cross_50_200
Values: -1, 0, +1
```

**What it measures**: Relative position of moving averages.

**For swing trading**:
- +1: Faster MA above slower MA (bullish)
- -1: Faster MA below slower MA (bearish)
- Crossover: Trend change signal

**How to use**:
- Bias: Only long when sma_cross_20_50 = +1
- Golden cross (50/200): Major trend signal

### Volatility Indicators

#### ATR (Average True Range)
```
Feature: atr_14
Period: 14 days
Range: Depends on price level
```

**What it measures**: Average daily range including gaps.

**For swing trading**:
- **Critical** for position sizing
- Stop-loss placement: 2× ATR below entry
- Profit target: 2-3× ATR above entry

**How to use**:
```python
stop_loss = entry_price - (2 * atr_14)
position_size = risk_amount / (2 * atr_14)
```

#### Bollinger Band %B
```
Feature: bb_pct
Period: 20 days, 2 std dev
Range: 0-1 (can exceed)
```

**What it measures**: Price position within Bollinger Bands.

**For swing trading**:
- bb_pct < 0: Below lower band (oversold)
- bb_pct > 1: Above upper band (overbought)
- bb_pct = 0.5: At middle band

**How to use**:
- Mean reversion: Buy when bb_pct < 0.05
- Breakout: Buy when bb_pct > 0.95 with volume

#### Historical Volatility
```
Feature: hist_vol_20
Period: 20 days
Range: 0-200%+ (annualized)
```

**What it measures**: Standard deviation of returns, annualized.

**For swing trading**:
- High vol (>50%): Wide stops, smaller positions
- Low vol (<20%): Tight stops, larger positions
- Vol expansion: Often follows breakouts

### Binary Signals

#### RSI Thresholds
```
Features: rsi_overbought, rsi_oversold
Values: 0 or 1
```

**What they measure**: RSI crossing threshold levels.

**For swing trading**:
- Direct entry/exit triggers
- Backtestable with VectorBT
- Combine with other signals

#### Bollinger Band Breakouts
```
Features: above_bb_upper, below_bb_lower
Values: 0 or 1
```

**What they measure**: Price breaking band boundaries.

**For swing trading**:
- Mean reversion: Buy below_bb_lower
- Breakout: Buy above_bb_upper with confirmation

#### Strong Trend
```
Feature: strong_trend
Values: 0 or 1
```

**What it measures**: ADX > 25 threshold.

**For swing trading**:
- Filter for momentum strategies
- Avoid mean reversion in strong trends

## Using Features Effectively

### For Stock Screening

Use features to filter candidates:
```bash
# Find oversold stocks in weak trends (mean reversion setup)
python data_processing/evaluate_factors.py \
    --factor-column rsi_14 \
    --filter "rsi_14 < 30 AND adx_14 < 20"
```

### For Factor Evaluation

Test which features predict returns:
```bash
# Evaluate RSI predictive power
python data_processing/evaluate_factors.py \
    --factor-column rsi_14 \
    --periods 1,5,10

# Check IC and quantile returns
```

**Interpreting results**:
- Positive IC: High values predict positive returns
- Negative IC: High values predict negative returns (or low values predict positive)
- For RSI (mean reversion): Expect negative IC (oversold = positive returns)

### For ML Models

Features feed into LightGBM for predictions:
```bash
python data_processing/train_ml_model.py \
    --data-path data/cache/technical_features.parquet \
    --model-type classifier
```

**Feature importance**: After training, check which features the model uses most.

### For Backtesting

Use binary signals with VectorBT:
```bash
python data_processing/backtest_vectorbt.py \
    --strategy rsi \
    --oversold 30 \
    --overbought 70
```

## Combining Features

### Mean Reversion Setup

Entry when:
- RSI < 30 (oversold)
- Below lower Bollinger Band
- ADX < 20 (no strong trend)

```python
entry_signal = (
    (df['rsi_14'] < 30) & 
    (df['bb_pct'] < 0) & 
    (df['adx_14'] < 20)
)
```

### Momentum Setup

Entry when:
- RSI between 50-70 (not overbought)
- Above SMA 20
- ADX > 25 (strong trend)
- Recent positive ROC

```python
entry_signal = (
    (df['rsi_14'].between(50, 70)) &
    (df['sma_cross_20_50'] == 1) &
    (df['adx_14'] > 25) &
    (df['roc_10'] > 0)
)
```

## Feature Performance Summary

Based on typical small-cap behavior:

| Feature | Best Use | IC Direction |
|---------|----------|--------------|
| RSI | Mean reversion | Negative (oversold → positive returns) |
| Stochastic | Mean reversion | Negative |
| ROC | Momentum | Positive (momentum continues) |
| ADX | Filter | N/A (filter, not signal) |
| SMA Cross | Trend following | Positive (align with trend) |
| ATR | Position sizing | N/A (not predictive) |
| BB %B | Mean reversion | Negative |

## Customizing Features

### Adding New Features

Edit `build_technical_features.py`:

```python
# Example: Add Williams %R
def compute_williams_r(high, low, close, period=14):
    highest_high = high.rolling(period).max()
    lowest_low = low.rolling(period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low)

df['williams_r_14'] = compute_williams_r(df['high'], df['low'], df['close'])
```

### Modifying Parameters

Adjust periods for different timeframes:
- Shorter periods (7, 10): More signals, more noise
- Longer periods (20, 30): Fewer signals, smoother

## Best Practices

1. **Don't overfit**: More features isn't always better
2. **Test IC**: Validate features actually predict returns
3. **Correlation**: Avoid redundant features (RSI and Stochastic are similar)
4. **Context matters**: Mean reversion works when ADX < 20
5. **Combine wisely**: 2-3 confirming signals beats 1 perfect signal

---

*Continue to: [Linear Models & Cross-Validation](02-linear-models.md)*

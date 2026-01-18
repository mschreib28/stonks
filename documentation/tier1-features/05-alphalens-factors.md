# Alphalens Factor Evaluation (Tier 1)

**Status**: ✅ Complete  
**Module**: `data_processing/evaluate_factors.py`  
**Book Reference**: Python for Algorithmic Trading Cookbook, Chapters 36-38

## Why This Matters for Short-Term Trading

Before using any technical indicator in your strategy, you need to answer: **Does this factor actually predict future returns?**

Factor evaluation helps you:
1. **Avoid false signals**: Not every indicator works for your universe
2. **Quantify edge**: Know exactly how predictive your signals are
3. **Compare factors**: Which indicators are strongest?
4. **Set realistic expectations**: Understand what's achievable

Alphalens methodology is the industry standard for factor analysis.

## Key Concepts

### Information Coefficient (IC)

The **correlation between factor values and future returns**.

```
IC = correlation(factor_rank, forward_return_rank)
```

| IC Value | Interpretation |
|----------|----------------|
| +0.10 | Strong positive predictive power |
| +0.05 | Moderate positive |
| +0.02 | Weak positive (still useful at scale) |
| 0.00 | No predictive power |
| -0.05 | Moderate negative (high factor → low returns) |

**For swing trading, IC of 0.02-0.05 is realistic and profitable.**

### Why Rank Correlation?

We use **Spearman (rank) correlation** because:
- Handles non-linear relationships
- Robust to outliers
- What matters is relative ranking, not absolute values

### Quantile Returns

Divide stocks into groups (quintiles) by factor value and measure each group's average return.

```
Q1 (lowest 20%) → mean return: -0.5%
Q2 (20-40%)     → mean return: -0.1%
Q3 (40-60%)     → mean return: +0.1%
Q4 (60-80%)     → mean return: +0.3%
Q5 (highest 20%) → mean return: +0.7%
```

**Good factor**: Monotonic increase from Q1 to Q5 (or decrease for inverted factors)

### Factor Spread

The return difference between top and bottom quintiles.

```
Spread = Q5_return - Q1_return
```

**Example**: If Q5 returns +0.7% and Q1 returns -0.5%, spread = 1.2%

Spread represents the **theoretical max alpha** from going long Q5 and short Q1.

## Implemented Capabilities

### Simple IC Calculation

```python
def compute_ic_simple(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: list[int] = [1, 5, 10],
) -> dict:
    """
    Compute Information Coefficient using rank correlation.
    
    Returns:
        IC metrics for each forward period (1-day, 5-day, 10-day)
    """
```

**Output**:
```python
{
    '1D': {
        'mean': 0.032,          # Average IC
        'std': 0.145,           # IC variability
        't_stat': 4.21,         # Statistical significance
        'positive_pct': 0.58,   # % of days with positive IC
        'ic_series': {...}      # IC over time
    },
    '5D': {...},
    '10D': {...}
}
```

### Quantile Returns Analysis

```python
def compute_factor_quantile_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: list[int] = [1, 5, 10],
    quantiles: int = 5,
) -> dict:
    """
    Compute mean returns by factor quantile.
    Shows if high/low factor values predict returns.
    """
```

**Output**:
```python
{
    '1D': {
        'mean_returns': {
            1: -0.001,  # Q1 (lowest factor)
            2: 0.0,
            3: 0.001,
            4: 0.002,
            5: 0.003,   # Q5 (highest factor)
        },
        'spread': 0.004  # Q5 - Q1
    },
}
```

### Complete Factor Evaluation

```python
def evaluate_factor(
    factor_path: str,
    factor_column: str,
    price_path: str,
    periods: list[int] = [1, 5, 10],
    quantiles: int = 5,
    output_path: Optional[str] = None,
) -> dict:
    """Complete factor evaluation for a single factor."""
```

### Multi-Factor Comparison

```python
def evaluate_all_factors(
    factor_path: str,
    price_path: str,
    factor_columns: Optional[list[str]] = None,
    periods: list[int] = [1, 5, 10],
    quantiles: int = 5,
    output_dir: Optional[str] = None,
) -> dict:
    """Evaluate and compare multiple factors."""
```

## CLI Usage

### Single Factor Evaluation

```bash
python data_processing/evaluate_factors.py \
    --factor-path data/cache/technical_features.parquet \
    --factor-column rsi_14 \
    --price-path data/cache/daily_2025.parquet \
    --periods 1 5 10 \
    --quantiles 5
```

### All Factors

```bash
python data_processing/evaluate_factors.py \
    --factor-column all \
    --output data/cache/factor_evaluation/
```

## Interpreting Results

### IC Analysis

**Example output**:
```
Factor: rsi_14
Period  Mean IC  Std IC   t-stat   Positive%
1D      -0.032   0.145    -4.21    42%
5D      -0.028   0.120    -3.50    44%
10D     -0.022   0.095    -2.85    46%
```

**Interpretation**:
- **Negative IC**: Higher RSI predicts *lower* returns (confirms mean reversion)
- **t-stat > 2**: Statistically significant
- **Positive% < 50%**: More days with negative IC (consistent direction)
- **IC decays**: Weaker at longer horizons (short-term signal)

### Quantile Analysis

**Example output**:
```
Factor: rsi_14 (1-day forward returns)
Quantile  Mean Return
Q1        +0.35%      (lowest RSI = oversold)
Q2        +0.15%
Q3        +0.05%
Q4        -0.10%
Q5        -0.25%      (highest RSI = overbought)
Spread    -0.60%
```

**Interpretation**:
- **Q1 outperforms**: Oversold stocks bounce
- **Q5 underperforms**: Overbought stocks fall
- **Monotonic**: Clean relationship across quintiles
- **Negative spread**: High factor = low returns (mean reversion)

### Factor Comparison Table

```
Factor Summary (sorted by |Mean IC|)
=====================================
     factor    mean_ic   t_stat   positive_pct
   momentum      0.045     5.21         0.62
        rsi     -0.032    -3.85         0.42
   bb_pct       -0.028    -3.20         0.45
   volatility    0.018     2.15         0.54
        adx      0.012     1.42         0.51
```

**Use**: Rank factors by absolute IC, focus on those with |t-stat| > 2.

## Forward Period Selection

### Why Multiple Periods?

Different factors predict at different horizons:
- **RSI**: Best at 1-5 days (quick reversals)
- **Momentum**: Best at 5-20 days (trend continuation)
- **Volatility**: Best at 10-30 days (mean reversion)

### For Swing Trading

Focus on **1-day and 5-day IC**:
- 1-day: Next-day prediction (day trading overlap)
- 5-day: Typical swing trade holding period
- 10-day: Maximum swing trade duration

If IC is weak at 1D but strong at 5D, the factor may need time to play out.

## Factor Decay

How IC changes with forecast horizon:

```
Horizon   IC
1D        0.050
5D        0.035
10D       0.020
20D       0.005
```

**Fast decay**: Factor is a short-term signal (good for swing trading)
**Slow decay**: Factor captures longer-term patterns (better for position trading)

## Using Factor Evaluation

### Step 1: Evaluate All Available Factors

```bash
python evaluate_factors.py --factor-column all --output results/
```

### Step 2: Identify Top Factors

Sort by |t-stat| and focus on significant ones (|t| > 2).

### Step 3: Check Quantile Monotonicity

A good factor should show smooth progression from Q1 to Q5.

**Bad pattern** (non-monotonic):
```
Q1: +0.1%
Q2: +0.3%  ← Higher than expected
Q3: +0.1%
Q4: +0.2%  ← Lower than expected
Q5: +0.4%
```

This suggests non-linear relationship — may need feature transformation.

### Step 4: Combine Complementary Factors

If RSI and Momentum both have good IC but low correlation, combining them adds value.

### Step 5: Validate in Backtesting

Factor evaluation shows statistical relationship. Backtesting shows if you can actually trade it profitably (after costs, slippage, etc.).

## API Integration

### Endpoint

```python
@app.get("/api/factor-evaluation")
async def get_factor_evaluation(
    factor: str = "rsi_14",
    periods: str = "1,5,10"
):
    """Get IC and quantile analysis for a factor."""
```

### Frontend Display

The Factor Evaluation panel shows:
- IC time series chart
- Quantile return bar chart
- Factor comparison table

## Common Factor Behaviors

### Mean Reversion Factors

| Factor | Expected IC | Explanation |
|--------|-------------|-------------|
| RSI | Negative | High RSI → overbought → falls |
| BB %B | Negative | Above upper band → reverts |
| Distance from MA | Negative | Extended → pulls back |

### Momentum Factors

| Factor | Expected IC | Explanation |
|--------|-------------|-------------|
| ROC | Positive | Recent winners keep winning |
| Price vs MA | Positive | Above MA → bullish momentum |
| SMA Cross | Positive | Golden cross → uptrend continues |

### Trend Strength Factors

| Factor | Expected IC | Explanation |
|--------|-------------|-------------|
| ADX | Complex | Filter, not directional signal |

## Factor Evaluation Pitfalls

### 1. Look-Ahead Bias

Ensure factor values only use data available at time of calculation.

**Bad**: Using today's close to calculate RSI, then predicting today's close
**Good**: Using yesterday's close for RSI, predicting tomorrow's close

### 2. Survivorship Bias

Only analyzing stocks that exist today misses delisted stocks (often losers).

### 3. Small Sample Size

IC from 20 observations isn't reliable. Need 100+ data points per factor.

### 4. Data Mining / Multiple Testing

Testing 100 factors will find 5 "significant" ones by chance (5% significance level).

**Solution**: Use Bonferroni correction or out-of-sample validation.

### 5. Regime Dependence

A factor that works in bull markets may fail in bear markets.

**Solution**: Test across different market regimes.

## Best Practices

1. **Start with known factors**: RSI, momentum, volatility — they work for a reason
2. **Require significance**: |t-stat| > 2 minimum
3. **Check monotonicity**: Quantiles should progress smoothly
4. **Validate out-of-sample**: Split data, test on held-out period
5. **Consider transaction costs**: High turnover factors may not be tradeable
6. **Combine factors**: Multi-factor models are more robust

---

*Continue to: [Performance Analysis](06-performance-analysis.md)*

# Factor Research Workflow

A systematic approach to discovering and validating new predictive factors for your trading strategy.

## Overview

Factor research is the process of finding features that predict future returns. This workflow helps you:
1. Generate factor ideas
2. Test them rigorously
3. Avoid common pitfalls
4. Integrate successful factors into your strategy

```
Hypothesis Generation
    ↓
Factor Calculation
    ↓
Statistical Evaluation
    ↓
Out-of-Sample Validation
    ↓
Integration & Monitoring
```

## Step 1: Generate Factor Hypotheses

### Sources of Ideas

**Academic Literature**:
- RSI, MACD, Bollinger Bands (classic technical)
- Momentum, mean reversion (behavioral)
- Size, value, quality (Fama-French style)

**Market Observation**:
- "Stocks that gap down often bounce"
- "High volume breakouts continue"
- "RSI divergences precede reversals"

**Quantitative Exploration**:
- "Is there a relationship between ATR and forward returns?"
- "Do stocks with consistent daily ranges outperform?"

### Document Your Hypothesis

Before testing, write down:

```markdown
## Hypothesis: Volume Surge Momentum

**Theory**: Stocks with unusually high volume yesterday tend to continue 
moving in the same direction the next day.

**Calculation**: volume / 20-day average volume

**Expected IC**: Positive (high relative volume → continuation)

**Timeframe**: 1-5 days

**Literature**: Similar to Gervais et al. (2001) volume-return relationship
```

## Step 2: Calculate the Factor

### Add to Feature Engineering

Edit `data_processing/build_technical_features.py`:

```python
def compute_volume_surge(df):
    """
    Relative volume: today's volume / 20-day average.
    High values indicate unusual activity.
    """
    avg_volume = df['volume'].rolling(20).mean()
    return df['volume'] / avg_volume

# Add to feature pipeline
df['volume_surge'] = compute_volume_surge(df)
```

### Rebuild Features

```bash
python data_processing/build_technical_features.py
```

### Verify Calculation

```python
import polars as pl
df = pl.read_parquet('data/cache/technical_features.parquet')
print(df['volume_surge'].describe())
```

Check for:
- Reasonable range (typically 0.5 to 3.0 for relative volume)
- No NaN/infinite values
- Distribution makes sense

## Step 3: Statistical Evaluation

### Run Factor Evaluation

```bash
python data_processing/evaluate_factors.py \
    --factor-column volume_surge \
    --periods 1 5 10 \
    --quantiles 5 \
    --output data/cache/factor_eval/volume_surge.json
```

### Analyze Results

**IC Analysis**:
```
Period  Mean IC  t-stat  Positive%
1D      0.018    2.45    54%
5D      0.012    1.85    52%
10D     0.005    0.82    51%
```

**Interpretation**:
- 1D IC is significant (t > 2)
- Effect decays quickly
- Best for next-day prediction

**Quantile Returns**:
```
Quantile  1D Return
Q1        +0.05%
Q2        +0.08%
Q3        +0.12%
Q4        +0.15%
Q5        +0.25%
Spread    +0.20%
```

**Interpretation**:
- Monotonic increase (good!)
- Q5 outperforms Q1 by 0.20% per day
- Consistent with hypothesis

## Step 4: Out-of-Sample Validation

### Split Data Temporally

```python
# Use first 2/3 for development, last 1/3 for validation
total_days = len(df['date'].unique())
split_date = df['date'].unique().sort()[int(total_days * 0.67)]

in_sample = df.filter(pl.col('date') < split_date)
out_of_sample = df.filter(pl.col('date') >= split_date)
```

### Test on Holdout Data

```bash
# Evaluate only on recent data
python data_processing/evaluate_factors.py \
    --factor-column volume_surge \
    --start-date 2025-07-01 \
    --end-date 2025-12-31
```

### Compare Results

| Metric | In-Sample | Out-of-Sample |
|--------|-----------|---------------|
| Mean IC | 0.022 | 0.015 |
| t-stat | 2.85 | 1.92 |
| Spread | 0.25% | 0.18% |

**Assessment**:
- Some decay expected
- OOS > 50% of in-sample is acceptable
- OOS t-stat near 2 is borderline

## Step 5: Robustness Checks

### Sub-Period Analysis

Test across different market regimes:

```bash
# Bull market period
python evaluate_factors.py --start-date 2024-01-01 --end-date 2024-06-30

# Correction period  
python evaluate_factors.py --start-date 2024-07-01 --end-date 2024-09-30

# Recovery period
python evaluate_factors.py --start-date 2024-10-01 --end-date 2024-12-31
```

### Sector Analysis

Does the factor work across sectors?

```python
# Group by sector (if available)
for sector in df['sector'].unique():
    sector_df = df.filter(pl.col('sector') == sector)
    ic = evaluate_factor(sector_df, 'volume_surge')
    print(f"{sector}: IC = {ic:.3f}")
```

### Turnover Analysis

High turnover = high transaction costs.

```python
# How often do rankings change?
rankings = df.groupby('date').rank('volume_surge')
turnover = (rankings - rankings.shift(1)).abs().mean()
print(f"Daily turnover: {turnover:.1%}")
```

If turnover > 30%, transaction costs may eat profits.

## Step 6: Integration Decision

### Decision Framework

| Criterion | Threshold | Your Factor |
|-----------|-----------|-------------|
| OOS t-stat | > 2.0 | 1.92 ⚠️ |
| OOS IC | > 0.01 | 0.015 ✅ |
| Monotonic quantiles | Yes | Yes ✅ |
| Works across periods | Yes | Mostly ✅ |
| Reasonable turnover | < 30% | 25% ✅ |

### Options

**Strong evidence (all ✅)**: Add to production strategy

**Mixed evidence (some ⚠️)**: 
- Use as secondary signal
- Combine with other factors
- Continue monitoring

**Weak evidence (multiple ❌)**: 
- Reject hypothesis
- Document learnings
- Move to next idea

## Step 7: Production Integration

### Add to Scoring Criteria

If approved, add to the frontend scoring:

```typescript
// In ScoringPanel.tsx or via presets
{
  name: 'volume_surge',
  weight: 1.5,
  min_value: 0.5,
  max_value: 3.0,
  invert: false,  // High volume = good
}
```

### Monitor Performance

Track the factor's live performance:

```python
# Weekly review
ic_this_week = compute_ic(factor, returns, last_5_days)
print(f"Volume surge IC this week: {ic_this_week:.3f}")
```

### Set Decay Thresholds

If rolling IC drops below threshold, investigate:

```python
rolling_ic = compute_rolling_ic(factor, returns, window=60)

if rolling_ic[-1] < 0.005:
    print("WARNING: Factor may be decaying")
```

## Common Pitfalls

### 1. Data Mining Bias

Testing 100 factors will find 5 "significant" ones by chance.

**Solution**: 
- Pre-specify hypotheses
- Use Bonferroni correction
- Require economic rationale

### 2. Lookahead Bias

Using future information in factor calculation.

**Solution**:
- Lag all inputs appropriately
- Use point-in-time data
- Review calculation carefully

### 3. Survivorship Bias

Only testing stocks that still exist.

**Solution**:
- Include delisted stocks in data
- Note any survivorship issues

### 4. Overfitting

Factor tuned to historical quirks.

**Solution**:
- Simple factors
- Out-of-sample validation
- Avoid parameter optimization

### 5. Transaction Cost Ignorance

Factor works on paper but not after costs.

**Solution**:
- Calculate turnover
- Model realistic costs
- Backtest with VectorBT

## Factor Research Log

Keep a research log:

```markdown
# Factor Research Log

## 2026-01-18: Volume Surge

**Hypothesis**: High relative volume predicts continuation
**Result**: Marginal—IC=0.015 OOS, t-stat=1.92
**Decision**: Add as secondary factor with low weight
**Next**: Test volume surge + price direction interaction

## 2026-01-15: ATR Reversion

**Hypothesis**: Stocks with high ATR revert to mean volatility
**Result**: Rejected—no significant IC
**Learning**: ATR is better for position sizing than prediction
**Next**: Try ATR percentile instead of raw ATR
```

## Ideas to Test

### Momentum Variants
- 5-day return vs 20-day return
- Price vs 52-week high
- Industry-relative momentum

### Mean Reversion Variants
- RSI with different periods (7, 21, 28)
- Distance from moving average
- Bollinger Band width before expansion

### Volume Variants
- Volume trend (rising vs falling)
- Volume vs price correlation
- On-balance volume

### Volatility Variants
- ATR percentile
- Realized vs implied volatility
- Volatility breakout (low vol → high vol)

---

*Continue to: [Strategy Development Workflow](03-strategy-development.md)*

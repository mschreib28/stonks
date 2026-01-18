# Stock Selection Criteria

This guide details all available metrics for screening stocks and how to use them effectively for short-term trading.

## Overview

Stonks provides multiple metrics to filter and rank stocks. The right combination depends on your strategy, capital, and risk tolerance.

## Metric Categories

### 1. Trend Metrics

Measure overall price direction over time.

#### Price Trend % (trend_pct)
**What**: Percentage price change over the analysis period (6-12 months)
**Use for**: Finding stocks in uptrends or downtrends
**Swing trading**: Neutral - short-term trades can profit in any direction
**Typical range**: -50% to +100%

#### Total Return % (total_return_pct)
**What**: Total return including any dividends (mostly same as trend for small-caps)
**Use for**: Comparing overall performance
**Swing trading**: Neutral - doesn't affect short-term trades
**Typical range**: -50% to +100%

### 2. Volatility Metrics

Measure how much price moves.

#### Avg Daily Swing % (avg_daily_swing_pct)
**What**: Average (High - Low) / Open as percentage
**Use for**: Finding volatile stocks
**Swing trading**: **Critical** - this is your opportunity
**Target**: 2-8% for small-caps
**Example**: A $5 stock with 4% swing moves $0.20/day on average

#### Large Swing Frequency (large_swing_frequency)
**What**: Percentage of days with 20%+ swings
**Use for**: Avoiding extremely volatile stocks
**Swing trading**: High values = higher risk
**Target**: < 10% for consistent trading

#### Volatility (volatility)
**What**: Standard deviation of daily returns (annualized)
**Use for**: Risk assessment
**Swing trading**: Higher = more opportunity but more risk
**Target**: 30-80% annualized for active trading

### 3. Volume Metrics

Measure trading activity and liquidity.

#### Average Volume (avg_volume)
**What**: Average shares traded per day
**Use for**: Ensuring you can enter/exit
**Swing trading**: **Critical** for execution quality
**Target**: > 500,000 shares/day minimum

#### Liquidity Multiple (liquidity_multiple)
**What**: avg_volume / your_position_size (assumes 10k shares)
**Use for**: Understanding your market impact
**Swing trading**: **Essential** - high values mean clean execution
**Target**: > 10x preferred, > 50x ideal

**Example**:
- Stock trades 1M shares/day
- Your position: 10k shares
- Liquidity multiple: 100x (excellent)

#### Position % of Volume (position_pct_of_volume)
**What**: Your position as percentage of daily volume (inverse of liquidity multiple)
**Use for**: Same as liquidity multiple, different perspective
**Swing trading**: Lower is better
**Target**: < 5% of daily volume

### 4. Daily Range Metrics (Dollar Terms)

These are **the most important** for swing trading profit potential.

#### Avg Daily Range ($) (avg_daily_range_dollars)
**What**: Average (High - Low) in dollars
**Use for**: Direct profit potential per trade
**Swing trading**: **Primary metric** for opportunity sizing
**Target**: $0.20-$0.80 for small-caps

**Calculation**:
```
If you capture 50% of daily range:
10,000 shares × $0.30 × 50% = $1,500 profit
```

#### Median Daily Range ($) (median_daily_range_dollars)
**What**: Median (High - Low) in dollars, less affected by outliers
**Use for**: More robust estimate of typical day
**Swing trading**: Better than average for setting expectations
**Target**: Similar to average, but check for divergence

#### Sweet Spot Days % (sweet_spot_range_pct)
**What**: Percentage of days with $0.20-$0.60 range
**Use for**: Finding stocks with consistent, tradeable ranges
**Swing trading**: **Highly valuable** - ideal profit zone
**Target**: > 40% is excellent

**Why this range?**
- Below $0.20: Not worth the effort and commission
- Above $0.60: Often indicates news events or high risk
- $0.20-$0.60: Sweet spot for consistent profits

#### Range Consistency (CV) (daily_range_cv)
**What**: Coefficient of variation = std(range) / mean(range)
**Use for**: Finding predictable stocks
**Swing trading**: Lower is better - predictable = easier to trade
**Target**: < 0.5 is consistent, < 0.3 is very consistent

**Why it matters**:
- Low CV: Range is similar most days, easy to set stops
- High CV: Range varies wildly, hard to position size

### 5. Combined Scores

Pre-calculated combinations of metrics.

#### Tradability Score (tradability_score)
**What**: 0-100 score combining liquidity, range, and consistency
**Use for**: Quick overall assessment
**Swing trading**: **Primary screening metric**
**Target**: > 70 is excellent, > 50 is tradeable

**Components**:
- Liquidity (weighted)
- Daily range in ideal zone (weighted)
- Consistency/predictability (weighted)

#### Profit Potential Score (profit_potential_score)
**What**: daily_range × liquidity (theoretical max opportunity)
**Use for**: Ranking by opportunity size
**Swing trading**: High values = more room to profit
**Target**: Higher is generally better

### 6. Price Metrics

Filter by absolute price level.

#### Price Range % (price_range_pct)
**What**: (Max price - Min price) / Min price over period
**Use for**: Understanding price volatility over time
**Swing trading**: Wide range means trending or volatile

#### Current Price (current_price)
**What**: Most recent closing price
**Use for**: **Filtering** to your target price range
**Swing trading**: Essential filter for capital allocation
**Target for small-cap focus**: $1-$8

## Building Effective Criteria Sets

### For Day Trading Small-Caps

```
current_price:       $1-$8 (filter)
avg_daily_range:     $0.20-$1.00 (high weight)
liquidity_multiple:  > 20 (high weight)
daily_range_cv:      < 0.5 (medium weight, inverted)
tradability_score:   > 50 (as confirmation)
```

### For Momentum Swing Trading

```
current_price:       $2-$15 (filter)
trend_pct:           > 10% (uptrend bias)
avg_daily_swing_pct: > 3% (weight)
avg_volume:          > 1M (filter)
volatility:          40-80% (weight)
```

### For Mean Reversion

```
current_price:       $1-$10 (filter)
daily_range_cv:      < 0.4 (high weight, inverted)
sweet_spot_range_pct:> 40% (high weight)
avg_daily_range:     $0.15-$0.50 (medium weight)
liquidity_multiple:  > 30 (medium weight)
```

## How Weights Work

Weights determine how much each criterion affects the final score:

```
Final Score = Σ (normalized_value × weight) / Σ weights
```

**Example**:
- Tradability (weight 3.0): Stock scores 80/100 → contributes 240
- Range (weight 2.5): Stock scores 70/100 → contributes 175
- Liquidity (weight 2.0): Stock scores 90/100 → contributes 180
- Total: (240 + 175 + 180) / 7.5 = **79.3**

### Weight Guidelines

| Weight | Meaning |
|--------|---------|
| 1.0 | Nice to have |
| 2.0 | Important |
| 3.0 | Very important |
| 5.0+ | Critical / Must have |
| 10.0 | Essentially a hard filter |

## The Invert Option

For metrics where **lower is better**, use invert:

| Metric | Invert? | Reason |
|--------|---------|--------|
| daily_range_cv | Yes | Low CV = consistent = better |
| position_pct_of_volume | Yes | Low = less market impact |
| large_swing_frequency | Yes | Low = less extreme risk |

When inverted:
```
normalized_value = 100 - original_normalized_value
```

## Min/Max Value Ranges

Use these to:
1. **Filter out** stocks outside acceptable range
2. **Normalize** values for scoring

**Example for avg_daily_range_dollars**:
- Min: $0.10 (below this isn't worth trading)
- Max: $1.00 (above this may be too volatile)
- Stock with $0.40 range → normalized to 50/100

**Setting ranges**:
- Too narrow: Few stocks pass
- Too wide: Doesn't differentiate well
- Just right: Meaningful ranking with good coverage

## Practical Tips

### Start with Defaults
The Swing Trading Default preset is battle-tested. Modify incrementally.

### Watch for Correlations
Many metrics are correlated:
- Price and daily range (higher price often = higher range)
- Volume and liquidity (by definition)
- Volatility and range CV

Don't over-weight correlated metrics.

### Validate with Backtest
After screening, backtest your top picks:
1. Do they actually perform well?
2. Are the signals tradeable?
3. What's the drawdown?

### Seasonal Adjustments
Market conditions change:
- High VIX periods: Tighten volatility filters
- Low VIX periods: May need to accept lower ranges
- Earnings season: Exclude or note catalyst dates

---

*Continue to: [Risk Management](03-risk-management.md)*

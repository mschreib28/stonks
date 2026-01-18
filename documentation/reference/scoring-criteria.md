# Scoring Criteria Reference

Complete reference for all metrics available in the Stonks scoring system.

## Quick Reference Table

| Metric | Type | Range | Higher = Better | Good Target |
|--------|------|-------|-----------------|-------------|
| `tradability_score` | Score | 0-100 | Yes | > 70 |
| `profit_potential_score` | Score | 0-∞ | Yes | Higher |
| `avg_daily_range_dollars` | Range | $0-$∞ | Depends | $0.20-$0.80 |
| `median_daily_range_dollars` | Range | $0-$∞ | Depends | $0.20-$0.80 |
| `sweet_spot_range_pct` | Range | 0-100% | Yes | > 40% |
| `daily_range_cv` | Range | 0-∞ | No (invert) | < 0.5 |
| `avg_volume` | Volume | 0-∞ | Yes | > 500k |
| `liquidity_multiple` | Volume | 0-∞ | Yes | > 20x |
| `position_pct_of_volume` | Volume | 0-100% | No (invert) | < 5% |
| `trend_pct` | Trend | -100-∞% | Depends | Strategy |
| `total_return_pct` | Trend | -100-∞% | Depends | Strategy |
| `avg_daily_swing_pct` | Volatility | 0-100% | Depends | 2-8% |
| `large_swing_frequency` | Volatility | 0-100% | No (invert) | < 10% |
| `volatility` | Volatility | 0-∞% | Depends | 30-60% |
| `price_range_pct` | Price | 0-∞% | Depends | Context |
| `current_price` | Price | $0-$∞ | Filter | $1-$8 |

---

## Combined Scores

### tradability_score

**Definition**: Composite score (0-100) measuring overall suitability for swing trading.

**Components**:
- Liquidity (can you trade 10k shares easily?)
- Daily range (is there enough movement?)
- Consistency (is the movement predictable?)

**Calculation**:
```python
tradability_score = weighted_combination(
    liquidity_component,   # Weight: High
    range_component,       # Weight: Medium
    consistency_component  # Weight: Medium
)
```

**Usage**:
```typescript
{
  name: 'tradability_score',
  weight: 3.0,
  min_value: 0,
  max_value: 100,
  invert: false,
}
```

**Interpretation**:
| Score | Quality |
|-------|---------|
| 80-100 | Excellent trading candidate |
| 60-80 | Good candidate |
| 40-60 | Fair, check individual metrics |
| < 40 | Poor, likely liquidity or range issues |

---

### profit_potential_score

**Definition**: Theoretical maximum profit opportunity based on range × liquidity.

**Calculation**:
```python
profit_potential_score = avg_daily_range_dollars * liquidity_multiple
```

**Usage**:
```typescript
{
  name: 'profit_potential_score',
  weight: 1.5,
  min_value: 0,
  max_value: 50,
  invert: false,
}
```

**Interpretation**:
Higher = more room to profit. But high values may come from either:
- Good: High liquidity AND good range
- Misleading: Extremely high liquidity but tiny range

Always verify underlying components.

---

## Daily Range Metrics (Dollar Terms)

### avg_daily_range_dollars

**Definition**: Average (High - Low) in dollar terms over the analysis period.

**Calculation**:
```python
daily_range = high - low
avg_daily_range_dollars = daily_range.mean()
```

**Why it matters**: This is your profit potential per day. A stock with $0.40 average range offers $400 profit potential per 1,000 shares.

**Usage**:
```typescript
{
  name: 'avg_daily_range_dollars',
  weight: 2.5,
  min_value: 0.10,
  max_value: 1.00,
  invert: false,
}
```

**Targets by price**:
| Price Range | Target Daily Range |
|-------------|-------------------|
| $1-$3 | $0.10-$0.30 |
| $3-$5 | $0.20-$0.50 |
| $5-$8 | $0.30-$0.80 |

---

### median_daily_range_dollars

**Definition**: Median daily range, less affected by outlier days.

**When to prefer over average**:
- Stock has occasional huge moves (news days)
- Want more conservative estimate of typical day
- Comparing stocks with different outlier frequencies

---

### sweet_spot_range_pct

**Definition**: Percentage of days with daily range between $0.20 and $0.60.

**Why this range**:
- < $0.20: Not worth the effort after commissions
- $0.20-$0.60: Ideal for consistent swing trading profits
- > $0.60: Often indicates news/events, higher risk

**Usage**:
```typescript
{
  name: 'sweet_spot_range_pct',
  weight: 1.5,
  min_value: 0,
  max_value: 100,
  invert: false,
}
```

**Targets**:
| Sweet Spot % | Quality |
|--------------|---------|
| > 50% | Excellent |
| 30-50% | Good |
| 15-30% | Fair |
| < 15% | Poor |

---

### daily_range_cv

**Definition**: Coefficient of variation of daily range = std(range) / mean(range).

**Why it matters**: 
- Low CV = consistent daily ranges = easier to set stops and targets
- High CV = unpredictable ranges = harder to position size

**Usage** (note: invert):
```typescript
{
  name: 'daily_range_cv',
  weight: 1.0,
  min_value: 0,
  max_value: 1.0,
  invert: true,  // Lower is better
}
```

**Targets**:
| CV | Consistency |
|----|-------------|
| < 0.3 | Very consistent |
| 0.3-0.5 | Consistent |
| 0.5-0.7 | Moderate |
| > 0.7 | Unpredictable |

---

## Volume Metrics

### avg_volume

**Definition**: Average daily trading volume in shares.

**Why it matters**: Base liquidity check. Need enough volume to enter/exit positions.

**Usage**:
```typescript
{
  name: 'avg_volume',
  weight: 1.0,
  min_value: 100000,
  max_value: 10000000,
  invert: false,
}
```

**Minimums by position size**:
| Position Size | Min Volume |
|---------------|------------|
| 1,000 shares | 100,000 |
| 5,000 shares | 250,000 |
| 10,000 shares | 500,000 |

---

### liquidity_multiple

**Definition**: avg_volume / 10,000 shares (standard position size).

**Why it matters**: Tells you how many times your position fits into daily volume.

**Calculation**:
```python
liquidity_multiple = avg_volume / 10000
```

**Usage**:
```typescript
{
  name: 'liquidity_multiple',
  weight: 2.0,
  min_value: 0,
  max_value: 100,
  invert: false,
}
```

**Interpretation**:
| Multiple | Meaning |
|----------|---------|
| > 50x | Excellent, no market impact |
| 20-50x | Good, minimal impact |
| 10-20x | Acceptable, slight impact |
| 5-10x | Marginal, noticeable impact |
| < 5x | Poor, significant impact |

---

### position_pct_of_volume

**Definition**: 10,000 shares / avg_volume * 100.

**Why it matters**: Same as liquidity_multiple, inverted perspective. What percentage of daily volume is your position?

**Usage** (note: invert):
```typescript
{
  name: 'position_pct_of_volume',
  weight: 1.0,
  min_value: 0,
  max_value: 20,
  invert: true,  // Lower is better
}
```

**Targets**:
| % of Volume | Impact |
|-------------|--------|
| < 1% | Negligible |
| 1-5% | Minimal |
| 5-10% | Noticeable |
| > 10% | Significant |

---

## Trend Metrics

### trend_pct

**Definition**: Price change percentage over the analysis period.

**Calculation**:
```python
trend_pct = (end_price - start_price) / start_price * 100
```

**Usage**:
- For momentum: Higher is better
- For mean reversion: May want stocks that are down
- As filter: Avoid extreme values (>100% or <-50%)

---

### total_return_pct

**Definition**: Total return including dividends (mostly same as trend_pct for small-caps).

---

## Volatility Metrics

### avg_daily_swing_pct

**Definition**: Average (High - Low) / Open as percentage.

**Calculation**:
```python
daily_swing_pct = (high - low) / open * 100
avg_daily_swing_pct = daily_swing_pct.mean()
```

**Why it matters**: Percentage view of daily movement.

**Targets**:
| Swing % | Character |
|---------|-----------|
| 1-3% | Low volatility |
| 3-5% | Moderate |
| 5-8% | High volatility |
| > 8% | Very volatile |

---

### large_swing_frequency

**Definition**: Percentage of days with 20%+ price swing.

**Why it matters**: Measures extreme volatility risk. High values indicate potential for catastrophic moves.

**Usage** (note: invert):
```typescript
{
  name: 'large_swing_frequency',
  weight: 0.5,
  min_value: 0,
  max_value: 20,
  invert: true,  // Lower is better
}
```

**Targets**: Generally want < 5% of days with extreme moves.

---

### volatility

**Definition**: Annualized standard deviation of daily returns.

**Calculation**:
```python
volatility = daily_returns.std() * sqrt(252) * 100
```

**Usage**: Depends on strategy. Higher volatility = more opportunity but more risk.

---

## Price Metrics

### current_price

**Definition**: Most recent closing price.

**Primary use**: Filtering to target price range.

**Usage** (as filter):
```typescript
{
  name: 'current_price',
  weight: 10.0,  // High weight to act as filter
  min_value: 0,
  max_value: 8,
  invert: false,
}
```

**Common ranges**:
| Strategy | Price Range |
|----------|-------------|
| Penny stocks | $0.01-$1 |
| Small-cap swing | $1-$8 |
| Mid-cap swing | $5-$25 |
| Large-cap | $25+ |

---

### price_range_pct

**Definition**: (Max price - Min price) / Min price over period.

**Use**: Understanding price volatility over time. Wide range = trending or volatile.

---

## Building Custom Criteria

### Example: Conservative Small-Cap

Focus on consistency and liquidity:

```typescript
[
  { name: 'tradability_score', weight: 4.0, min: 0, max: 100, invert: false },
  { name: 'daily_range_cv', weight: 3.0, min: 0, max: 0.5, invert: true },
  { name: 'liquidity_multiple', weight: 3.0, min: 30, max: 200, invert: false },
  { name: 'sweet_spot_range_pct', weight: 2.0, min: 30, max: 100, invert: false },
  { name: 'current_price', weight: 10.0, min: 2, max: 8, invert: false },
]
```

### Example: Aggressive Momentum

Focus on movement and recent performance:

```typescript
[
  { name: 'avg_daily_range_dollars', weight: 3.0, min: 0.20, max: 2.00, invert: false },
  { name: 'avg_daily_swing_pct', weight: 2.5, min: 3, max: 10, invert: false },
  { name: 'trend_pct', weight: 2.0, min: 10, max: 100, invert: false },
  { name: 'liquidity_multiple', weight: 1.5, min: 10, max: 100, invert: false },
  { name: 'current_price', weight: 10.0, min: 1, max: 15, invert: false },
]
```

### Example: Mean Reversion Focus

For stocks likely to bounce:

```typescript
[
  { name: 'tradability_score', weight: 2.0, min: 50, max: 100, invert: false },
  { name: 'trend_pct', weight: 2.5, min: -30, max: 0, invert: true },  // Want down stocks
  { name: 'daily_range_cv', weight: 2.0, min: 0, max: 0.4, invert: true },
  { name: 'liquidity_multiple', weight: 2.0, min: 20, max: 150, invert: false },
  { name: 'current_price', weight: 10.0, min: 1, max: 10, invert: false },
]
```

---

## API Format

### Request

```json
POST /api/score
{
  "criteria": [
    {
      "name": "tradability_score",
      "weight": 3.0,
      "min_value": 0,
      "max_value": 100,
      "invert": false
    }
  ],
  "months_back": 6,
  "min_days": 20,
  "min_avg_volume": 100000,
  "min_price": 1,
  "max_price": 8
}
```

### Response

```json
{
  "results": [
    {
      "ticker": "XYZ",
      "score": 85.2,
      "tradability_score": 78,
      "avg_daily_range_dollars": 0.42,
      "liquidity_multiple": 45,
      ...
    }
  ]
}
```

---

*Continue to: [CLI Commands Reference](cli-commands.md)*

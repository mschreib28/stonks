# Risk Management

Effective risk management is the difference between long-term success and blowing up your account. This guide covers position sizing, stop-loss strategies, and capital preservation.

## The Foundation: Risk Per Trade

**Rule**: Never risk more than 1-2% of your account on any single trade.

### Why This Matters

| Risk per Trade | Consecutive Losses to Halve Account |
|----------------|-------------------------------------|
| 10% | 7 losses |
| 5% | 14 losses |
| 2% | 35 losses |
| 1% | 69 losses |

With 1% risk, you can survive long losing streaks and stay in the game.

### Calculating Risk

```
Risk Amount = Account Size × Risk Percentage

Example:
$100,000 account × 1% = $1,000 max risk per trade
```

## Position Sizing

### The Formula

```
Position Size = Risk Amount / (Entry Price - Stop Price)

Example:
Risk Amount: $1,000
Entry Price: $5.00
Stop Price: $4.80 (4% below entry)
Stop Distance: $0.20

Position Size = $1,000 / $0.20 = 5,000 shares
Position Value = 5,000 × $5.00 = $25,000
```

### Position Size Limits

Even with correct risk calculation, add these limits:

| Limit | Typical Value | Purpose |
|-------|---------------|---------|
| Max % of account | 20-25% | Diversification |
| Max % of daily volume | 5-10% | Liquidity |
| Max shares | Varies | Practical execution |

### Example: Full Position Sizing

```
Account: $100,000
Risk: 1% = $1,000

Stock: XYZ at $4.50
Stop: $4.30 (4.4% risk)
Daily Volume: 2,000,000 shares

Calculation:
1. Risk-based: $1,000 / $0.20 = 5,000 shares
2. Account-based: $100,000 × 20% / $4.50 = 4,444 shares
3. Volume-based: 2,000,000 × 5% = 100,000 shares

Take minimum: 4,444 shares (account limit)
Actual risk: 4,444 × $0.20 = $889 (0.89% of account)
```

## Stop-Loss Strategies

### Types of Stops

#### 1. Fixed Percentage Stop
```
Stop = Entry × (1 - stop_percentage)
Example: $5.00 × (1 - 0.05) = $4.75
```
**Pros**: Simple, consistent
**Cons**: Doesn't adapt to volatility

#### 2. ATR-Based Stop
```
Stop = Entry - (ATR × multiplier)
Example: $5.00 - ($0.25 × 2) = $4.50
```
**Pros**: Adapts to volatility
**Cons**: Requires ATR calculation

#### 3. Support/Resistance Stop
```
Stop = Below significant support level
Example: Support at $4.40, stop at $4.35
```
**Pros**: Technically sound
**Cons**: Requires chart analysis

#### 4. Time-Based Stop
```
Exit if target not hit within X days
Example: Exit after 5 days regardless
```
**Pros**: Frees capital
**Cons**: May miss eventual move

### Recommended: ATR-Based Stops

For Stonks swing trading:

```python
stop_distance = ATR_14 * 2.0
stop_price = entry_price - stop_distance
position_size = risk_amount / stop_distance
```

Why 2× ATR?
- 1× ATR: Too tight, stopped out by noise
- 2× ATR: Room to breathe, catches real reversals
- 3× ATR: Very wide, slow to exit losers

## Profit Targets

### Fixed Risk:Reward Ratios

| Style | Target | Risk:Reward |
|-------|--------|-------------|
| Scalping | 1× ATR | 1:1 |
| Day Trading | 1.5-2× ATR | 1:1.5-2 |
| Swing Trading | 2-3× ATR | 1:2-3 |

### Trail Stops for Runners

When a trade moves in your favor:

```
Initial: 2× ATR stop
After +1 ATR profit: Move stop to breakeven
After +2 ATR profit: Trail at 1.5× ATR
After +3 ATR profit: Trail at 1× ATR
```

This locks in profits while allowing winners to run.

## Portfolio-Level Risk

### Maximum Exposure

| Metric | Conservative | Moderate | Aggressive |
|--------|--------------|----------|------------|
| Total market exposure | 50-70% | 70-90% | 90-100% |
| Single sector | 20% | 30% | 40% |
| Correlated positions | 3-4 | 5-6 | 7-8 |

### Correlation Awareness

Don't have 5 similar small-cap tech stocks:
- They'll move together
- Your "5 positions" is really 1 big bet

**Diversify by**:
- Sector (tech, healthcare, energy)
- Market cap (if not focused)
- Strategy (mean reversion vs momentum)

## Risk Metrics to Monitor

### Daily

| Metric | Action Threshold |
|--------|------------------|
| Day P&L | -2% = reduce exposure |
| Open positions | > 10 = review |
| Cash available | < 30% = limit new trades |

### Weekly

| Metric | Action Threshold |
|--------|------------------|
| Week P&L | -5% = review strategy |
| Win rate | < 40% = pause and analyze |
| Avg win/loss ratio | < 1.5 = tighten stops |

### Monthly

| Metric | Target | Concern |
|--------|--------|---------|
| Sharpe Ratio | > 1.0 | < 0.5 |
| Max Drawdown | < 10% | > 20% |
| Profit Factor | > 1.5 | < 1.2 |

## Drawdown Management

### What is Drawdown?

```
Drawdown = (Peak Value - Current Value) / Peak Value

Example:
Peak: $120,000
Current: $100,000
Drawdown: ($120,000 - $100,000) / $120,000 = 16.7%
```

### Drawdown Rules

| Drawdown Level | Action |
|----------------|--------|
| 5% | Review recent trades |
| 10% | Reduce position sizes by 50% |
| 15% | Pause new entries, review strategy |
| 20% | Stop trading, full analysis |

### Recovery Math

The deeper the hole, the harder to climb out:

| Drawdown | Return Needed to Recover |
|----------|--------------------------|
| 10% | 11.1% |
| 20% | 25.0% |
| 30% | 42.9% |
| 50% | 100.0% |

**Prevention > Recovery**

## Gap Risk

Small-cap stocks can gap significantly overnight.

### Managing Gap Risk

1. **Position size for gaps**: Assume 10-20% gap against you
2. **Avoid over earnings**: Check calendar before holding overnight
3. **Sector awareness**: Sector news can gap multiple stocks
4. **Diversification**: Spread risk across multiple positions

### Example: Gap-Adjusted Position Size

```
Account: $100,000
Max acceptable gap loss: 3%
Potential gap: 20%

Max position value = $100,000 × 3% / 20% = $15,000
At $5/share = 3,000 shares max
```

## Psychological Risk Management

### Trading Rules

1. **No revenge trading**: Accept losses, move on
2. **No FOMO entries**: Missed it? There's always tomorrow
3. **Stop after 3 losers**: Take a break, clear your head
4. **Daily loss limit**: -2% = done for the day
5. **Keep a journal**: Track decisions, not just results

### Warning Signs

| Sign | Action |
|------|--------|
| Increasing position sizes after losses | Stop, reset |
| Ignoring stop-losses | Stop, review rules |
| Trading when tired/emotional | Stop, rest |
| Checking P&L constantly | Set alerts instead |

## Risk Management Checklist

Before every trade:

- [ ] Risk is ≤ 1-2% of account
- [ ] Position ≤ 20% of account
- [ ] Position ≤ 5% of daily volume
- [ ] Stop-loss is defined and entered
- [ ] No earnings/catalyst in holding period
- [ ] Not correlated with existing positions
- [ ] Fits within daily exposure limit

## Using Stonks for Risk Management

### Liquidity Screening
Use `liquidity_multiple` to ensure you can exit:
- Target: > 10x your position size
- Filter out illiquid stocks before analysis

### Volatility Assessment
Use `daily_range_cv` to predict stop distances:
- Low CV: Can use tighter stops
- High CV: Need wider stops, smaller positions

### Performance Analysis
Use the Performance Panel to monitor:
- Rolling Sharpe (degrading?)
- Drawdown (approaching limits?)
- Win rate (sustainable?)

---

*Continue to: [Feature Engineering](../tier1-features/01-feature-engineering.md)*

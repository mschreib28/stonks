# Swing Trading Philosophy

This document explains the trading philosophy behind Stonks and why each component matters for short-term profits.

## What is Swing Trading?

Swing trading captures short-term price movements over 1-10 days. Unlike day trading (intraday) or investing (months/years), swing trading:

- **Holds overnight**: Captures gap moves and multi-day trends
- **Uses technical analysis**: Relies on price patterns and indicators
- **Manages risk actively**: Uses stop-losses and position sizing
- **Requires patience**: Waits for high-probability setups

## Why Small to Mid-Cap Stocks?

The Stonks default preset focuses on stocks priced $1-$8. Here's why:

### Advantages

| Factor | Benefit |
|--------|---------|
| **Higher volatility** | More profit potential per trade |
| **Less institutional coverage** | Inefficiencies persist longer |
| **Technical patterns work** | Less HFT interference |
| **Position sizing** | 10k shares = manageable capital |
| **Daily ranges** | $0.20-$0.60 moves = real profits |

### The Math

For a stock at $5 with a typical $0.40 daily range:
- 10,000 shares × $0.20 profit = **$2,000 per trade**
- Even capturing 50% of the range = **$1,000**
- Risk 1% of a $100k account = **$1,000 max loss**
- Risk/reward often 1:2 or better

### Risks to Understand

| Risk | Mitigation |
|------|------------|
| **Higher spreads** | Only trade liquid stocks (Liquidity Multiple filter) |
| **News volatility** | Avoid earnings/catalyst dates |
| **Manipulation** | Focus on established small-caps, not penny stocks |
| **Gaps** | Position size accounts for gap risk |

## The Stonks Approach

### 1. Systematic Screening

Don't hunt randomly. Use data-driven screening to find:
- Stocks with consistent daily ranges
- Adequate liquidity for your position size
- Technical setups aligned with your strategy

### 2. Factor-Based Selection

The project evaluates which indicators actually predict returns:
- **Information Coefficient**: Measures predictive power
- **Quantile analysis**: Do high-signal stocks outperform?
- **Statistical significance**: Is the edge real?

### 3. Backtested Strategies

Before trading real money:
- Test on historical data
- Understand expected drawdowns
- Know your win rate and profit factor

### 4. Risk Management First

Every trade has a plan:
- Entry price
- Stop-loss level
- Profit target
- Position size based on risk

## The Default Criteria Explained

The Swing Trading Default preset uses these criteria:

### Tradability Score (Weight: 3.0)
**What it is**: A composite score (0-100) combining liquidity, range, and consistency.

**Why it matters**: Saves time by combining multiple factors into one metric. A high tradability score means the stock is suitable for swing trading on multiple dimensions.

**Target**: Higher is better. Stocks above 70 are excellent candidates.

### Average Daily Range in Dollars (Weight: 2.5)
**What it is**: The average high-low range in dollar terms.

**Why it matters**: This is your profit potential. A stock that moves $0.30/day offers more opportunity than one moving $0.05/day.

**Target**: $0.10-$1.00. Below $0.10 isn't worth the effort. Above $1.00 may be too volatile.

### Liquidity Multiple (Weight: 2.0)
**What it is**: How many times your position size (10k shares) fits into average daily volume.

**Why it matters**: If your position is 50% of daily volume, you'll move the price entering and exiting. You want to be a small fish in a big pond.

**Target**: Higher is better. 10x+ means minimal market impact.

### Sweet Spot Days % (Weight: 1.5)
**What it is**: Percentage of days with $0.20-$0.60 range.

**Why it matters**: This is the ideal range for profitable swing trades. Too small = not worth it. Too large = higher risk.

**Target**: Higher is better. 40%+ of days in sweet spot is excellent.

### Range Consistency (CV) (Weight: 1.0, Inverted)
**What it is**: Coefficient of variation of daily range. Lower = more consistent.

**Why it matters**: Predictable stocks are easier to trade. If range varies wildly (CV > 1), position sizing and stop placement become difficult.

**Target**: Lower is better (inverted in scoring). CV < 0.5 is very consistent.

### Current Price (Weight: 10.0)
**What it is**: The most recent closing price.

**Why it matters**: This is actually used as a **filter** more than a scorer. The high weight with max value of $8 strongly penalizes stocks outside the target range.

**Target**: $0-$8 for small-cap focus.

## Edge Sources in Swing Trading

Where does profit come from? Understanding your edge:

### 1. Mean Reversion
Stocks that move too far too fast tend to revert:
- RSI oversold → bounce
- Price at lower Bollinger Band → rebound
- Extended from moving average → pullback

### 2. Trend Continuation
Strong trends persist:
- MACD crossover → momentum continuation
- ADX strong → trend likely to continue
- Above key moving averages → bullish

### 3. Volatility Patterns
Volatility clusters and expands:
- Bollinger Band squeeze → breakout coming
- ATR compression → expansion imminent
- Low volatility → opportunity

### 4. Information Asymmetry
Small-caps are less efficient:
- Analyst coverage is thin
- News travels slowly
- Technical patterns remain valid longer

## Psychology of Swing Trading

### What Works
- **Systematic approach**: Follow your rules
- **Patience**: Wait for A+ setups
- **Discipline**: Take stops, take profits
- **Record keeping**: Track what works

### What Fails
- **Chasing**: Entering after the move
- **Revenge trading**: Trying to make back losses
- **Overtrading**: Taking mediocre setups
- **Moving stops**: Letting losers run

## Putting It Together

The Stonks workflow for swing trading:

```
Morning: Screen for candidates (Scoring Panel)
    ↓
Analysis: Check charts, factors, recent performance
    ↓
Planning: Define entry, stop, target for best setups
    ↓
Execution: Enter trades meeting all criteria
    ↓
Management: Monitor, adjust stops, take profits
    ↓
Review: Log results, update factor research
```

## Expected Performance

Realistic expectations for disciplined swing trading:

| Metric | Conservative | Optimistic |
|--------|--------------|------------|
| Win Rate | 45-55% | 55-65% |
| Avg Win : Avg Loss | 1.5:1 | 2:1 |
| Monthly Return | 3-5% | 8-12% |
| Max Drawdown | 10-15% | 5-10% |
| Sharpe Ratio | 1.0-1.5 | 1.5-2.5 |

**Note**: These assume proper risk management and systematic execution.

---

*Continue to: [Stock Selection Criteria](02-stock-selection-criteria.md)*

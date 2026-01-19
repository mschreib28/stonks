# Daily Screening Workflow

A step-by-step guide for finding tradeable stocks each morning using Stonks.

## Overview

This workflow takes ~10-15 minutes and produces a ranked list of swing trading candidates.

```
Update Data (2 min)
    ↓
Run Screening (1 min)
    ↓
Review Top Picks (5 min)
    ↓
Detailed Analysis (5 min)
    ↓
Trading Plan
```

## Step 1: Update Market Data

Before market open, refresh your data.

### Command Line

**Option 1: Automated Incremental Update (Recommended)**
```bash
# Automatically processes only new data (checks dates and skips existing)
uv run python data_processing/process_all_data.py
```

**Option 2: Manual Update**
```bash
# Update daily cache (processes all files in input directory)
uv run python data_processing/build_polygon_cache.py --add-returns

# Rebuild technical features
uv run python data_processing/build_technical_features.py
```

### Verify Data Freshness

Check that data includes the most recent trading day:
```bash
uv run python -c "
import polars as pl
df = pl.read_parquet('data/cache/daily_2025.parquet')
print('Latest date:', df['date'].max())
print('Total tickers:', df['ticker'].n_unique())
"
```

## Step 2: Start the Application

### Terminal 1: Backend

```bash
uv run python backend/api_server.py
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### Terminal 2: Frontend

```bash
cd frontend && npm run dev
```

Wait for: `Local: http://localhost:3000/`

### Open Browser

Navigate to http://localhost:3000

## Step 3: Run Stock Screening

### Use the Scoring Panel

1. Click the **Scoring** tab
2. Click **Load Swing Trading Default** preset
3. Verify filters:
   - Months Back: 6 (or your preferred lookback)
   - Min Days: 20
   - Min Avg Volume: 100000
   - Price: $0 - $8 (adjust as needed)
4. Click **Score Stocks**

### Wait for Results

Scoring typically takes 10-30 seconds depending on universe size.

## Step 4: Review Results

### Top 10 Table

The scored stocks appear ranked by weighted criteria:

| Rank | Ticker | Score | Tradability | Range | Liquidity |
|------|--------|-------|-------------|-------|-----------|
| 1 | XYZ | 85.2 | 78 | $0.42 | 45x |
| 2 | ABC | 82.1 | 72 | $0.38 | 52x |
| ... | ... | ... | ... | ... | ... |

### Quick Filters

Mentally filter for:
- **Score > 75**: Strong candidates
- **Tradability > 60**: Good trading characteristics
- **Liquidity > 20x**: Clean execution

### Mark Interesting Tickers

Note 5-10 tickers for deeper analysis.

## Step 5: Detailed Analysis

For each interesting ticker:

### Check the Chart

1. Click the ticker to open Charts panel
2. Look for:
   - Clear trend or range
   - Not at earnings/catalyst
   - Volume patterns
   - Technical setup

### Key Technical Checks

| Check | What to Look For |
|-------|------------------|
| **Trend** | Is it trending or ranging? |
| **RSI** | Oversold (<30) or overbought (>70)? |
| **MACD** | Recent crossover? |
| **Volume** | Any unusual spikes? |
| **Support/Resistance** | Near key levels? |

### Red Flags

Remove from list if you see:
- Recent news event (gap up/down)
- Extremely low volume day
- Already extended (RSI >80 or <20 and moving)
- About to report earnings

## Step 6: Build Today's Watchlist

### Prioritize Setups

Rank remaining candidates by setup quality:

**A-Grade Setup**:
- Score > 80
- Clear technical pattern
- Volume confirmation
- No near-term catalyst

**B-Grade Setup**:
- Score 70-80
- Decent pattern
- Good liquidity
- Minor concerns

### Create Trading Plan

For each A/B-grade stock:

```
Ticker: XYZ
Setup: RSI oversold bounce
Entry: $4.50 (on confirmation above yesterday's high)
Stop: $4.30 (below recent low)
Target 1: $4.80 (first resistance)
Target 2: $5.00 (measured move)
Risk: $0.20 per share
Position: 2,500 shares (based on $500 risk)
```

## Step 7: Monitor Throughout Day

### Pre-Market

- Check any overnight news on watchlist stocks
- Note pre-market prices/gaps
- Remove any with material news

### Market Open (9:30-10:00)

- Don't trade first 30 minutes (too volatile)
- Let setups develop
- Watch for volume confirmation

### Mid-Day (10:00-15:00)

- Enter trades meeting your plan criteria
- Manage existing positions
- Note any new setups forming

### End of Day (15:00-16:00)

- Review open positions
- Decide on overnight holds
- Note stocks for tomorrow's watchlist

## Alternative: Command-Line Screening

If you prefer CLI over the web interface:

### Quick Score

```bash
# Output top 20 stocks to terminal
uv run python -c "
from api_server import score_stocks
results = score_stocks(months_back=6, min_days=20)
for r in results[:20]:
    print(f\"{r['ticker']}: {r['score']:.1f}\")
"
```

### Export to CSV

```bash
uv run python -c "
import json
from api_server import score_stocks
results = score_stocks(months_back=6, min_days=20)
import pandas as pd
pd.DataFrame(results).to_csv('daily_picks.csv', index=False)
"
```

## Weekly Adjustments

### Monday Morning

- Review previous week's picks
- Check for any earnings this week
- Note sector performance

### Friday Afternoon

- Review week's performance
- Identify what worked/didn't
- Adjust scoring weights if needed

## Logging Your Picks

Keep a trading journal:

```markdown
## 2026-01-18 Daily Screen

### Top Picks
1. XYZ (Score 85) - RSI oversold, watching for bounce
2. ABC (Score 82) - Consolidating, potential breakout

### Trades Taken
- XYZ: Entered $4.50, Stop $4.30, Target $4.80

### Notes
- Market trending up, bullish bias
- Volume low, lighter positions
```

## Common Issues

### "No stocks match criteria"

- Widen price range
- Lower minimum volume
- Increase months back (need more data)

### "Too many results"

- Tighten tradability score minimum
- Reduce price range
- Add maximum volatility filter

### "Data not updating"

- Check Polygon API key
- Verify market was open
- Check data directory permissions

## Optimization Tips

### Speed Up Daily Workflow

1. **Automate data update**: Schedule `build_polygon_cache.py` to run before market open
2. **Save presets**: Create presets for different strategies
3. **Use CLI**: Faster than web UI for experienced users

### Improve Selection Quality

1. **Track results**: Note which scores/setups actually perform
2. **Adjust weights**: Increase weight on metrics that matter for your style
3. **Combine with factors**: Cross-reference with factor evaluation

---

*Continue to: [Factor Research Workflow](02-factor-research.md)*

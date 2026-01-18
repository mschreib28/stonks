#!/usr/bin/env python3
"""
Build technical indicator features from daily OHLCV data.

Implements technical indicators recommended in ML for Algorithmic Trading:
- Momentum: RSI, Stochastic Oscillator, ROC
- Trend: ADX, Bollinger Bands, SMA crossovers
- Volume: OBV, VWAP approximation
- Volatility: ATR, Historical Volatility

Uses pandas-ta for indicator calculations (pure Python, no native dependencies).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import polars as pl
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).
    RSI = 100 - (100 / (1 + RS))
    RS = Average Gain / Average Loss
    """
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, float('inf'))
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> tuple[pd.Series, pd.Series]:
    """
    Compute Stochastic Oscillator (%K and %D).
    %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
    %D = SMA of %K
    """
    lowest_low = df['low'].rolling(window=k_period).min()
    highest_high = df['high'].rolling(window=k_period).max()
    
    stoch_k = ((df['close'] - lowest_low) / (highest_high - lowest_low)) * 100
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d


def compute_roc(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Compute Rate of Change (ROC).
    ROC = ((Close - Close_n) / Close_n) * 100
    """
    return ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average Directional Index (ADX).
    Measures trend strength regardless of direction.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
    
    # Smoothed averages
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
    
    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, float('inf'))
    adx = dx.ewm(span=period, adjust=False).mean()
    
    return adx


def compute_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.
    Middle = SMA(period)
    Upper = Middle + std_dev * StdDev(period)
    Lower = Middle - std_dev * StdDev(period)
    """
    middle = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower


def compute_sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """
    Compute SMA crossover signal.
    Returns: fast_sma - slow_sma (positive = bullish, negative = bearish)
    """
    fast_sma = df['close'].rolling(window=fast).mean()
    slow_sma = df['close'].rolling(window=slow).mean()
    return fast_sma - slow_sma


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """
    Compute On-Balance Volume (OBV).
    Cumulative volume based on price direction.
    """
    direction = df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (direction * df['volume']).cumsum()
    return obv


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Compute Volume Weighted Average Price (VWAP).
    For daily data, this is an approximation using typical price.
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    cumulative_tp_vol = (typical_price * df['volume']).cumsum()
    cumulative_vol = df['volume'].cumsum()
    vwap = cumulative_tp_vol / cumulative_vol.replace(0, float('nan'))
    return vwap


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Compute Average True Range (ATR).
    Measures volatility.
    """
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(span=period, adjust=False).mean()
    return atr


def compute_historical_volatility(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Compute historical volatility (annualized).
    Standard deviation of log returns, annualized.
    """
    import numpy as np
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hist_vol = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized
    return hist_vol


def add_technical_indicators_pandas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to a pandas DataFrame.
    Assumes DataFrame has: ticker, date, open, high, low, close, volume
    """
    # Group by ticker and apply indicators
    result_dfs = []
    
    for ticker, group in df.groupby('ticker'):
        group = group.sort_values('date').copy()
        
        # Momentum indicators
        group['rsi_14'] = compute_rsi(group, period=14)
        stoch_k, stoch_d = compute_stochastic(group, k_period=14, d_period=3)
        group['stoch_k'] = stoch_k
        group['stoch_d'] = stoch_d
        group['roc_10'] = compute_roc(group, period=10)
        
        # Trend indicators
        group['adx_14'] = compute_adx(group, period=14)
        bb_upper, bb_middle, bb_lower = compute_bollinger_bands(group, period=20, std_dev=2.0)
        group['bb_upper'] = bb_upper
        group['bb_middle'] = bb_middle
        group['bb_lower'] = bb_lower
        group['bb_pct'] = (group['close'] - bb_lower) / (bb_upper - bb_lower)  # Position within bands
        
        # SMA crossovers
        group['sma_cross_20_50'] = compute_sma_crossover(group, fast=20, slow=50)
        group['sma_cross_50_200'] = compute_sma_crossover(group, fast=50, slow=200)
        
        # Simple moving averages
        group['sma_20'] = group['close'].rolling(window=20).mean()
        group['sma_50'] = group['close'].rolling(window=50).mean()
        group['sma_200'] = group['close'].rolling(window=200).mean()
        
        # Volume indicators
        group['obv'] = compute_obv(group)
        group['vwap'] = compute_vwap(group)
        
        # Volatility indicators
        group['atr_14'] = compute_atr(group, period=14)
        group['hist_vol_20'] = compute_historical_volatility(group, period=20)
        
        # Derived signals
        # RSI overbought/oversold
        group['rsi_overbought'] = (group['rsi_14'] > 70).astype(int)
        group['rsi_oversold'] = (group['rsi_14'] < 30).astype(int)
        
        # Price relative to Bollinger Bands
        group['above_bb_upper'] = (group['close'] > group['bb_upper']).astype(int)
        group['below_bb_lower'] = (group['close'] < group['bb_lower']).astype(int)
        
        # Trend strength
        group['strong_trend'] = (group['adx_14'] > 25).astype(int)
        
        result_dfs.append(group)
    
    return pd.concat(result_dfs, ignore_index=True)


def build_technical_features_lazy(input_glob: str) -> pl.LazyFrame:
    """
    Load daily data and compute technical indicators.
    Returns a LazyFrame with all technical features.
    """
    # Load daily data
    lf = pl.scan_parquet(input_glob)
    
    # Get required columns
    required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    available_cols = [c for c in required_cols if c in lf.collect_schema().names()]
    
    # Select only required columns
    lf = lf.select(available_cols)
    
    # Convert to pandas for indicator calculations
    # (Technical indicators are complex and easier in pandas)
    df = lf.collect().to_pandas()
    
    # Add technical indicators
    df = add_technical_indicators_pandas(df)
    
    # Convert back to Polars LazyFrame
    return pl.from_pandas(df).lazy()


def build_technical_features_from_daily(daily_path: str, output_path: str) -> None:
    """
    Build technical features from daily cache file.
    
    Args:
        daily_path: Path to daily parquet file
        output_path: Path to output parquet file
    """
    print(f"Loading daily data from: {daily_path}")
    
    # Load daily data
    df = pl.read_parquet(daily_path)
    print(f"  Loaded {df.height:,} rows, {len(df.columns)} columns")
    
    # Check for required columns
    required_cols = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert to pandas for indicator calculations
    print("Computing technical indicators...")
    pdf = df.select(required_cols).to_pandas()
    
    # Add technical indicators
    pdf = add_technical_indicators_pandas(pdf)
    print(f"  Added {len(pdf.columns) - len(required_cols)} technical indicator columns")
    
    # Convert back to Polars
    result = pl.from_pandas(pdf)
    
    # Add year/month for partitioning compatibility
    if 'year' not in result.columns:
        result = result.with_columns([
            pl.col('date').dt.year().alias('year'),
            pl.col('date').dt.month().alias('month'),
        ])
    
    # Write output
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.write_parquet(out_path, compression='zstd')
    
    print(f"Wrote technical features to: {out_path}")
    print(f"  Rows: {result.height:,}")
    print(f"  Columns: {len(result.columns)}")
    
    # Print column summary
    indicator_cols = [c for c in result.columns if c not in required_cols + ['year', 'month']]
    print(f"\nTechnical indicator columns ({len(indicator_cols)}):")
    for col in sorted(indicator_cols):
        print(f"  - {col}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build technical indicator features from daily OHLCV data"
    )
    parser.add_argument(
        "--daily-path",
        default="data/cache/daily_2025.parquet",
        help="Path to daily parquet file",
    )
    parser.add_argument(
        "--output",
        default="data/cache/technical_features.parquet",
        help="Output path for technical features parquet",
    )
    
    args = parser.parse_args()
    
    build_technical_features_from_daily(
        daily_path=args.daily_path,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Vectorized backtesting using VectorBT.

Provides fast backtesting for signal-based strategies with:
- MACD crossover strategies
- RSI mean reversion strategies
- Custom signal backtesting
- Walk-forward optimization

Based on "Python for Algorithmic Trading Cookbook" Chapter 6.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_ohlcv_data(
    data_path: str,
    ticker: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLCV data from parquet file.
    
    Args:
        data_path: Path to parquet file with OHLCV data
        ticker: Optional ticker to filter (if None, loads all)
        start_date: Optional start date filter (YYYY-MM-DD)
        end_date: Optional end date filter (YYYY-MM-DD)
    
    Returns:
        DataFrame with OHLCV data
    """
    df = pl.read_parquet(data_path)
    
    if ticker:
        df = df.filter(pl.col('ticker') == ticker)
    
    if start_date:
        df = df.filter(pl.col('date') >= start_date)
    
    if end_date:
        df = df.filter(pl.col('date') <= end_date)
    
    # Convert to pandas
    pdf = df.to_pandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    
    return pdf.sort_values(['ticker', 'date'])


def backtest_macd_strategy(
    data: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
    initial_cash: float = 100000,
    fees: float = 0.001,  # 0.1% per trade
    slippage: float = 0.001,  # 0.1% slippage
) -> dict:
    """
    Backtest MACD crossover strategy using VectorBT.
    
    Entry: MACD histogram crosses above zero
    Exit: MACD histogram crosses below zero
    
    Args:
        data: DataFrame with OHLCV data (must have 'close' column)
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        initial_cash: Starting capital
        fees: Trading fees as fraction
        slippage: Slippage as fraction
    
    Returns:
        Dictionary with backtest results
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("vectorbt is required. Install with: pip install vectorbt")
    
    # If multiple tickers, process each separately
    if 'ticker' in data.columns and data['ticker'].nunique() > 1:
        results = {}
        for ticker, group in data.groupby('ticker'):
            if len(group) < slow_period + signal_period:
                continue
            group = group.sort_values('date').set_index('date')
            try:
                results[ticker] = _backtest_macd_single(
                    group, fast_period, slow_period, signal_period,
                    initial_cash, fees, slippage
                )
            except Exception as e:
                results[ticker] = {'error': str(e)}
        return {'by_ticker': results}
    else:
        # Single ticker
        if 'date' in data.columns:
            data = data.sort_values('date').set_index('date')
        return _backtest_macd_single(
            data, fast_period, slow_period, signal_period,
            initial_cash, fees, slippage
        )


def _backtest_macd_single(
    data: pd.DataFrame,
    fast_period: int,
    slow_period: int,
    signal_period: int,
    initial_cash: float,
    fees: float,
    slippage: float,
) -> dict:
    """Backtest MACD strategy for a single ticker."""
    import vectorbt as vbt
    
    close = data['close']
    
    # Compute MACD
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Generate signals
    # Entry: histogram crosses above zero
    # Exit: histogram crosses below zero
    entries = (histogram > 0) & (histogram.shift(1) <= 0)
    exits = (histogram < 0) & (histogram.shift(1) >= 0)
    
    # Run backtest
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        freq='1D',
    )
    
    # Extract results
    stats = pf.stats()
    
    return {
        'strategy': 'MACD Crossover',
        'parameters': {
            'fast_period': fast_period,
            'slow_period': slow_period,
            'signal_period': signal_period,
        },
        'performance': {
            'total_return': float(stats.get('Total Return [%]', 0)),
            'cagr': float(stats.get('CAGR [%]', 0)) if 'CAGR [%]' in stats else None,
            'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
            'sortino_ratio': float(stats.get('Sortino Ratio', 0)) if 'Sortino Ratio' in stats else None,
            'max_drawdown': float(stats.get('Max Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
            'profit_factor': float(stats.get('Profit Factor', 0)) if 'Profit Factor' in stats else None,
        },
        'trades': {
            'total_trades': int(stats.get('Total Trades', 0)),
            'avg_trade_return': float(stats.get('Avg Winning Trade [%]', 0)),
            'best_trade': float(stats.get('Best Trade [%]', 0)) if 'Best Trade [%]' in stats else None,
            'worst_trade': float(stats.get('Worst Trade [%]', 0)) if 'Worst Trade [%]' in stats else None,
        },
        'equity_curve': pf.value().to_dict() if hasattr(pf, 'value') else None,
        'drawdowns': pf.drawdowns.drawdown.to_dict() if hasattr(pf, 'drawdowns') else None,
    }


def backtest_rsi_strategy(
    data: pd.DataFrame,
    rsi_period: int = 14,
    oversold: float = 30,
    overbought: float = 70,
    initial_cash: float = 100000,
    fees: float = 0.001,
    slippage: float = 0.001,
) -> dict:
    """
    Backtest RSI mean reversion strategy.
    
    Entry: RSI crosses below oversold level
    Exit: RSI crosses above overbought level
    
    Args:
        data: DataFrame with OHLCV data
        rsi_period: RSI calculation period
        oversold: RSI level for entry
        overbought: RSI level for exit
        initial_cash: Starting capital
        fees: Trading fees
        slippage: Slippage
    
    Returns:
        Dictionary with backtest results
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("vectorbt is required")
    
    # If multiple tickers, process each separately
    if 'ticker' in data.columns and data['ticker'].nunique() > 1:
        results = {}
        for ticker, group in data.groupby('ticker'):
            if len(group) < rsi_period + 10:
                continue
            group = group.sort_values('date').set_index('date')
            try:
                results[ticker] = _backtest_rsi_single(
                    group, rsi_period, oversold, overbought,
                    initial_cash, fees, slippage
                )
            except Exception as e:
                results[ticker] = {'error': str(e)}
        return {'by_ticker': results}
    else:
        if 'date' in data.columns:
            data = data.sort_values('date').set_index('date')
        return _backtest_rsi_single(
            data, rsi_period, oversold, overbought,
            initial_cash, fees, slippage
        )


def _backtest_rsi_single(
    data: pd.DataFrame,
    rsi_period: int,
    oversold: float,
    overbought: float,
    initial_cash: float,
    fees: float,
    slippage: float,
) -> dict:
    """Backtest RSI strategy for a single ticker."""
    import vectorbt as vbt
    
    close = data['close']
    
    # Compute RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float('inf'))
    rsi = 100 - (100 / (1 + rs))
    
    # Generate signals
    entries = (rsi < oversold) & (rsi.shift(1) >= oversold)
    exits = (rsi > overbought) & (rsi.shift(1) <= overbought)
    
    # Run backtest
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        freq='1D',
    )
    
    stats = pf.stats()
    
    return {
        'strategy': 'RSI Mean Reversion',
        'parameters': {
            'rsi_period': rsi_period,
            'oversold': oversold,
            'overbought': overbought,
        },
        'performance': {
            'total_return': float(stats.get('Total Return [%]', 0)),
            'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
            'max_drawdown': float(stats.get('Max Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
        },
        'trades': {
            'total_trades': int(stats.get('Total Trades', 0)),
        },
    }


def backtest_custom_signals(
    data: pd.DataFrame,
    entries: pd.Series,
    exits: pd.Series,
    initial_cash: float = 100000,
    fees: float = 0.001,
    slippage: float = 0.001,
    strategy_name: str = 'Custom Strategy',
) -> dict:
    """
    Backtest using custom entry/exit signals.
    
    Args:
        data: DataFrame with at least 'close' column
        entries: Boolean series for entry signals
        exits: Boolean series for exit signals
        initial_cash: Starting capital
        fees: Trading fees
        slippage: Slippage
        strategy_name: Name for the strategy
    
    Returns:
        Dictionary with backtest results
    """
    try:
        import vectorbt as vbt
    except ImportError:
        raise ImportError("vectorbt is required")
    
    close = data['close']
    
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        slippage=slippage,
        freq='1D',
    )
    
    stats = pf.stats()
    
    return {
        'strategy': strategy_name,
        'performance': {
            'total_return': float(stats.get('Total Return [%]', 0)),
            'sharpe_ratio': float(stats.get('Sharpe Ratio', 0)),
            'max_drawdown': float(stats.get('Max Drawdown [%]', 0)),
            'win_rate': float(stats.get('Win Rate [%]', 0)),
        },
        'trades': {
            'total_trades': int(stats.get('Total Trades', 0)),
        },
    }


def walk_forward_optimization(
    data: pd.DataFrame,
    strategy_func: callable,
    param_grid: dict,
    train_months: int = 6,
    test_months: int = 1,
    metric: str = 'sharpe_ratio',
) -> dict:
    """
    Perform walk-forward optimization.
    
    Splits data into rolling train/test windows, optimizes on train,
    and validates on test.
    
    Args:
        data: DataFrame with OHLCV data (single ticker, date-indexed)
        strategy_func: Function that takes (data, **params) and returns results dict
        param_grid: Dict of parameter names to lists of values to test
        train_months: Training window in months
        test_months: Test window in months
        metric: Performance metric to optimize ('sharpe_ratio', 'total_return', etc.)
    
    Returns:
        Dictionary with optimization results
    """
    from itertools import product
    
    # Ensure date index
    if 'date' in data.columns:
        data = data.sort_values('date').set_index('date')
    
    # Generate parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(product(*param_values))
    
    # Get date range
    start_date = data.index.min()
    end_date = data.index.max()
    
    # Generate walk-forward windows
    windows = []
    current_start = start_date
    
    while True:
        train_end = current_start + pd.DateOffset(months=train_months)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        
        if test_end > end_date:
            break
        
        windows.append({
            'train_start': current_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
        })
        
        current_start = current_start + pd.DateOffset(months=test_months)
    
    if not windows:
        return {'error': 'Insufficient data for walk-forward optimization'}
    
    # Run walk-forward optimization
    window_results = []
    
    for window in windows:
        train_data = data[window['train_start']:window['train_end']]
        test_data = data[window['test_start']:window['test_end']]
        
        # Find best parameters on training data
        best_metric = float('-inf')
        best_params = None
        
        for combo in param_combinations:
            params = dict(zip(param_names, combo))
            try:
                result = strategy_func(train_data, **params)
                metric_value = result.get('performance', {}).get(metric, float('-inf'))
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params
            except Exception:
                continue
        
        if best_params is None:
            continue
        
        # Validate on test data
        try:
            test_result = strategy_func(test_data, **best_params)
            window_results.append({
                'window': {
                    'train': f"{window['train_start'].date()} to {window['train_end'].date()}",
                    'test': f"{window['test_start'].date()} to {window['test_end'].date()}",
                },
                'best_params': best_params,
                'train_metric': best_metric,
                'test_result': test_result,
            })
        except Exception as e:
            window_results.append({
                'window': {
                    'train': f"{window['train_start'].date()} to {window['train_end'].date()}",
                    'test': f"{window['test_start'].date()} to {window['test_end'].date()}",
                },
                'error': str(e),
            })
    
    # Summarize results
    test_returns = [
        w['test_result']['performance']['total_return']
        for w in window_results
        if 'test_result' in w and 'performance' in w['test_result']
    ]
    
    return {
        'windows': window_results,
        'summary': {
            'num_windows': len(window_results),
            'avg_test_return': float(np.mean(test_returns)) if test_returns else None,
            'std_test_return': float(np.std(test_returns)) if test_returns else None,
        },
    }


def run_backtest(
    data_path: str,
    strategy: str = 'macd',
    ticker: Optional[str] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Run a backtest from command line.
    
    Args:
        data_path: Path to data parquet file
        strategy: Strategy type ('macd', 'rsi')
        ticker: Optional ticker to filter
        output_path: Optional path to save results
        **kwargs: Strategy-specific parameters
    
    Returns:
        Backtest results dictionary
    """
    import json
    
    # Load data
    print(f"Loading data from: {data_path}")
    data = load_ohlcv_data(data_path, ticker=ticker)
    print(f"  Loaded {len(data):,} rows")
    
    if ticker:
        print(f"  Ticker: {ticker}")
    else:
        tickers = data['ticker'].nunique()
        print(f"  Tickers: {tickers}")
    
    # Run backtest
    print(f"\nRunning {strategy.upper()} strategy backtest...")
    
    if strategy.lower() == 'macd':
        results = backtest_macd_strategy(data, **kwargs)
    elif strategy.lower() == 'rsi':
        results = backtest_rsi_strategy(data, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Backtest Results")
    print("=" * 60)
    
    if 'by_ticker' in results:
        # Multiple tickers
        print(f"\nResults for {len(results['by_ticker'])} tickers:")
        for ticker_name, ticker_results in list(results['by_ticker'].items())[:10]:
            if 'error' in ticker_results:
                print(f"  {ticker_name}: Error - {ticker_results['error']}")
            else:
                perf = ticker_results.get('performance', {})
                print(f"  {ticker_name}: Return={perf.get('total_return', 0):.1f}%, "
                      f"Sharpe={perf.get('sharpe_ratio', 0):.2f}, "
                      f"MaxDD={perf.get('max_drawdown', 0):.1f}%")
    else:
        # Single ticker
        perf = results.get('performance', {})
        trades = results.get('trades', {})
        
        print(f"\nStrategy: {results.get('strategy', 'Unknown')}")
        print(f"Parameters: {results.get('parameters', {})}")
        print(f"\nPerformance:")
        print(f"  Total Return: {perf.get('total_return', 0):.2f}%")
        print(f"  Sharpe Ratio: {perf.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {perf.get('max_drawdown', 0):.2f}%")
        print(f"  Win Rate: {perf.get('win_rate', 0):.1f}%")
        print(f"\nTrades:")
        print(f"  Total Trades: {trades.get('total_trades', 0)}")
    
    # Save results
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            elif hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                return obj
        
        with open(out_path, 'w') as f:
            json.dump(make_serializable(results), f, indent=2, default=str)
        
        print(f"\nResults saved to: {out_path}")
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vectorized backtests using VectorBT"
    )
    parser.add_argument(
        "--data-path",
        default="data/cache/daily_2025.parquet",
        help="Path to OHLCV data parquet file",
    )
    parser.add_argument(
        "--strategy",
        choices=["macd", "rsi"],
        default="macd",
        help="Strategy to backtest",
    )
    parser.add_argument(
        "--ticker",
        help="Optional: filter to specific ticker",
    )
    parser.add_argument(
        "--output",
        help="Output path for results (JSON)",
    )
    
    # MACD parameters
    parser.add_argument("--fast-period", type=int, default=12, help="MACD fast period")
    parser.add_argument("--slow-period", type=int, default=26, help="MACD slow period")
    parser.add_argument("--signal-period", type=int, default=9, help="MACD signal period")
    
    # RSI parameters
    parser.add_argument("--rsi-period", type=int, default=14, help="RSI period")
    parser.add_argument("--oversold", type=float, default=30, help="RSI oversold level")
    parser.add_argument("--overbought", type=float, default=70, help="RSI overbought level")
    
    # Common parameters
    parser.add_argument("--initial-cash", type=float, default=100000, help="Initial cash")
    parser.add_argument("--fees", type=float, default=0.001, help="Trading fees (fraction)")
    parser.add_argument("--slippage", type=float, default=0.001, help="Slippage (fraction)")
    
    args = parser.parse_args()
    
    # Build kwargs based on strategy
    if args.strategy == 'macd':
        kwargs = {
            'fast_period': args.fast_period,
            'slow_period': args.slow_period,
            'signal_period': args.signal_period,
            'initial_cash': args.initial_cash,
            'fees': args.fees,
            'slippage': args.slippage,
        }
    else:  # rsi
        kwargs = {
            'rsi_period': args.rsi_period,
            'oversold': args.oversold,
            'overbought': args.overbought,
            'initial_cash': args.initial_cash,
            'fees': args.fees,
            'slippage': args.slippage,
        }
    
    run_backtest(
        data_path=args.data_path,
        strategy=args.strategy,
        ticker=args.ticker,
        output_path=args.output,
        **kwargs,
    )


if __name__ == "__main__":
    main()

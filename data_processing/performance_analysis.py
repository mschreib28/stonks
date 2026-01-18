#!/usr/bin/env python3
"""
Performance analytics using Pyfolio methodology.

Provides comprehensive strategy performance analysis:
- Returns analysis (cumulative, rolling)
- Risk metrics (Sharpe, Sortino, VaR, CVaR)
- Drawdown analysis
- Trade analysis

Based on "Python for Algorithmic Trading Cookbook" Chapter 9.
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


# Trading days per year for annualization
TRADING_DAYS_PER_YEAR = 252


def compute_returns(prices: pd.Series) -> pd.Series:
    """Compute simple returns from price series."""
    return prices.pct_change().dropna()


def compute_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Compute cumulative returns."""
    return (1 + returns).cumprod() - 1


def compute_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Sharpe = (Mean Return - Risk Free Rate) / Std Dev * sqrt(annualization_factor)
    """
    excess_returns = returns - risk_free_rate / annualization_factor
    if returns.std() == 0:
        return 0.0
    return float(excess_returns.mean() / returns.std() * np.sqrt(annualization_factor))


def compute_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute annualized Sortino ratio.
    
    Uses downside deviation instead of standard deviation.
    """
    excess_returns = returns - risk_free_rate / annualization_factor
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    downside_std = downside_returns.std()
    return float(excess_returns.mean() / downside_std * np.sqrt(annualization_factor))


def compute_max_drawdown(returns: pd.Series) -> dict:
    """
    Compute maximum drawdown and related metrics.
    
    Returns dict with:
    - max_drawdown: Maximum percentage drawdown
    - max_drawdown_duration: Duration of worst drawdown
    - drawdown_series: Series of drawdowns over time
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    
    max_dd = float(drawdown.min())
    
    # Find drawdown periods
    in_drawdown = drawdown < 0
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1).fillna(False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1).fillna(False)
    
    # Calculate longest drawdown duration
    durations = []
    start_idx = None
    for i, (is_start, is_end) in enumerate(zip(drawdown_starts, drawdown_ends)):
        if is_start:
            start_idx = i
        if is_end and start_idx is not None:
            durations.append(i - start_idx)
            start_idx = None
    
    max_duration = max(durations) if durations else 0
    
    return {
        'max_drawdown': max_dd,
        'max_drawdown_pct': max_dd * 100,
        'max_drawdown_duration_days': max_duration,
        'drawdown_series': drawdown.to_dict(),
    }


def compute_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Value at Risk (VaR) at given confidence level.
    
    VaR represents the worst expected loss at the given confidence level.
    """
    return float(np.percentile(returns, (1 - confidence_level) * 100))


def compute_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
) -> float:
    """
    Compute Conditional Value at Risk (CVaR / Expected Shortfall).
    
    CVaR is the average of returns worse than VaR.
    """
    var = compute_var(returns, confidence_level)
    return float(returns[returns <= var].mean())


def compute_calmar_ratio(
    returns: pd.Series,
    annualization_factor: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Compute Calmar ratio.
    
    Calmar = Annualized Return / Max Drawdown
    """
    annualized_return = returns.mean() * annualization_factor
    dd_info = compute_max_drawdown(returns)
    max_dd = abs(dd_info['max_drawdown'])
    
    if max_dd == 0:
        return 0.0
    
    return float(annualized_return / max_dd)


def compute_win_rate(returns: pd.Series) -> float:
    """Compute percentage of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns))


def compute_profit_factor(returns: pd.Series) -> float:
    """
    Compute profit factor.
    
    Profit Factor = Sum of Wins / Sum of Losses
    """
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf') if wins > 0 else 0.0
    
    return float(wins / losses)


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """Compute rolling Sharpe ratio."""
    excess_returns = returns - risk_free_rate / TRADING_DAYS_PER_YEAR
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = rolling_mean / rolling_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    return rolling_sharpe


def compute_rolling_volatility(
    returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Compute rolling annualized volatility."""
    return returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS_PER_YEAR)


def analyze_returns(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Comprehensive returns analysis.
    
    Args:
        returns: Strategy returns series
        benchmark_returns: Optional benchmark returns for comparison
        risk_free_rate: Annual risk-free rate
    
    Returns:
        Dictionary with all performance metrics
    """
    # Basic metrics
    total_return = float((1 + returns).prod() - 1)
    annualized_return = float(returns.mean() * TRADING_DAYS_PER_YEAR)
    annualized_volatility = float(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    
    # Risk-adjusted metrics
    sharpe = compute_sharpe_ratio(returns, risk_free_rate)
    sortino = compute_sortino_ratio(returns, risk_free_rate)
    calmar = compute_calmar_ratio(returns)
    
    # Drawdown analysis
    dd_info = compute_max_drawdown(returns)
    
    # Risk metrics
    var_95 = compute_var(returns, 0.95)
    cvar_95 = compute_cvar(returns, 0.95)
    
    # Trade metrics
    win_rate = compute_win_rate(returns)
    profit_factor = compute_profit_factor(returns)
    
    # Rolling metrics
    rolling_sharpe = compute_rolling_sharpe(returns)
    rolling_vol = compute_rolling_volatility(returns)
    
    results = {
        'summary': {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'annualized_return': annualized_return,
            'annualized_return_pct': annualized_return * 100,
            'annualized_volatility': annualized_volatility,
            'annualized_volatility_pct': annualized_volatility * 100,
            'trading_days': len(returns),
        },
        'risk_adjusted': {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
        },
        'drawdown': {
            'max_drawdown_pct': dd_info['max_drawdown_pct'],
            'max_drawdown_duration_days': dd_info['max_drawdown_duration_days'],
        },
        'risk': {
            'var_95_pct': var_95 * 100,
            'cvar_95_pct': cvar_95 * 100,
        },
        'trade_metrics': {
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'best_day_pct': float(returns.max() * 100),
            'worst_day_pct': float(returns.min() * 100),
            'avg_daily_return_pct': float(returns.mean() * 100),
        },
        'time_series': {
            'cumulative_returns': compute_cumulative_returns(returns).to_dict(),
            'rolling_sharpe_60d': rolling_sharpe.dropna().to_dict(),
            'rolling_volatility_20d': rolling_vol.dropna().to_dict(),
            'drawdown': dd_info['drawdown_series'],
        },
    }
    
    # Add benchmark comparison if provided
    if benchmark_returns is not None:
        # Align returns
        aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ['strategy', 'benchmark']
        
        if len(aligned) > 0:
            # Beta
            covariance = aligned.cov().iloc[0, 1]
            benchmark_var = aligned['benchmark'].var()
            beta = covariance / benchmark_var if benchmark_var != 0 else 0
            
            # Alpha (Jensen's alpha)
            benchmark_return = aligned['benchmark'].mean() * TRADING_DAYS_PER_YEAR
            strategy_return = aligned['strategy'].mean() * TRADING_DAYS_PER_YEAR
            alpha = strategy_return - (risk_free_rate + beta * (benchmark_return - risk_free_rate))
            
            # Information ratio
            tracking_error = (aligned['strategy'] - aligned['benchmark']).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
            excess_return = strategy_return - benchmark_return
            information_ratio = excess_return / tracking_error if tracking_error != 0 else 0
            
            results['benchmark_comparison'] = {
                'beta': float(beta),
                'alpha': float(alpha),
                'alpha_pct': float(alpha * 100),
                'information_ratio': float(information_ratio),
                'correlation': float(aligned['strategy'].corr(aligned['benchmark'])),
                'benchmark_sharpe': compute_sharpe_ratio(aligned['benchmark'], risk_free_rate),
            }
    
    return results


def analyze_portfolio(
    prices: pd.DataFrame,
    weights: Optional[dict] = None,
    rebalance_freq: str = 'M',
    initial_value: float = 100000,
) -> dict:
    """
    Analyze a portfolio of assets.
    
    Args:
        prices: DataFrame with asset prices (columns = tickers)
        weights: Optional dict of ticker -> weight (None = equal weight)
        rebalance_freq: Rebalancing frequency ('D', 'W', 'M', 'Q', 'Y')
        initial_value: Initial portfolio value
    
    Returns:
        Portfolio performance analysis
    """
    # Default to equal weight
    if weights is None:
        n_assets = len(prices.columns)
        weights = {col: 1.0 / n_assets for col in prices.columns}
    
    # Compute returns
    returns = prices.pct_change().dropna()
    
    # Compute portfolio returns
    weight_series = pd.Series(weights)
    portfolio_returns = (returns * weight_series).sum(axis=1)
    
    # Analyze portfolio
    analysis = analyze_returns(portfolio_returns)
    
    # Add portfolio-specific metrics
    analysis['portfolio'] = {
        'num_assets': len(prices.columns),
        'weights': weights,
        'rebalance_freq': rebalance_freq,
    }
    
    # Asset-level metrics
    asset_metrics = {}
    for col in prices.columns:
        asset_returns = returns[col]
        asset_metrics[col] = {
            'weight': weights.get(col, 0),
            'total_return_pct': float((1 + asset_returns).prod() - 1) * 100,
            'sharpe': compute_sharpe_ratio(asset_returns),
            'max_drawdown_pct': compute_max_drawdown(asset_returns)['max_drawdown_pct'],
        }
    
    analysis['assets'] = asset_metrics
    
    return analysis


def load_and_analyze(
    data_path: str,
    ticker: Optional[str] = None,
    benchmark_ticker: Optional[str] = None,
    price_column: str = 'close',
    output_path: Optional[str] = None,
) -> dict:
    """
    Load data and run performance analysis.
    
    Args:
        data_path: Path to parquet file with price data
        ticker: Ticker to analyze (if None, analyzes all as portfolio)
        benchmark_ticker: Optional benchmark ticker for comparison
        price_column: Column name for prices
        output_path: Optional path to save results
    
    Returns:
        Performance analysis results
    """
    import json
    
    print(f"Loading data from: {data_path}")
    df = pl.read_parquet(data_path)
    
    # Convert to pandas
    pdf = df.to_pandas()
    pdf['date'] = pd.to_datetime(pdf['date'])
    
    if ticker:
        # Single ticker analysis
        print(f"Analyzing ticker: {ticker}")
        ticker_data = pdf[pdf['ticker'] == ticker].sort_values('date').set_index('date')
        
        if len(ticker_data) == 0:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        prices = ticker_data[price_column]
        returns = compute_returns(prices)
        
        # Load benchmark if specified
        benchmark_returns = None
        if benchmark_ticker:
            bench_data = pdf[pdf['ticker'] == benchmark_ticker].sort_values('date').set_index('date')
            if len(bench_data) > 0:
                benchmark_returns = compute_returns(bench_data[price_column])
        
        results = analyze_returns(returns, benchmark_returns)
        results['ticker'] = ticker
        
    else:
        # Portfolio analysis
        print("Analyzing portfolio (all tickers, equal weight)")
        
        # Pivot to wide format
        prices_wide = pdf.pivot(index='date', columns='ticker', values=price_column)
        prices_wide = prices_wide.dropna(how='all')
        
        results = analyze_portfolio(prices_wide)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Performance Analysis Summary")
    print("=" * 60)
    
    summary = results.get('summary', {})
    risk_adj = results.get('risk_adjusted', {})
    dd = results.get('drawdown', {})
    trade = results.get('trade_metrics', {})
    
    print(f"\nReturns:")
    print(f"  Total Return: {summary.get('total_return_pct', 0):.2f}%")
    print(f"  Annualized Return: {summary.get('annualized_return_pct', 0):.2f}%")
    print(f"  Annualized Volatility: {summary.get('annualized_volatility_pct', 0):.2f}%")
    
    print(f"\nRisk-Adjusted:")
    print(f"  Sharpe Ratio: {risk_adj.get('sharpe_ratio', 0):.2f}")
    print(f"  Sortino Ratio: {risk_adj.get('sortino_ratio', 0):.2f}")
    print(f"  Calmar Ratio: {risk_adj.get('calmar_ratio', 0):.2f}")
    
    print(f"\nDrawdown:")
    print(f"  Max Drawdown: {dd.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Max DD Duration: {dd.get('max_drawdown_duration_days', 0)} days")
    
    print(f"\nTrade Metrics:")
    print(f"  Win Rate: {trade.get('win_rate_pct', 0):.1f}%")
    print(f"  Best Day: {trade.get('best_day_pct', 0):.2f}%")
    print(f"  Worst Day: {trade.get('worst_day_pct', 0):.2f}%")
    
    if 'benchmark_comparison' in results:
        bench = results['benchmark_comparison']
        print(f"\nBenchmark Comparison ({benchmark_ticker}):")
        print(f"  Beta: {bench.get('beta', 0):.2f}")
        print(f"  Alpha: {bench.get('alpha_pct', 0):.2f}%")
        print(f"  Information Ratio: {bench.get('information_ratio', 0):.2f}")
    
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
        description="Analyze strategy or portfolio performance using Pyfolio methodology"
    )
    parser.add_argument(
        "--data-path",
        default="data/cache/daily_2025.parquet",
        help="Path to price data parquet file",
    )
    parser.add_argument(
        "--ticker",
        help="Ticker to analyze (if not specified, analyzes all as portfolio)",
    )
    parser.add_argument(
        "--benchmark",
        help="Benchmark ticker for comparison",
    )
    parser.add_argument(
        "--price-column",
        default="close",
        help="Column name for prices",
    )
    parser.add_argument(
        "--output",
        help="Output path for results (JSON)",
    )
    
    args = parser.parse_args()
    
    load_and_analyze(
        data_path=args.data_path,
        ticker=args.ticker,
        benchmark_ticker=args.benchmark,
        price_column=args.price_column,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Factor evaluation using Alphalens methodology.

Computes Information Coefficient (IC), factor returns, and turnover analysis
for any factor/signal from the technical features or MACD data.

Based on "Python for Algorithmic Trading Cookbook" Chapter 8.
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

# Suppress some warnings from alphalens
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_factor_and_prices(
    factor_path: str,
    factor_column: str,
    price_path: str,
    ticker_column: str = 'ticker',
    date_column: str = 'date',
    price_column: str = 'close',
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Load factor values and price data for Alphalens analysis.
    
    Args:
        factor_path: Path to parquet file containing factor values
        factor_column: Name of the factor column to evaluate
        price_path: Path to parquet file containing price data
        ticker_column: Name of the ticker column
        date_column: Name of the date column
        price_column: Name of the price column
    
    Returns:
        Tuple of (factor_series, prices_df) formatted for Alphalens
    """
    # Load factor data
    factor_df = pl.read_parquet(factor_path)
    
    # Ensure required columns exist
    required_cols = [ticker_column, date_column, factor_column]
    missing = [c for c in required_cols if c not in factor_df.columns]
    if missing:
        raise ValueError(f"Missing columns in factor file: {missing}")
    
    # Convert to pandas
    factor_pdf = factor_df.select([ticker_column, date_column, factor_column]).to_pandas()
    
    # Convert date column to datetime
    factor_pdf[date_column] = pd.to_datetime(factor_pdf[date_column])
    
    # Create MultiIndex factor series (date, ticker) -> factor_value
    factor_pdf = factor_pdf.dropna(subset=[factor_column])
    factor_pdf = factor_pdf.set_index([date_column, ticker_column])
    factor_series = factor_pdf[factor_column]
    
    # Load price data
    price_df = pl.read_parquet(price_path)
    
    # Ensure required columns exist
    price_required = [ticker_column, date_column, price_column]
    price_missing = [c for c in price_required if c not in price_df.columns]
    if price_missing:
        raise ValueError(f"Missing columns in price file: {price_missing}")
    
    # Convert to pandas and pivot to wide format (date x ticker)
    price_pdf = price_df.select([ticker_column, date_column, price_column]).to_pandas()
    price_pdf[date_column] = pd.to_datetime(price_pdf[date_column])
    
    prices_wide = price_pdf.pivot(index=date_column, columns=ticker_column, values=price_column)
    
    return factor_series, prices_wide


def compute_information_coefficient(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: tuple[int, ...] = (1, 5, 10),
    quantiles: int = 5,
) -> dict:
    """
    Compute Information Coefficient and related metrics using Alphalens.
    
    Args:
        factor: Factor values with MultiIndex (date, ticker)
        prices: Price data with date index and ticker columns
        periods: Forward return periods to compute
        quantiles: Number of quantiles for factor grouping
    
    Returns:
        Dictionary containing IC metrics and factor returns
    """
    try:
        import alphalens.utils as al_utils
        import alphalens.performance as al_perf
        import alphalens.tears as al_tears
    except ImportError:
        raise ImportError("alphalens-reloaded is required. Install with: pip install alphalens-reloaded")
    
    # Align factor and price data
    # Get clean factor and forward returns
    factor_data = al_utils.get_clean_factor_and_forward_returns(
        factor=factor,
        prices=prices,
        periods=periods,
        quantiles=quantiles,
        max_loss=0.5,  # Allow up to 50% data loss for alignment
    )
    
    # Compute IC (Information Coefficient)
    ic = al_perf.factor_information_coefficient(factor_data)
    ic_summary = al_perf.mean_information_coefficient(factor_data)
    
    # Compute factor returns by quantile
    factor_returns = al_perf.factor_returns(factor_data)
    mean_returns = al_perf.mean_return_by_quantile(factor_data)
    
    # Compute turnover
    turnover = al_perf.factor_rank_autocorrelation(factor_data)
    
    # Compute cumulative returns
    cumulative_returns = al_perf.factor_cumulative_returns(factor_data)
    
    results = {
        'ic_summary': ic_summary.to_dict() if hasattr(ic_summary, 'to_dict') else dict(ic_summary),
        'ic_by_period': ic.to_dict() if hasattr(ic, 'to_dict') else None,
        'mean_returns_by_quantile': mean_returns[0].to_dict() if isinstance(mean_returns, tuple) else mean_returns.to_dict(),
        'factor_returns': factor_returns.to_dict() if hasattr(factor_returns, 'to_dict') else None,
        'turnover': turnover.to_dict() if hasattr(turnover, 'to_dict') else None,
        'cumulative_returns': cumulative_returns.to_dict() if hasattr(cumulative_returns, 'to_dict') else None,
    }
    
    return results


def compute_ic_simple(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: list[int] = [1, 5, 10],
) -> dict:
    """
    Compute Information Coefficient using simple rank correlation.
    
    This is a fallback when Alphalens has issues, using basic Spearman correlation
    between factor ranks and forward returns.
    
    Args:
        factor: Factor values with MultiIndex (date, ticker)
        prices: Price data with date index and ticker columns
        periods: Forward return periods to compute
    
    Returns:
        Dictionary containing IC metrics
    """
    from scipy.stats import spearmanr
    
    # Convert factor to DataFrame for easier manipulation
    factor_df = factor.reset_index()
    factor_df.columns = ['date', 'ticker', 'factor']
    
    # Pivot prices to long format
    prices_long = prices.stack().reset_index()
    prices_long.columns = ['date', 'ticker', 'close']
    
    # Merge factor with prices
    merged = pd.merge(factor_df, prices_long, on=['date', 'ticker'])
    merged = merged.sort_values(['ticker', 'date'])
    
    ic_results = {}
    
    for period in periods:
        # Calculate forward returns
        merged[f'fwd_ret_{period}d'] = merged.groupby('ticker')['close'].pct_change(period).shift(-period)
        
        # Group by date and compute IC (Spearman correlation)
        ic_by_date = []
        for date, group in merged.groupby('date'):
            valid = group.dropna(subset=['factor', f'fwd_ret_{period}d'])
            if len(valid) >= 10:  # Need at least 10 stocks
                corr, _ = spearmanr(valid['factor'], valid[f'fwd_ret_{period}d'])
                ic_by_date.append({'date': date, 'ic': corr})
        
        ic_df = pd.DataFrame(ic_by_date)
        if len(ic_df) > 0:
            ic_results[f'{period}D'] = {
                'mean': float(ic_df['ic'].mean()),
                'std': float(ic_df['ic'].std()),
                't_stat': float(ic_df['ic'].mean() / (ic_df['ic'].std() / np.sqrt(len(ic_df)))) if ic_df['ic'].std() > 0 else 0,
                'positive_pct': float((ic_df['ic'] > 0).mean()),
                'ic_series': ic_df.set_index('date')['ic'].to_dict(),
            }
    
    return ic_results


def compute_factor_quantile_returns(
    factor: pd.Series,
    prices: pd.DataFrame,
    periods: list[int] = [1, 5, 10],
    quantiles: int = 5,
) -> dict:
    """
    Compute mean returns by factor quantile.
    
    Args:
        factor: Factor values with MultiIndex (date, ticker)
        prices: Price data with date index and ticker columns
        periods: Forward return periods
        quantiles: Number of quantiles
    
    Returns:
        Dictionary of mean returns by quantile
    """
    # Convert factor to DataFrame
    factor_df = factor.reset_index()
    factor_df.columns = ['date', 'ticker', 'factor']
    
    # Pivot prices to long format
    prices_long = prices.stack().reset_index()
    prices_long.columns = ['date', 'ticker', 'close']
    
    # Merge
    merged = pd.merge(factor_df, prices_long, on=['date', 'ticker'])
    merged = merged.sort_values(['ticker', 'date'])
    
    results = {}
    
    for period in periods:
        # Calculate forward returns
        merged[f'fwd_ret_{period}d'] = merged.groupby('ticker')['close'].pct_change(period).shift(-period)
        
        # Assign quantiles by date
        def assign_quantile(group):
            try:
                group['quantile'] = pd.qcut(group['factor'], q=quantiles, labels=False, duplicates='drop') + 1
            except ValueError:
                group['quantile'] = np.nan
            return group
        
        merged_q = merged.groupby('date', group_keys=False).apply(assign_quantile)
        
        # Compute mean return by quantile
        quantile_returns = merged_q.groupby('quantile')[f'fwd_ret_{period}d'].mean()
        
        results[f'{period}D'] = {
            'mean_returns': quantile_returns.to_dict(),
            'spread': float(quantile_returns.iloc[-1] - quantile_returns.iloc[0]) if len(quantile_returns) >= 2 else 0,
        }
    
    return results


def evaluate_factor(
    factor_path: str,
    factor_column: str,
    price_path: str,
    periods: list[int] = [1, 5, 10],
    quantiles: int = 5,
    output_path: Optional[str] = None,
) -> dict:
    """
    Complete factor evaluation.
    
    Args:
        factor_path: Path to factor data parquet
        factor_column: Column name of the factor to evaluate
        price_path: Path to price data parquet
        periods: Forward return periods
        quantiles: Number of quantiles
        output_path: Optional path to save results as JSON
    
    Returns:
        Dictionary with all evaluation metrics
    """
    import json
    
    print(f"Evaluating factor: {factor_column}")
    print(f"  Factor data: {factor_path}")
    print(f"  Price data: {price_path}")
    print(f"  Periods: {periods}")
    print(f"  Quantiles: {quantiles}")
    print()
    
    # Load data
    factor, prices = load_factor_and_prices(
        factor_path=factor_path,
        factor_column=factor_column,
        price_path=price_path,
    )
    
    print(f"Loaded {len(factor):,} factor observations")
    print(f"Loaded prices for {len(prices.columns):,} tickers, {len(prices):,} dates")
    print()
    
    # Try Alphalens first, fall back to simple method
    try:
        print("Computing IC using Alphalens...")
        results = compute_information_coefficient(factor, prices, tuple(periods), quantiles)
        print("  Success!")
    except Exception as e:
        print(f"  Alphalens failed: {e}")
        print("  Falling back to simple IC computation...")
        results = {}
    
    # Always compute simple IC for comparison
    print("Computing simple IC (Spearman correlation)...")
    simple_ic = compute_ic_simple(factor, prices, periods)
    results['simple_ic'] = simple_ic
    
    # Compute quantile returns
    print("Computing quantile returns...")
    quantile_returns = compute_factor_quantile_returns(factor, prices, periods, quantiles)
    results['quantile_returns'] = quantile_returns
    
    # Print summary
    print()
    print("=" * 60)
    print(f"Factor Evaluation Summary: {factor_column}")
    print("=" * 60)
    
    for period_key, ic_data in simple_ic.items():
        print(f"\n{period_key} Forward Returns:")
        print(f"  Mean IC: {ic_data['mean']:.4f}")
        print(f"  IC Std:  {ic_data['std']:.4f}")
        print(f"  t-stat:  {ic_data['t_stat']:.2f}")
        print(f"  % Positive IC: {ic_data['positive_pct']:.1%}")
    
    print("\nQuantile Returns (spread = Q5 - Q1):")
    for period_key, qr_data in quantile_returns.items():
        print(f"  {period_key}: spread = {qr_data['spread']:.4f}")
    
    # Save results if output path provided
    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert any non-serializable objects
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif pd.isna(obj):
                return None
            else:
                return obj
        
        serializable_results = make_serializable(results)
        
        with open(out_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {out_path}")
    
    return results


def evaluate_all_factors(
    factor_path: str,
    price_path: str,
    factor_columns: Optional[list[str]] = None,
    periods: list[int] = [1, 5, 10],
    quantiles: int = 5,
    output_dir: Optional[str] = None,
) -> dict:
    """
    Evaluate multiple factors and create a summary comparison.
    
    Args:
        factor_path: Path to factor data parquet
        price_path: Path to price data parquet  
        factor_columns: List of factor columns to evaluate (None = all numeric columns)
        periods: Forward return periods
        quantiles: Number of quantiles
        output_dir: Optional directory to save results
    
    Returns:
        Dictionary with evaluation results for all factors
    """
    import json
    
    # Load factor data to get column names
    factor_df = pl.read_parquet(factor_path)
    
    if factor_columns is None:
        # Get all numeric columns except identifiers
        exclude_cols = {'ticker', 'date', 'year', 'month', 'open', 'high', 'low', 'close', 'volume'}
        factor_columns = [
            c for c in factor_df.columns 
            if c not in exclude_cols and factor_df[c].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
        ]
    
    print(f"Evaluating {len(factor_columns)} factors...")
    print()
    
    all_results = {}
    summary = []
    
    for factor_col in factor_columns:
        try:
            results = evaluate_factor(
                factor_path=factor_path,
                factor_column=factor_col,
                price_path=price_path,
                periods=periods,
                quantiles=quantiles,
            )
            all_results[factor_col] = results
            
            # Extract key metrics for summary
            if 'simple_ic' in results and f'{periods[0]}D' in results['simple_ic']:
                ic_1d = results['simple_ic'][f'{periods[0]}D']
                summary.append({
                    'factor': factor_col,
                    'mean_ic': ic_1d['mean'],
                    't_stat': ic_1d['t_stat'],
                    'positive_pct': ic_1d['positive_pct'],
                })
        except Exception as e:
            print(f"Error evaluating {factor_col}: {e}")
            all_results[factor_col] = {'error': str(e)}
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary)
    if len(summary_df) > 0:
        summary_df = summary_df.sort_values('mean_ic', ascending=False)
        
        print()
        print("=" * 80)
        print("Factor Comparison Summary (sorted by Mean IC)")
        print("=" * 80)
        print(summary_df.to_string(index=False))
    
    # Save results
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary_df.to_csv(out_dir / "factor_summary.csv", index=False)
        print(f"\nSummary saved to: {out_dir / 'factor_summary.csv'}")
        
        # Save detailed results
        with open(out_dir / "factor_details.json", 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"Details saved to: {out_dir / 'factor_details.json'}")
    
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate factor/signal quality using Information Coefficient and quantile returns"
    )
    parser.add_argument(
        "--factor-path",
        default="data/cache/technical_features.parquet",
        help="Path to factor data parquet file",
    )
    parser.add_argument(
        "--factor-column",
        default="rsi_14",
        help="Column name of the factor to evaluate (or 'all' for all factors)",
    )
    parser.add_argument(
        "--price-path",
        default="data/cache/daily_all.parquet",
        help="Path to price data parquet file",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        help="Forward return periods to evaluate",
    )
    parser.add_argument(
        "--quantiles",
        type=int,
        default=5,
        help="Number of quantiles for grouping",
    )
    parser.add_argument(
        "--output",
        help="Output path for results (JSON for single factor, directory for all)",
    )
    
    args = parser.parse_args()
    
    if args.factor_column.lower() == 'all':
        evaluate_all_factors(
            factor_path=args.factor_path,
            price_path=args.price_path,
            periods=args.periods,
            quantiles=args.quantiles,
            output_dir=args.output or "data/cache/factor_evaluation",
        )
    else:
        evaluate_factor(
            factor_path=args.factor_path,
            factor_column=args.factor_column,
            price_path=args.price_path,
            periods=args.periods,
            quantiles=args.quantiles,
            output_path=args.output,
        )


if __name__ == "__main__":
    main()

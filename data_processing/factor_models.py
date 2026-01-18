#!/usr/bin/env python3
"""
Factor model analysis for portfolio risk attribution.

Implements CAPM and Fama-French factor models for:
- Market beta calculation
- Factor exposure analysis
- Risk attribution

Based on "Machine Learning for Algorithmic Trading" Chapter 3.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings
import json

import numpy as np
import pandas as pd
import polars as pl

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Trading days per year
TRADING_DAYS_PER_YEAR = 252


def load_returns_data(
    data_path: str,
    ticker: Optional[str] = None,
    price_column: str = 'close',
) -> pd.DataFrame:
    """
    Load price data and compute returns.
    
    Args:
        data_path: Path to parquet file with price data
        ticker: Optional ticker to filter
        price_column: Column name for prices
    
    Returns:
        DataFrame with date index and returns column
    """
    df = pl.read_parquet(data_path)
    pdf = df.to_pandas()
    
    if ticker:
        pdf = pdf[pdf['ticker'] == ticker].copy()
    
    pdf['date'] = pd.to_datetime(pdf['date'])
    pdf = pdf.sort_values('date')
    
    # Compute returns
    pdf['returns'] = pdf[price_column].pct_change()
    
    return pdf.set_index('date')[['returns', price_column]].dropna()


def compute_capm_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute CAPM beta and related statistics using OLS regression.
    
    The CAPM model: E[r_i] = r_f + beta * (E[r_m] - r_f)
    
    Regression: (r_i - r_f) = alpha + beta * (r_m - r_f) + epsilon
    
    Args:
        returns: Asset returns series with datetime index
        market_returns: Market returns series with datetime index
        risk_free_rate: Daily risk-free rate (default 0)
    
    Returns:
        Dictionary with:
        - beta: Market sensitivity
        - alpha: Jensen's alpha (annualized)
        - alpha_daily: Daily alpha
        - r_squared: Explained variance
        - t_stat_beta: t-statistic for beta
        - t_stat_alpha: t-statistic for alpha
        - p_value_beta: p-value for beta
        - p_value_alpha: p-value for alpha
        - n_observations: Number of data points
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")
    
    # Align returns
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    aligned.columns = ['asset', 'market']
    
    if len(aligned) < 30:
        return {
            'error': f'Insufficient data: {len(aligned)} observations (need >= 30)',
            'n_observations': len(aligned),
        }
    
    # Compute excess returns
    y = aligned['asset'] - risk_free_rate
    X = aligned['market'] - risk_free_rate
    X = sm.add_constant(X)  # Add intercept
    
    # Run OLS regression
    model = sm.OLS(y, X).fit()
    
    # Extract results
    alpha_daily = model.params['const']
    beta = model.params['market']
    
    # Annualize alpha
    alpha_annualized = alpha_daily * TRADING_DAYS_PER_YEAR
    
    return {
        'beta': float(beta),
        'alpha': float(alpha_annualized),
        'alpha_daily': float(alpha_daily),
        'r_squared': float(model.rsquared),
        'adj_r_squared': float(model.rsquared_adj),
        't_stat_beta': float(model.tvalues['market']),
        't_stat_alpha': float(model.tvalues['const']),
        'p_value_beta': float(model.pvalues['market']),
        'p_value_alpha': float(model.pvalues['const']),
        'n_observations': len(aligned),
        'std_error_beta': float(model.bse['market']),
        'std_error_alpha': float(model.bse['const']),
        # Confidence intervals (95%)
        'beta_ci_lower': float(model.conf_int().loc['market', 0]),
        'beta_ci_upper': float(model.conf_int().loc['market', 1]),
    }


def compute_rolling_beta(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
    min_periods: int = 30,
) -> pd.Series:
    """
    Compute rolling window beta over time.
    
    Args:
        returns: Asset returns series
        market_returns: Market returns series
        window: Rolling window size in days
        min_periods: Minimum observations required
    
    Returns:
        Series with rolling beta values
    """
    # Align returns
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    aligned.columns = ['asset', 'market']
    
    # Compute rolling covariance and variance
    rolling_cov = aligned['asset'].rolling(window=window, min_periods=min_periods).cov(aligned['market'])
    rolling_var = aligned['market'].rolling(window=window, min_periods=min_periods).var()
    
    # Beta = Cov(asset, market) / Var(market)
    rolling_beta = rolling_cov / rolling_var
    rolling_beta.name = 'rolling_beta'
    
    return rolling_beta


def compute_rolling_alpha(
    returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
    min_periods: int = 30,
    risk_free_rate: float = 0.0,
) -> pd.Series:
    """
    Compute rolling Jensen's alpha (annualized).
    
    Args:
        returns: Asset returns series
        market_returns: Market returns series
        window: Rolling window size in days
        min_periods: Minimum observations required
        risk_free_rate: Daily risk-free rate
    
    Returns:
        Series with rolling alpha values (annualized)
    """
    # Align returns
    aligned = pd.concat([returns, market_returns], axis=1).dropna()
    aligned.columns = ['asset', 'market']
    
    # Compute rolling beta
    rolling_cov = aligned['asset'].rolling(window=window, min_periods=min_periods).cov(aligned['market'])
    rolling_var = aligned['market'].rolling(window=window, min_periods=min_periods).var()
    rolling_beta = rolling_cov / rolling_var
    
    # Compute rolling means
    rolling_asset_mean = aligned['asset'].rolling(window=window, min_periods=min_periods).mean()
    rolling_market_mean = aligned['market'].rolling(window=window, min_periods=min_periods).mean()
    
    # Alpha = E[r_asset] - rf - beta * (E[r_market] - rf)
    rolling_alpha_daily = (rolling_asset_mean - risk_free_rate) - rolling_beta * (rolling_market_mean - risk_free_rate)
    rolling_alpha_annual = rolling_alpha_daily * TRADING_DAYS_PER_YEAR
    rolling_alpha_annual.name = 'rolling_alpha'
    
    return rolling_alpha_annual


def get_market_returns(
    data_path: str,
    market_ticker: str = 'SPY',
    price_column: str = 'close',
) -> pd.Series:
    """
    Load market benchmark returns.
    
    Args:
        data_path: Path to parquet file with price data
        market_ticker: Ticker symbol for market benchmark
        price_column: Column name for prices
    
    Returns:
        Series with market returns, datetime index
    """
    df = pl.read_parquet(data_path)
    pdf = df.to_pandas()
    
    # Filter to market ticker
    market_df = pdf[pdf['ticker'] == market_ticker].copy()
    
    if len(market_df) == 0:
        raise ValueError(f"Market ticker '{market_ticker}' not found in data")
    
    market_df['date'] = pd.to_datetime(market_df['date'])
    market_df = market_df.sort_values('date')
    market_df['returns'] = market_df[price_column].pct_change()
    
    return market_df.set_index('date')['returns'].dropna()


def analyze_ticker_capm(
    data_path: str,
    ticker: str,
    market_ticker: str = 'SPY',
    rolling_window: int = 60,
) -> Dict[str, Any]:
    """
    Complete CAPM analysis for a ticker.
    
    Args:
        data_path: Path to price data parquet
        ticker: Ticker to analyze
        market_ticker: Market benchmark ticker
        rolling_window: Window for rolling calculations
    
    Returns:
        Dictionary with full CAPM analysis results
    """
    print(f"Analyzing CAPM for: {ticker}")
    print(f"  Market benchmark: {market_ticker}")
    
    # Load data
    ticker_data = load_returns_data(data_path, ticker)
    market_returns = get_market_returns(data_path, market_ticker)
    
    # Align dates
    aligned = pd.concat([ticker_data['returns'], market_returns], axis=1).dropna()
    aligned.columns = ['asset', 'market']
    
    print(f"  Data points: {len(aligned)}")
    print(f"  Date range: {aligned.index.min().date()} to {aligned.index.max().date()}")
    
    # Compute CAPM
    capm_results = compute_capm_beta(aligned['asset'], aligned['market'])
    
    if 'error' in capm_results:
        return capm_results
    
    # Compute rolling metrics
    rolling_beta = compute_rolling_beta(
        aligned['asset'], aligned['market'],
        window=rolling_window
    )
    rolling_alpha = compute_rolling_alpha(
        aligned['asset'], aligned['market'],
        window=rolling_window
    )
    
    # Summary statistics
    print(f"\nCAPM Results:")
    print(f"  Beta: {capm_results['beta']:.3f} (t={capm_results['t_stat_beta']:.2f}, p={capm_results['p_value_beta']:.4f})")
    print(f"  Alpha (annualized): {capm_results['alpha']*100:.2f}% (t={capm_results['t_stat_alpha']:.2f}, p={capm_results['p_value_alpha']:.4f})")
    print(f"  R-squared: {capm_results['r_squared']:.3f}")
    print(f"  95% CI for Beta: [{capm_results['beta_ci_lower']:.3f}, {capm_results['beta_ci_upper']:.3f}]")
    
    # Add rolling metrics to results
    capm_results['rolling_beta'] = rolling_beta.dropna().to_dict()
    capm_results['rolling_alpha'] = rolling_alpha.dropna().to_dict()
    capm_results['rolling_beta_current'] = float(rolling_beta.iloc[-1]) if len(rolling_beta.dropna()) > 0 else None
    capm_results['rolling_alpha_current'] = float(rolling_alpha.iloc[-1]) if len(rolling_alpha.dropna()) > 0 else None
    
    # Add metadata
    capm_results['ticker'] = ticker
    capm_results['market_ticker'] = market_ticker
    capm_results['rolling_window'] = rolling_window
    
    return capm_results


# =============================================================================
# FAMA-FRENCH FACTOR MODELS
# =============================================================================


def fetch_fama_french_factors(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    frequency: str = 'daily',
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch Fama-French factors from Kenneth French's data library.
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        frequency: 'daily' or 'monthly'
        cache_dir: Directory to cache downloaded data
    
    Returns:
        DataFrame with columns: Mkt-RF, SMB, HML, RMW, CMA, RF
        Index is datetime
    """
    try:
        import pandas_datareader as pdr
    except ImportError:
        raise ImportError("pandas-datareader is required. Install with: pip install pandas-datareader")
    
    # Check cache first
    if cache_dir:
        cache_path = Path(cache_dir) / f"fama_french_5factor_{frequency}.parquet"
        if cache_path.exists():
            print(f"Loading cached Fama-French data from: {cache_path}")
            ff_data = pd.read_parquet(cache_path)
            # Filter by date if specified
            if start_date:
                ff_data = ff_data[ff_data.index >= start_date]
            if end_date:
                ff_data = ff_data[ff_data.index <= end_date]
            return ff_data
    
    # Dataset names
    if frequency == 'daily':
        dataset = 'F-F_Research_Data_5_Factors_2x3_daily'
    else:
        dataset = 'F-F_Research_Data_5_Factors_2x3'
    
    print(f"Fetching Fama-French 5-factor data ({frequency})...")
    
    try:
        ff_raw = pdr.get_data_famafrench(dataset, start=start_date, end=end_date)
        ff_data = ff_raw[0]  # First table contains the factors
    except Exception as e:
        raise RuntimeError(f"Failed to fetch Fama-French data: {e}")
    
    # Convert to decimal returns (FF data is in percentage)
    ff_data = ff_data / 100
    
    # Ensure datetime index
    ff_data.index = pd.to_datetime(ff_data.index, format='%Y%m%d' if frequency == 'daily' else '%Y%m')
    
    # Cache the data
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        ff_data.to_parquet(cache_path / f"fama_french_5factor_{frequency}.parquet")
        print(f"Cached Fama-French data to: {cache_path}")
    
    print(f"  Loaded {len(ff_data)} observations")
    print(f"  Date range: {ff_data.index.min().date()} to {ff_data.index.max().date()}")
    print(f"  Factors: {list(ff_data.columns)}")
    
    return ff_data


def compute_factor_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,
    model: str = '5-factor',
) -> Dict[str, Any]:
    """
    Regress returns on Fama-French factors to compute exposures.
    
    Args:
        returns: Asset returns series with datetime index
        factors: Fama-French factors DataFrame
        model: '3-factor' (Mkt-RF, SMB, HML) or '5-factor' (+ RMW, CMA)
    
    Returns:
        Dictionary with:
        - alpha: Annualized abnormal return
        - exposures: Dict of factor -> beta
        - t_stats: Dict of factor -> t-statistic
        - p_values: Dict of factor -> p-value
        - r_squared: Variance explained by factors
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        raise ImportError("statsmodels is required")
    
    # Select factors based on model
    factor_cols = {
        '3-factor': ['Mkt-RF', 'SMB', 'HML'],
        '5-factor': ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'],
    }
    
    if model not in factor_cols:
        raise ValueError(f"Unknown model: {model}. Choose '3-factor' or '5-factor'")
    
    selected_factors = factor_cols[model]
    
    # Verify factors exist
    missing = [f for f in selected_factors if f not in factors.columns]
    if missing:
        raise ValueError(f"Missing factors: {missing}")
    
    # Align dates
    aligned = pd.concat([returns, factors], axis=1).dropna()
    
    if len(aligned) < 30:
        return {
            'error': f'Insufficient data: {len(aligned)} observations (need >= 30)',
            'n_observations': len(aligned),
        }
    
    # Excess returns (subtract risk-free rate)
    y = aligned.iloc[:, 0] - aligned['RF']
    X = aligned[selected_factors]
    X = sm.add_constant(X)
    
    # Run regression
    model_result = sm.OLS(y, X).fit()
    
    # Extract results
    alpha_daily = model_result.params['const']
    alpha_annualized = alpha_daily * TRADING_DAYS_PER_YEAR
    
    exposures = {}
    t_stats = {}
    p_values = {}
    
    for factor in selected_factors:
        exposures[factor] = float(model_result.params[factor])
        t_stats[factor] = float(model_result.tvalues[factor])
        p_values[factor] = float(model_result.pvalues[factor])
    
    return {
        'model': model,
        'alpha': float(alpha_annualized),
        'alpha_daily': float(alpha_daily),
        'alpha_t_stat': float(model_result.tvalues['const']),
        'alpha_p_value': float(model_result.pvalues['const']),
        'exposures': exposures,
        't_stats': t_stats,
        'p_values': p_values,
        'r_squared': float(model_result.rsquared),
        'adj_r_squared': float(model_result.rsquared_adj),
        'n_observations': len(aligned),
    }


def analyze_ticker_factors(
    data_path: str,
    ticker: str,
    model: str = '5-factor',
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Complete Fama-French factor analysis for a ticker.
    
    Args:
        data_path: Path to price data parquet
        ticker: Ticker to analyze
        model: '3-factor' or '5-factor'
        cache_dir: Directory for caching FF data
    
    Returns:
        Dictionary with factor analysis results
    """
    print(f"\nAnalyzing Fama-French {model} for: {ticker}")
    
    # Load ticker returns
    ticker_data = load_returns_data(data_path, ticker)
    
    # Get date range for FF data
    start_date = ticker_data.index.min().strftime('%Y-%m-%d')
    end_date = ticker_data.index.max().strftime('%Y-%m-%d')
    
    # Fetch Fama-French factors
    if cache_dir is None:
        cache_dir = str(Path(data_path).parent)
    
    ff_factors = fetch_fama_french_factors(
        start_date=start_date,
        end_date=end_date,
        frequency='daily',
        cache_dir=cache_dir,
    )
    
    # Compute factor exposures
    results = compute_factor_exposures(
        ticker_data['returns'],
        ff_factors,
        model=model,
    )
    
    if 'error' in results:
        return results
    
    # Print summary
    print(f"\nFama-French {model} Results:")
    print(f"  Alpha (annualized): {results['alpha']*100:.2f}% (t={results['alpha_t_stat']:.2f}, p={results['alpha_p_value']:.4f})")
    print(f"  R-squared: {results['r_squared']:.3f}")
    print(f"\n  Factor Exposures:")
    for factor, exposure in results['exposures'].items():
        t_stat = results['t_stats'][factor]
        p_val = results['p_values'][factor]
        sig = '*' if p_val < 0.05 else ''
        print(f"    {factor}: {exposure:.3f} (t={t_stat:.2f}, p={p_val:.4f}){sig}")
    
    # Add metadata
    results['ticker'] = ticker
    
    return results


def compare_factor_models(
    data_path: str,
    ticker: str,
    market_ticker: str = 'SPY',
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare CAPM vs Fama-French models for a ticker.
    
    Args:
        data_path: Path to price data
        ticker: Ticker to analyze
        market_ticker: Market benchmark
        cache_dir: Cache directory
    
    Returns:
        Comparison results
    """
    print(f"\n{'='*60}")
    print(f"Factor Model Comparison: {ticker}")
    print(f"{'='*60}")
    
    # CAPM
    capm_results = analyze_ticker_capm(data_path, ticker, market_ticker)
    
    # Remove time series data for comparison
    capm_clean = {k: v for k, v in capm_results.items() 
                  if k not in ['rolling_beta', 'rolling_alpha']}
    
    # Fama-French 3-factor
    ff3_results = analyze_ticker_factors(data_path, ticker, '3-factor', cache_dir)
    
    # Fama-French 5-factor
    ff5_results = analyze_ticker_factors(data_path, ticker, '5-factor', cache_dir)
    
    # Summary comparison
    print(f"\n{'='*60}")
    print("Model Comparison Summary")
    print(f"{'='*60}")
    print(f"\n{'Model':<20} {'Alpha':<12} {'R²':<10} {'Market Beta':<12}")
    print("-" * 54)
    
    if 'error' not in capm_clean:
        print(f"{'CAPM':<20} {capm_clean['alpha']*100:>8.2f}%   {capm_clean['r_squared']:>8.3f}   {capm_clean['beta']:>10.3f}")
    
    if 'error' not in ff3_results:
        print(f"{'FF 3-Factor':<20} {ff3_results['alpha']*100:>8.2f}%   {ff3_results['r_squared']:>8.3f}   {ff3_results['exposures']['Mkt-RF']:>10.3f}")
    
    if 'error' not in ff5_results:
        print(f"{'FF 5-Factor':<20} {ff5_results['alpha']*100:>8.2f}%   {ff5_results['r_squared']:>8.3f}   {ff5_results['exposures']['Mkt-RF']:>10.3f}")
    
    return {
        'ticker': ticker,
        'capm': capm_clean,
        'ff_3factor': ff3_results,
        'ff_5factor': ff5_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Factor model analysis (CAPM and Fama-French)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CAPM analysis for a ticker
  python factor_models.py --ticker AAPL --analysis capm
  
  # Fama-French 5-factor analysis
  python factor_models.py --ticker AAPL --analysis ff --model 5-factor
  
  # Compare all factor models
  python factor_models.py --ticker AAPL --analysis compare
        """
    )
    parser.add_argument(
        "--data-path",
        default="data/cache/daily_2025.parquet",
        help="Path to price data parquet file",
    )
    parser.add_argument(
        "--ticker",
        required=True,
        help="Ticker symbol to analyze",
    )
    parser.add_argument(
        "--analysis",
        choices=["capm", "ff", "compare"],
        default="compare",
        help="Type of analysis: capm, ff (Fama-French), or compare",
    )
    parser.add_argument(
        "--model",
        choices=["3-factor", "5-factor"],
        default="5-factor",
        help="Fama-French model type",
    )
    parser.add_argument(
        "--market-ticker",
        default="SPY",
        help="Market benchmark ticker",
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=60,
        help="Rolling window size for beta calculation",
    )
    parser.add_argument(
        "--output",
        help="Output path for results (JSON)",
    )
    
    args = parser.parse_args()
    
    # Determine cache directory
    cache_dir = str(Path(args.data_path).parent)
    
    # Run analysis
    if args.analysis == 'capm':
        results = analyze_ticker_capm(
            args.data_path,
            args.ticker,
            args.market_ticker,
            args.rolling_window,
        )
    elif args.analysis == 'ff':
        results = analyze_ticker_factors(
            args.data_path,
            args.ticker,
            args.model,
            cache_dir,
        )
    else:  # compare
        results = compare_factor_models(
            args.data_path,
            args.ticker,
            args.market_ticker,
            cache_dir,
        )
    
    # Save results if output path provided
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Make serializable
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


if __name__ == "__main__":
    main()

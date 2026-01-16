"""
Utility functions to filter out non-tradable tickers.

Non-tradable securities include:
- Warrants (tickers ending in W, WS, or containing .W)
- Units (tickers ending in U or containing .U)
- Rights (tickers ending in R or containing .R)
- Preferred shares (tickers with P before last char)
- Test symbols (containing TEST)
- Special securities with unusual characters (., -, ^, +, =)
- Very long tickers (7+ characters)
- Tickers with embedded numbers
"""

from __future__ import annotations

import re

import polars as pl


def is_non_tradable_ticker(ticker: str) -> bool:
    """
    Check if a ticker represents a non-tradable security.
    Returns True if the ticker should be EXCLUDED.
    """
    if not ticker or not isinstance(ticker, str):
        return True
    
    ticker = ticker.strip().upper()
    
    # Empty or very short
    if len(ticker) < 1:
        return True
    
    # Contains special characters (periods, hyphens, carets, etc.)
    if re.search(r'[.\-^+=/#@!$%&*]', ticker):
        return True
    
    # Test symbols
    if 'TEST' in ticker:
        return True
    
    # Very long tickers (7+ chars) are typically special securities
    if len(ticker) >= 7:
        return True
    
    # For 5-6 character tickers, check for warrant/unit/rights suffixes
    if len(ticker) >= 5:
        # Warrants: ends with W, WS, or WT
        if ticker.endswith('W') or ticker.endswith('WS') or ticker.endswith('WT'):
            return True
        
        # Units: ends with U
        if ticker.endswith('U'):
            return True
        
        # Rights: ends with R (but not all - some legit stocks end in R)
        # Be more specific: if 5+ chars and ends in R, likely rights
        if len(ticker) >= 5 and ticker.endswith('R') and ticker[-2].isalpha():
            base = ticker[:-1]
            if len(base) == 4:
                return True
    
    # Preferred shares patterns
    if len(ticker) >= 4:
        # Pattern like XXXPR, XXXPRA, XXXPRB (preferred)
        if 'PR' in ticker and ticker.index('PR') >= 2:
            return True
        # Pattern like XXXP where base is 3+ chars
        if ticker.endswith('P') and len(ticker) >= 4 and ticker[:-1].isalpha():
            return True
    
    # Tickers with numbers in the middle (not at end)
    if re.search(r'[0-9]', ticker[:-1]):
        return True
    
    return False


def filter_tradable_tickers(lf: pl.LazyFrame, ticker_col: str = "ticker") -> pl.LazyFrame:
    """
    Filter a LazyFrame to only include tradable tickers.
    
    Args:
        lf: Input LazyFrame
        ticker_col: Name of the ticker column
    
    Returns:
        Filtered LazyFrame with only tradable tickers
    """
    # Get all unique tickers
    df = lf.collect()
    
    if ticker_col not in df.columns:
        return lf
    
    all_tickers = set(df[ticker_col].unique().to_list())
    non_tradable = {t for t in all_tickers if is_non_tradable_ticker(t)}
    
    print(f"Filtering tickers: {len(all_tickers):,} total, {len(non_tradable):,} non-tradable, {len(all_tickers) - len(non_tradable):,} tradable")
    
    # Filter out non-tradable tickers
    return df.lazy().filter(~pl.col(ticker_col).is_in(list(non_tradable)))


def filter_tradable_tickers_df(df: pl.DataFrame, ticker_col: str = "ticker") -> pl.DataFrame:
    """
    Filter a DataFrame to only include tradable tickers.
    
    Args:
        df: Input DataFrame
        ticker_col: Name of the ticker column
    
    Returns:
        Filtered DataFrame with only tradable tickers
    """
    if ticker_col not in df.columns:
        return df
    
    all_tickers = set(df[ticker_col].unique().to_list())
    non_tradable = {t for t in all_tickers if is_non_tradable_ticker(t)}
    
    print(f"Filtering tickers: {len(all_tickers):,} total, {len(non_tradable):,} non-tradable, {len(all_tickers) - len(non_tradable):,} tradable")
    
    return df.filter(~pl.col(ticker_col).is_in(list(non_tradable)))

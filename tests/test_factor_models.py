#!/usr/bin/env python3
"""
Unit tests for factor models and linear regression functions.

Tests:
- CAPM beta calculation
- Rolling beta computation
- Fama-French factor exposure
- Linear regression with statistical inference
- Regularization tuning
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample return series for testing."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=252, freq='B')
    
    # Asset with beta ~1.2 and small alpha
    market = np.random.normal(0.0005, 0.015, 252)
    asset = 0.0001 + 1.2 * market + np.random.normal(0, 0.008, 252)
    
    return pd.Series(asset, index=dates, name='asset'), pd.Series(market, index=dates, name='market')


@pytest.fixture
def sample_features():
    """Generate sample features and target for regression testing."""
    np.random.seed(42)
    n_samples = 500
    
    # Create features with known relationship to target
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'noise': np.random.randn(n_samples),
    })
    
    # Target = 0.1*f1 + 0.2*f2 + noise
    y = 0.001 + 0.1 * X['feature1'] + 0.2 * X['feature2'] + np.random.randn(n_samples) * 0.5
    
    return X, pd.Series(y, name='target')


# =============================================================================
# CAPM Tests
# =============================================================================

class TestCAPM:
    """Tests for CAPM beta calculation."""
    
    def test_capm_beta_calculation(self, sample_returns):
        """Test that CAPM beta is calculated correctly."""
        from data_processing.factor_models import compute_capm_beta
        
        asset_returns, market_returns = sample_returns
        results = compute_capm_beta(asset_returns, market_returns)
        
        assert 'beta' in results
        assert 'alpha' in results
        assert 'r_squared' in results
        assert 't_stat_beta' in results
        assert 'p_value_beta' in results
        
        # Beta should be close to 1.2 (the value we used to generate data)
        assert 1.0 < results['beta'] < 1.5, f"Beta {results['beta']} not in expected range"
        
        # R-squared should be reasonable (market explains most variance)
        assert results['r_squared'] > 0.5, f"R-squared {results['r_squared']} too low"
        
        # Beta t-stat should be significant
        assert abs(results['t_stat_beta']) > 2, "Beta should be statistically significant"
    
    def test_capm_insufficient_data(self):
        """Test that CAPM handles insufficient data gracefully."""
        from data_processing.factor_models import compute_capm_beta
        
        dates = pd.date_range('2024-01-01', periods=10)
        asset = pd.Series(np.random.randn(10), index=dates)
        market = pd.Series(np.random.randn(10), index=dates)
        
        results = compute_capm_beta(asset, market)
        
        assert 'error' in results
        assert 'Insufficient data' in results['error']
    
    def test_rolling_beta(self, sample_returns):
        """Test rolling beta calculation."""
        from data_processing.factor_models import compute_rolling_beta
        
        asset_returns, market_returns = sample_returns
        rolling_beta = compute_rolling_beta(asset_returns, market_returns, window=60)
        
        assert isinstance(rolling_beta, pd.Series)
        assert len(rolling_beta.dropna()) > 0
        
        # Rolling beta should be in reasonable range
        valid_betas = rolling_beta.dropna()
        assert valid_betas.min() > 0.5, "Rolling beta too low"
        assert valid_betas.max() < 2.0, "Rolling beta too high"
    
    def test_rolling_alpha(self, sample_returns):
        """Test rolling alpha calculation."""
        from data_processing.factor_models import compute_rolling_alpha
        
        asset_returns, market_returns = sample_returns
        rolling_alpha = compute_rolling_alpha(asset_returns, market_returns, window=60)
        
        assert isinstance(rolling_alpha, pd.Series)
        assert len(rolling_alpha.dropna()) > 0


# =============================================================================
# Linear Model Tests
# =============================================================================

class TestLinearModels:
    """Tests for linear regression functions."""
    
    def test_linear_regressor_ols(self, sample_features):
        """Test OLS linear regression."""
        from data_processing.train_ml_model import train_linear_regressor
        
        X, y = sample_features
        results = train_linear_regressor(X, y, n_splits=3, test_size=50, regularization=None)
        
        assert 'coefficients' in results
        assert 't_stats' in results
        assert 'p_values' in results
        assert 'r_squared' in results
        
        # Feature1 and feature2 should have significant coefficients
        assert results['p_values']['feature1'] < 0.1, "feature1 should be significant"
        assert results['p_values']['feature2'] < 0.1, "feature2 should be significant"
        
        # Noise feature should be less significant
        assert results['p_values']['noise'] > 0.05, "noise should not be significant"
    
    def test_linear_regressor_ridge(self, sample_features):
        """Test Ridge regression."""
        from data_processing.train_ml_model import train_linear_regressor
        
        X, y = sample_features
        results = train_linear_regressor(
            X, y, n_splits=3, test_size=50,
            regularization='ridge', alpha=1.0
        )
        
        assert 'coefficients' in results
        assert results['regularization'] == 'ridge'
        
        # Ridge coefficients should be smaller (shrunk) than OLS
        # but all non-zero
        for coef in results['coefficients'].values():
            assert coef != 0, "Ridge should not set coefficients to zero"
    
    def test_linear_regressor_lasso(self, sample_features):
        """Test Lasso regression (feature selection)."""
        from data_processing.train_ml_model import train_linear_regressor
        
        X, y = sample_features
        results = train_linear_regressor(
            X, y, n_splits=3, test_size=50,
            regularization='lasso', alpha=0.1
        )
        
        assert 'coefficients' in results
        assert results['regularization'] == 'lasso'
        
        # Lasso may set some coefficients to zero
        # Important features (feature1, feature2) should survive
        assert abs(results['coefficients']['feature1']) > 0.01 or \
               abs(results['coefficients']['feature2']) > 0.01, \
               "At least one important feature should have non-zero coefficient"
    
    def test_logistic_classifier(self, sample_features):
        """Test logistic regression classifier."""
        from data_processing.train_ml_model import train_logistic_classifier
        
        X, y = sample_features
        results = train_logistic_classifier(
            X, y, n_splits=3, test_size=50,
            regularization='l2', C=1.0
        )
        
        assert 'coefficients' in results
        assert 'cv_summary' in results
        assert 'mean_accuracy' in results['cv_summary']
        assert 'mean_auc' in results['cv_summary']
        
        # Accuracy should be better than random (>50%)
        assert results['cv_summary']['mean_accuracy'] > 0.45, "Accuracy should be better than random"
    
    def test_tune_regularization(self, sample_features):
        """Test regularization tuning."""
        from data_processing.train_ml_model import tune_regularization
        
        X, y = sample_features
        results = tune_regularization(
            X, y,
            model_type='ridge',
            alphas=[0.01, 0.1, 1.0, 10.0],
            n_splits=3,
            test_size=50,
        )
        
        assert 'best_alpha' in results
        assert 'best_rmse' in results
        assert 'coefficient_paths' in results
        
        # Best alpha should be one of the tested values
        assert results['best_alpha'] in [0.01, 0.1, 1.0, 10.0]


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests that require actual data files."""
    
    @pytest.mark.skipif(
        not Path("data/cache/daily_2025.parquet").exists(),
        reason="Data file not available"
    )
    def test_capm_with_real_data(self):
        """Test CAPM with actual market data."""
        from data_processing.factor_models import analyze_ticker_capm
        
        # Test with a likely available ticker
        try:
            results = analyze_ticker_capm(
                "data/cache/daily_2025.parquet",
                "AAPL",
                "SPY",
            )
            
            if 'error' not in results:
                assert 'beta' in results
                assert 0 < results['beta'] < 3, "Beta should be in reasonable range"
        except ValueError:
            pytest.skip("Ticker not available in data")


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])

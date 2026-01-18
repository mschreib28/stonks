#!/usr/bin/env python3
"""
ML model training pipeline for trading signal prediction.

Uses LightGBM with proper time-series cross-validation to predict:
- Next-day return direction (classification)
- Next-day return magnitude (regression)

Features SHAP for model interpretability.

Based on "Machine Learning for Algorithmic Trading" Chapters 11-12.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple, List
import warnings
import json

import numpy as np
import pandas as pd
import polars as pl

# Linear models and preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# Default feature columns (from technical_features.py)
DEFAULT_FEATURES = [
    'rsi_14', 'stoch_k', 'stoch_d', 'roc_10',
    'adx_14', 'bb_pct', 'sma_cross_20_50', 'sma_cross_50_200',
    'atr_14', 'hist_vol_20',
    'rsi_overbought', 'rsi_oversold', 'above_bb_upper', 'below_bb_lower',
    'strong_trend',
]


def load_training_data(
    data_path: str,
    feature_columns: Optional[List[str]] = None,
    target_column: str = 'ret_d',
    min_samples_per_ticker: int = 100,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load and prepare training data.
    
    Args:
        data_path: Path to parquet file with features
        feature_columns: List of feature columns to use
        target_column: Column to predict (returns)
        min_samples_per_ticker: Minimum samples required per ticker
    
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    print(f"Loading data from: {data_path}")
    df = pl.read_parquet(data_path)
    pdf = df.to_pandas()
    
    print(f"  Loaded {len(pdf):,} rows, {len(pdf.columns)} columns")
    
    # Filter tickers with enough data
    ticker_counts = pdf['ticker'].value_counts()
    valid_tickers = ticker_counts[ticker_counts >= min_samples_per_ticker].index
    pdf = pdf[pdf['ticker'].isin(valid_tickers)]
    print(f"  After filtering: {len(pdf):,} rows, {pdf['ticker'].nunique()} tickers")
    
    # Determine feature columns
    if feature_columns is None:
        # Use default features that exist in data
        feature_columns = [c for c in DEFAULT_FEATURES if c in pdf.columns]
    
    # Verify columns exist
    missing_features = [c for c in feature_columns if c not in pdf.columns]
    if missing_features:
        print(f"  Warning: Missing features: {missing_features}")
        feature_columns = [c for c in feature_columns if c in pdf.columns]
    
    # Create target: next-day return (shifted)
    if target_column not in pdf.columns:
        # Compute from close prices
        pdf = pdf.sort_values(['ticker', 'date'])
        pdf['target'] = pdf.groupby('ticker')['close'].pct_change().shift(-1)
    else:
        pdf = pdf.sort_values(['ticker', 'date'])
        pdf['target'] = pdf.groupby('ticker')[target_column].shift(-1)
    
    # Drop rows with NaN in features or target
    pdf = pdf.dropna(subset=feature_columns + ['target'])
    
    print(f"  Features: {len(feature_columns)}")
    print(f"  Final samples: {len(pdf):,}")
    
    X = pdf[feature_columns]
    y = pdf['target']
    
    return X, y, pdf


class TimeSeriesSplit:
    """
    Time-series cross-validation with expanding or rolling window.
    
    Ensures no lookahead bias by always training on past data only.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 60,  # trading days
        gap: int = 1,  # gap between train and test
        expanding: bool = True,
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding
    
    def split(self, X, y=None, groups=None):
        """Generate train/test indices."""
        n_samples = len(X)
        
        # Calculate sizes
        min_train_size = n_samples // (self.n_splits + 1)
        
        for i in range(self.n_splits):
            if self.expanding:
                # Expanding window
                train_end = min_train_size + i * self.test_size
            else:
                # Rolling window
                train_start = i * self.test_size
                train_end = train_start + min_train_size
            
            test_start = train_end + self.gap
            test_end = min(test_start + self.test_size, n_samples)
            
            if test_end <= test_start:
                break
            
            if self.expanding:
                train_indices = np.arange(0, train_end)
            else:
                train_indices = np.arange(train_start, train_end)
            
            test_indices = np.arange(test_start, test_end)
            
            yield train_indices, test_indices


def train_lightgbm_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 60,
    params: Optional[dict] = None,
) -> dict:
    """
    Train LightGBM classifier for return direction prediction.
    
    Args:
        X: Feature DataFrame
        y: Target Series (returns - will be converted to direction)
        n_splits: Number of CV splits
        test_size: Test size in samples
        params: LightGBM parameters
    
    Returns:
        Dictionary with model, metrics, and predictions
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    except ImportError:
        raise ImportError("lightgbm and scikit-learn are required")
    
    # Convert returns to direction (1 = up, 0 = down)
    y_binary = (y > 0).astype(int)
    
    # Default parameters
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 100,
        }
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    cv_results = []
    feature_importance = np.zeros(len(X.columns))
    all_predictions = []
    all_actuals = []
    
    print(f"\nTraining LightGBM Classifier with {n_splits}-fold time-series CV...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        cv_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        })
        
        feature_importance += model.feature_importances_
        all_predictions.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        
        print(f"  Fold {fold + 1}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
    
    # Average feature importance
    feature_importance /= n_splits
    
    # Train final model on all data
    final_model = lgb.LGBMClassifier(**params)
    final_model.fit(X, y_binary)
    
    # Create feature importance dict
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance,
    }).sort_values('importance', ascending=False)
    
    # Summary metrics
    cv_df = pd.DataFrame(cv_results)
    
    return {
        'model': final_model,
        'model_type': 'classifier',
        'target': 'direction',
        'cv_results': cv_results,
        'cv_summary': {
            'mean_accuracy': float(cv_df['accuracy'].mean()),
            'std_accuracy': float(cv_df['accuracy'].std()),
            'mean_auc': float(cv_df['auc'].mean()),
            'std_auc': float(cv_df['auc'].std()),
            'mean_f1': float(cv_df['f1'].mean()),
        },
        'feature_importance': importance_df.to_dict('records'),
        'parameters': params,
        'predictions': {
            'predicted': all_predictions,
            'actual': all_actuals,
        },
    }


def train_lightgbm_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 60,
    params: Optional[dict] = None,
) -> dict:
    """
    Train LightGBM regressor for return magnitude prediction.
    
    Args:
        X: Feature DataFrame
        y: Target Series (returns)
        n_splits: Number of CV splits
        test_size: Test size in samples
        params: LightGBM parameters
    
    Returns:
        Dictionary with model, metrics, and predictions
    """
    try:
        import lightgbm as lgb
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        raise ImportError("lightgbm and scikit-learn are required")
    
    # Default parameters
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 100,
        }
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    cv_results = []
    feature_importance = np.zeros(len(X.columns))
    all_predictions = []
    all_actuals = []
    
    print(f"\nTraining LightGBM Regressor with {n_splits}-fold time-series CV...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(10, verbose=False)],
        )
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Directional accuracy (sign of prediction matches sign of actual)
        direction_accuracy = ((y_pred > 0) == (y_test > 0)).mean()
        
        cv_results.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        })
        
        feature_importance += model.feature_importances_
        all_predictions.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        
        print(f"  Fold {fold + 1}: RMSE={rmse:.5f}, R2={r2:.3f}, Dir Acc={direction_accuracy:.3f}")
    
    # Average feature importance
    feature_importance /= n_splits
    
    # Train final model on all data
    final_model = lgb.LGBMRegressor(**params)
    final_model.fit(X, y)
    
    # Create feature importance dict
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance,
    }).sort_values('importance', ascending=False)
    
    # Summary metrics
    cv_df = pd.DataFrame(cv_results)
    
    return {
        'model': final_model,
        'model_type': 'regressor',
        'target': 'returns',
        'cv_results': cv_results,
        'cv_summary': {
            'mean_rmse': float(cv_df['rmse'].mean()),
            'std_rmse': float(cv_df['rmse'].std()),
            'mean_r2': float(cv_df['r2'].mean()),
            'mean_direction_accuracy': float(cv_df['direction_accuracy'].mean()),
        },
        'feature_importance': importance_df.to_dict('records'),
        'parameters': params,
        'predictions': {
            'predicted': all_predictions,
            'actual': all_actuals,
        },
    }


# =============================================================================
# LINEAR MODELS (Book 1 Chapter 3)
# =============================================================================


def train_linear_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 60,
    regularization: Optional[str] = None,
    alpha: float = 1.0,
) -> dict:
    """
    Train linear regression model with time-series CV and statistical inference.
    
    Uses statsmodels for OLS to get t-statistics and p-values for coefficients.
    Supports Ridge/Lasso regularization via sklearn.
    
    Args:
        X: Feature DataFrame
        y: Target Series (returns)
        n_splits: Number of CV splits
        test_size: Test size in samples
        regularization: None, 'ridge', 'lasso', or 'elasticnet'
        alpha: Regularization strength (higher = more regularization)
    
    Returns:
        Dictionary with model, coefficients, t-stats, p-values, and CV metrics
    """
    try:
        import statsmodels.api as sm
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        raise ImportError("statsmodels and scikit-learn are required")
    
    # Standardize features (important for regularization and coefficient comparison)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    cv_results = []
    all_predictions = []
    all_actuals = []
    
    reg_name = regularization if regularization else 'OLS'
    print(f"\nTraining Linear Regressor ({reg_name}) with {n_splits}-fold time-series CV...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Select model based on regularization
        if regularization == 'ridge':
            model = Ridge(alpha=alpha)
        elif regularization == 'lasso':
            model = Lasso(alpha=alpha, max_iter=10000)
        elif regularization == 'elasticnet':
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
        else:
            model = LinearRegression()
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        direction_accuracy = ((y_pred > 0) == (y_test > 0)).mean()
        
        cv_results.append({
            'fold': fold + 1,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        })
        
        all_predictions.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        
        print(f"  Fold {fold + 1}: RMSE={rmse:.5f}, R2={r2:.3f}, Dir Acc={direction_accuracy:.3f}")
    
    # Train final model on all data for coefficient analysis
    if regularization == 'ridge':
        final_model = Ridge(alpha=alpha)
    elif regularization == 'lasso':
        final_model = Lasso(alpha=alpha, max_iter=10000)
    elif regularization == 'elasticnet':
        final_model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)
    else:
        final_model = LinearRegression()
    
    final_model.fit(X_scaled, y)
    
    # Get coefficients
    coefficients = dict(zip(X.columns, final_model.coef_))
    
    # For OLS, use statsmodels for statistical inference
    t_stats = {}
    p_values = {}
    
    if regularization is None:
        # Use statsmodels OLS for t-stats and p-values
        X_with_const = sm.add_constant(X_scaled)
        ols_model = sm.OLS(y, X_with_const).fit()
        
        for i, col in enumerate(X.columns):
            t_stats[col] = float(ols_model.tvalues[i + 1])  # +1 to skip constant
            p_values[col] = float(ols_model.pvalues[i + 1])
        
        r_squared = ols_model.rsquared
        adj_r_squared = ols_model.rsquared_adj
        f_statistic = float(ols_model.fvalue)
        f_pvalue = float(ols_model.f_pvalue)
    else:
        # For regularized models, no closed-form t-stats
        # Approximate significance by coefficient magnitude
        r_squared = final_model.score(X_scaled, y)
        n = len(y)
        p = len(X.columns)
        adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        f_statistic = None
        f_pvalue = None
    
    # Create feature importance (absolute coefficient values)
    importance_df = pd.DataFrame({
        'feature': list(coefficients.keys()),
        'coefficient': list(coefficients.values()),
        'abs_coefficient': [abs(v) for v in coefficients.values()],
    }).sort_values('abs_coefficient', ascending=False)
    
    # Add t-stats and p-values if available
    if t_stats:
        importance_df['t_stat'] = importance_df['feature'].map(t_stats)
        importance_df['p_value'] = importance_df['feature'].map(p_values)
        importance_df['significant'] = importance_df['p_value'] < 0.05
    
    # Summary metrics
    cv_df = pd.DataFrame(cv_results)
    
    return {
        'model': final_model,
        'scaler': scaler,
        'model_type': 'linear_regressor',
        'regularization': regularization,
        'alpha': alpha,
        'target': 'returns',
        'cv_results': cv_results,
        'cv_summary': {
            'mean_rmse': float(cv_df['rmse'].mean()),
            'std_rmse': float(cv_df['rmse'].std()),
            'mean_r2': float(cv_df['r2'].mean()),
            'mean_direction_accuracy': float(cv_df['direction_accuracy'].mean()),
        },
        'coefficients': coefficients,
        't_stats': t_stats if t_stats else None,
        'p_values': p_values if p_values else None,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_statistic': f_statistic,
        'f_pvalue': f_pvalue,
        'feature_importance': importance_df.to_dict('records'),
        'predictions': {
            'predicted': all_predictions,
            'actual': all_actuals,
        },
    }


def train_logistic_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    test_size: int = 60,
    regularization: str = 'l2',
    C: float = 1.0,
) -> dict:
    """
    Train logistic regression for return direction prediction.
    
    Args:
        X: Feature DataFrame
        y: Target Series (returns - will be converted to direction)
        n_splits: Number of CV splits
        test_size: Test size in samples
        regularization: 'l1', 'l2', 'elasticnet', or 'none'
        C: Inverse of regularization strength (lower = more regularization)
    
    Returns:
        Dictionary with model, coefficients, and CV metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Convert returns to direction (1 = up, 0 = down)
    y_binary = (y > 0).astype(int)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    cv_results = []
    all_predictions = []
    all_actuals = []
    all_probas = []
    
    # Map regularization to sklearn penalty
    penalty_map = {
        'l1': 'l1',
        'l2': 'l2',
        'elasticnet': 'elasticnet',
        'none': None,
    }
    penalty = penalty_map.get(regularization, 'l2')
    
    # Solver selection based on penalty
    if penalty == 'l1' or penalty == 'elasticnet':
        solver = 'saga'
    else:
        solver = 'lbfgs'
    
    print(f"\nTraining Logistic Classifier ({regularization}) with {n_splits}-fold time-series CV...")
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y_binary.iloc[train_idx], y_binary.iloc[test_idx]
        
        # Create model
        if penalty == 'elasticnet':
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                l1_ratio=0.5,
                max_iter=1000,
            )
        elif penalty is None:
            model = LogisticRegression(
                penalty=None,
                solver='lbfgs',
                max_iter=1000,
            )
        else:
            model = LogisticRegression(
                penalty=penalty,
                C=C,
                solver=solver,
                max_iter=1000,
            )
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        
        cv_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        })
        
        all_predictions.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        all_probas.extend(y_pred_proba.tolist())
        
        print(f"  Fold {fold + 1}: Accuracy={accuracy:.3f}, AUC={auc:.3f}")
    
    # Train final model on all data
    if penalty == 'elasticnet':
        final_model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            l1_ratio=0.5,
            max_iter=1000,
        )
    elif penalty is None:
        final_model = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=1000,
        )
    else:
        final_model = LogisticRegression(
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=1000,
        )
    
    final_model.fit(X_scaled, y_binary)
    
    # Get coefficients
    coefficients = dict(zip(X.columns, final_model.coef_[0]))
    
    # Create feature importance (absolute coefficient values)
    importance_df = pd.DataFrame({
        'feature': list(coefficients.keys()),
        'coefficient': list(coefficients.values()),
        'abs_coefficient': [abs(v) for v in coefficients.values()],
    }).sort_values('abs_coefficient', ascending=False)
    
    # Summary metrics
    cv_df = pd.DataFrame(cv_results)
    
    return {
        'model': final_model,
        'scaler': scaler,
        'model_type': 'logistic_classifier',
        'regularization': regularization,
        'C': C,
        'target': 'direction',
        'cv_results': cv_results,
        'cv_summary': {
            'mean_accuracy': float(cv_df['accuracy'].mean()),
            'std_accuracy': float(cv_df['accuracy'].std()),
            'mean_auc': float(cv_df['auc'].mean()),
            'std_auc': float(cv_df['auc'].std()),
            'mean_f1': float(cv_df['f1'].mean()),
        },
        'coefficients': coefficients,
        'feature_importance': importance_df.to_dict('records'),
        'predictions': {
            'predicted': all_predictions,
            'actual': all_actuals,
            'probabilities': all_probas,
        },
    }


def tune_regularization(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'ridge',
    alphas: Optional[List[float]] = None,
    n_splits: int = 5,
    test_size: int = 60,
) -> dict:
    """
    Tune regularization strength using time-series CV.
    
    Performs grid search over alpha values and returns optimal parameters,
    coefficient paths, and cross-validation scores.
    
    Args:
        X: Feature DataFrame
        y: Target Series (returns)
        model_type: 'ridge', 'lasso', or 'elasticnet'
        alphas: List of alpha values to test (default: logspace from 1e-4 to 1e4)
        n_splits: Number of CV splits
        test_size: Test size in samples
    
    Returns:
        Dictionary with best alpha, CV scores, and coefficient paths
    """
    from sklearn.metrics import mean_squared_error, r2_score
    
    if alphas is None:
        alphas = np.logspace(-4, 4, 20).tolist()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Time-series CV
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    
    print(f"\nTuning {model_type} regularization with {len(alphas)} alpha values...")
    
    results_by_alpha = []
    coefficient_paths = {col: [] for col in X.columns}
    
    for alpha in alphas:
        # Select model
        if model_type == 'ridge':
            model_class = Ridge
            model_kwargs = {'alpha': alpha}
        elif model_type == 'lasso':
            model_class = Lasso
            model_kwargs = {'alpha': alpha, 'max_iter': 10000}
        elif model_type == 'elasticnet':
            model_class = ElasticNet
            model_kwargs = {'alpha': alpha, 'l1_ratio': 0.5, 'max_iter': 10000}
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        fold_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model = model_class(**model_kwargs)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            fold_scores.append({'rmse': rmse, 'r2': r2})
        
        # Compute mean scores
        mean_rmse = np.mean([s['rmse'] for s in fold_scores])
        mean_r2 = np.mean([s['r2'] for s in fold_scores])
        
        results_by_alpha.append({
            'alpha': alpha,
            'mean_rmse': mean_rmse,
            'mean_r2': mean_r2,
        })
        
        # Train on full data to get coefficients for this alpha
        full_model = model_class(**model_kwargs)
        full_model.fit(X_scaled, y)
        
        for i, col in enumerate(X.columns):
            coefficient_paths[col].append(float(full_model.coef_[i]))
    
    # Find best alpha (lowest RMSE)
    best_result = min(results_by_alpha, key=lambda x: x['mean_rmse'])
    best_alpha = best_result['alpha']
    
    print(f"  Best alpha: {best_alpha:.6f} (RMSE: {best_result['mean_rmse']:.5f}, R2: {best_result['mean_r2']:.3f})")
    
    # Count non-zero coefficients at best alpha (for Lasso)
    if model_type == 'lasso':
        model_kwargs['alpha'] = best_alpha
        best_model = model_class(**model_kwargs)
        best_model.fit(X_scaled, y)
        n_nonzero = np.sum(best_model.coef_ != 0)
        print(f"  Non-zero coefficients at best alpha: {n_nonzero}/{len(X.columns)}")
    
    # Identify features that survive L1 regularization (Lasso)
    surviving_features = []
    if model_type == 'lasso':
        for col in X.columns:
            # Check if coefficient is non-zero at best alpha
            alpha_idx = alphas.index(best_alpha)
            if coefficient_paths[col][alpha_idx] != 0:
                surviving_features.append(col)
    
    return {
        'model_type': model_type,
        'best_alpha': best_alpha,
        'best_rmse': best_result['mean_rmse'],
        'best_r2': best_result['mean_r2'],
        'alphas': alphas,
        'results_by_alpha': results_by_alpha,
        'coefficient_paths': coefficient_paths,
        'surviving_features': surviving_features if model_type == 'lasso' else None,
    }


def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> dict:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
        max_samples: Maximum samples for SHAP computation
    
    Returns:
        Dictionary with SHAP values and summary
    """
    try:
        import shap
    except ImportError:
        return {'error': 'shap package not installed'}
    
    print("\nComputing SHAP values for interpretability...")
    
    # Sample data if too large
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=42)
    else:
        X_sample = X
    
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Compute mean absolute SHAP value per feature
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    shap_importance = pd.DataFrame({
        'feature': X.columns,
        'mean_abs_shap': mean_shap,
    }).sort_values('mean_abs_shap', ascending=False)
    
    return {
        'shap_importance': shap_importance.to_dict('records'),
        'expected_value': float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[1]),
        'num_samples': len(X_sample),
    }


def save_model(
    results: dict,
    output_dir: str,
    model_name: str = 'lgbm_model',
) -> None:
    """
    Save model and results.
    
    Args:
        results: Training results dictionary
        output_dir: Output directory
        model_name: Name for the model files
    """
    import joblib
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = out_dir / f"{model_name}.joblib"
    joblib.dump(results['model'], model_path)
    print(f"Model saved to: {model_path}")
    
    # Save results (without model object)
    results_to_save = {k: v for k, v in results.items() if k != 'model'}
    
    results_path = out_dir / f"{model_name}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"Results saved to: {results_path}")


def train_and_evaluate(
    data_path: str,
    model_type: str = 'classifier',
    feature_columns: Optional[List[str]] = None,
    n_splits: int = 5,
    test_size: int = 60,
    compute_shap: bool = True,
    output_dir: Optional[str] = None,
    # Linear model options
    regularization: Optional[str] = None,
    alpha: float = 1.0,
    tune_alpha: bool = False,
) -> dict:
    """
    Complete training and evaluation pipeline.
    
    Args:
        data_path: Path to feature data
        model_type: 'classifier', 'regressor', 'linear', or 'logistic'
        feature_columns: List of features to use
        n_splits: Number of CV folds
        test_size: Test size per fold
        compute_shap: Whether to compute SHAP values (LightGBM only)
        output_dir: Directory to save model and results
        regularization: For linear models: None, 'ridge', 'lasso', 'elasticnet'
        alpha: Regularization strength for linear models
        tune_alpha: Whether to tune regularization strength via grid search
    
    Returns:
        Training results dictionary
    """
    # Load data
    X, y, df = load_training_data(data_path, feature_columns)
    
    # Handle alpha tuning for linear models
    if tune_alpha and model_type == 'linear' and regularization:
        print("\n" + "=" * 60)
        print("Tuning Regularization Strength")
        print("=" * 60)
        tune_results = tune_regularization(
            X, y,
            model_type=regularization,
            n_splits=n_splits,
            test_size=test_size,
        )
        alpha = tune_results['best_alpha']
        print(f"\nUsing tuned alpha: {alpha}")
        
        # Save tuning results
        if output_dir:
            import json
            out_dir = Path(output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            tune_path = out_dir / f"tune_{regularization}_results.json"
            with open(tune_path, 'w') as f:
                # Remove coefficient_paths for smaller file
                tune_save = {k: v for k, v in tune_results.items() if k != 'coefficient_paths'}
                json.dump(tune_save, f, indent=2, default=str)
            print(f"Tuning results saved to: {tune_path}")
    
    # Train model based on type
    if model_type == 'classifier':
        results = train_lightgbm_classifier(X, y, n_splits, test_size)
        model_name = 'lgbm_classifier'
    elif model_type == 'regressor':
        results = train_lightgbm_regressor(X, y, n_splits, test_size)
        model_name = 'lgbm_regressor'
    elif model_type == 'linear':
        results = train_linear_regressor(
            X, y, n_splits, test_size,
            regularization=regularization,
            alpha=alpha,
        )
        reg_suffix = f"_{regularization}" if regularization else "_ols"
        model_name = f'linear_regressor{reg_suffix}'
    elif model_type == 'logistic':
        reg = regularization if regularization else 'l2'
        results = train_logistic_classifier(
            X, y, n_splits, test_size,
            regularization=reg,
            C=1.0 / alpha if alpha > 0 else 1.0,  # C is inverse of alpha
        )
        model_name = f'logistic_classifier_{reg}'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Compute SHAP values (only for tree-based models)
    if compute_shap and model_type in ['classifier', 'regressor']:
        shap_results = compute_shap_values(results['model'], X)
        results['shap'] = shap_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    if regularization:
        print(f"Regularization: {regularization} (alpha={alpha})")
    print(f"Features: {len(X.columns)}")
    print(f"Samples: {len(X):,}")
    
    cv_summary = results.get('cv_summary', {})
    if model_type in ['classifier', 'logistic']:
        print(f"\nCross-Validation Results:")
        print(f"  Mean Accuracy: {cv_summary.get('mean_accuracy', 0):.3f} (+/- {cv_summary.get('std_accuracy', 0):.3f})")
        print(f"  Mean AUC: {cv_summary.get('mean_auc', 0):.3f} (+/- {cv_summary.get('std_auc', 0):.3f})")
        print(f"  Mean F1: {cv_summary.get('mean_f1', 0):.3f}")
    else:
        print(f"\nCross-Validation Results:")
        print(f"  Mean RMSE: {cv_summary.get('mean_rmse', 0):.5f} (+/- {cv_summary.get('std_rmse', 0):.5f})")
        print(f"  Mean R2: {cv_summary.get('mean_r2', 0):.3f}")
        print(f"  Direction Accuracy: {cv_summary.get('mean_direction_accuracy', 0):.3f}")
    
    # Print feature importance
    importance_key = 'importance' if 'importance' in results['feature_importance'][0] else 'abs_coefficient'
    print(f"\nTop 10 Features by {'Importance' if importance_key == 'importance' else 'Coefficient Magnitude'}:")
    for item in results['feature_importance'][:10]:
        value = item.get(importance_key, item.get('abs_coefficient', 0))
        print(f"  {item['feature']}: {value:.4f}")
    
    # Print statistical significance for linear models
    if model_type == 'linear' and results.get('t_stats'):
        print(f"\nSignificant Features (p < 0.05):")
        sig_features = [f for f in results['feature_importance'] if f.get('significant', False)]
        for item in sig_features[:10]:
            print(f"  {item['feature']}: coef={item['coefficient']:.4f}, t={item['t_stat']:.2f}, p={item['p_value']:.4f}")
    
    if 'shap' in results and 'shap_importance' in results['shap']:
        print(f"\nTop 10 Features by SHAP:")
        for item in results['shap']['shap_importance'][:10]:
            print(f"  {item['feature']}: {item['mean_abs_shap']:.6f}")
    
    # Save model and results
    if output_dir:
        save_model(results, output_dir, model_name)
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ML models for trading signal prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train LightGBM classifier (default)
  python train_ml_model.py
  
  # Train linear regression with OLS
  python train_ml_model.py --model-type linear
  
  # Train Ridge regression with alpha tuning
  python train_ml_model.py --model-type linear --regularization ridge --tune-alpha
  
  # Train Lasso for feature selection
  python train_ml_model.py --model-type linear --regularization lasso --alpha 0.01
  
  # Train logistic regression for direction prediction
  python train_ml_model.py --model-type logistic --regularization l2
        """
    )
    parser.add_argument(
        "--data-path",
        default="data/cache/technical_features.parquet",
        help="Path to feature data parquet file",
    )
    parser.add_argument(
        "--model-type",
        choices=["classifier", "regressor", "linear", "logistic"],
        default="classifier",
        help="Type of model: classifier/regressor (LightGBM), linear (regression), logistic (classification)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of time-series CV folds",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=60,
        help="Test size per fold (trading days)",
    )
    parser.add_argument(
        "--no-shap",
        action="store_true",
        help="Skip SHAP value computation (LightGBM only)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/cache/models",
        help="Directory to save model and results",
    )
    # Linear model options
    parser.add_argument(
        "--regularization",
        choices=["ridge", "lasso", "elasticnet", "l1", "l2", "none"],
        default=None,
        help="Regularization type for linear/logistic models",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength (higher = more regularization)",
    )
    parser.add_argument(
        "--tune-alpha",
        action="store_true",
        help="Tune regularization strength via grid search",
    )
    
    args = parser.parse_args()
    
    # Map regularization for logistic models
    reg = args.regularization
    if reg == 'none':
        reg = None
    
    train_and_evaluate(
        data_path=args.data_path,
        model_type=args.model_type,
        n_splits=args.n_splits,
        test_size=args.test_size,
        compute_shap=not args.no_shap,
        output_dir=args.output_dir,
        regularization=reg,
        alpha=args.alpha,
        tune_alpha=args.tune_alpha,
    )


if __name__ == "__main__":
    main()

# Linear Models & Cross-Validation (Tier 1)

**Status**: 🟡 Partial (Time-series CV complete, linear models in progress)  
**Module**: `data_processing/train_ml_model.py`  
**Book Reference**: Machine Learning for Algorithmic Trading, Chapter 3

## Why This Matters for Short-Term Trading

Linear models provide **interpretable baselines** that help you:

1. **Understand what drives returns**: Coefficients tell you which factors matter
2. **Avoid overfitting**: Simpler models generalize better
3. **Benchmark ML models**: If LightGBM doesn't beat linear, something's wrong
4. **Risk factor analysis**: CAPM and Fama-French reveal portfolio exposures

Time-series cross-validation prevents **lookahead bias**—the #1 mistake that makes backtests look better than reality.

## Time-Series Cross-Validation (✅ Implemented)

### The Problem with Standard CV

Standard K-fold cross-validation randomly splits data:

```
[Train] [Test] [Train] [Test] [Train]
```

**This leaks future information into training!** If you train on 2025 data and test on 2024 data, you're cheating.

### Time-Series Split

Stonks uses proper temporal ordering:

```
Window 1: [Train: Jan-Jun] → [Test: Jul]
Window 2: [Train: Jan-Jul] → [Test: Aug]
Window 3: [Train: Jan-Aug] → [Test: Sep]
...
```

### Implementation

```python
class TimeSeriesSplit:
    """
    Time-series cross-validation with expanding or rolling window.
    Ensures no lookahead bias by always training on past data only.
    """
    def __init__(
        self, 
        n_splits=5,      # Number of train/test splits
        test_size=60,    # Test period in days
        gap=1,           # Gap between train and test (avoid leakage)
        expanding=True   # True = expanding window, False = rolling
    ):
```

### Expanding vs Rolling Window

**Expanding Window** (default, recommended):
```
Fold 1: [Train: 100 days] → [Test: 60 days]
Fold 2: [Train: 160 days] → [Test: 60 days]
Fold 3: [Train: 220 days] → [Test: 60 days]
```
- Uses all historical data
- More stable estimates
- Better for longer-term patterns

**Rolling Window**:
```
Fold 1: [Train: 100 days] → [Test: 60 days]
Fold 2: [Train: 100 days] → [Test: 60 days]  (shifted)
Fold 3: [Train: 100 days] → [Test: 60 days]  (shifted)
```
- Fixed lookback period
- Adapts to regime changes
- Better for non-stationary markets

### How to Use

```python
from train_ml_model import TimeSeriesSplit

ts_cv = TimeSeriesSplit(n_splits=5, test_size=60, expanding=True)

for train_idx, test_idx in ts_cv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    # Evaluate out-of-sample
```

## Linear Regression Baseline (🔴 Planned)

### Why Linear Models?

| Benefit | Explanation |
|---------|-------------|
| **Interpretable** | Coefficient = marginal effect on returns |
| **Fast** | Trains in seconds vs minutes for GBMs |
| **Benchmark** | If complex models don't beat linear, they're overfitting |
| **Statistical inference** | t-stats tell you which features are significant |

### Planned Implementation

```python
def train_linear_regressor(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    regularization: str = None,  # 'ridge', 'lasso', or None
    alpha: float = 1.0,
) -> dict:
    """
    Train linear regression with time-series CV.
    
    Returns:
        - coefficients: Feature importances with signs
        - t_stats: Statistical significance
        - cv_r2: Out-of-sample R²
        - cv_rmse: Out-of-sample RMSE
    """
```

### Expected Output

```python
{
    'coefficients': {
        'rsi_14': -0.0015,       # Negative = oversold predicts positive returns
        'adx_14': 0.0008,        # Positive = strong trends continue
        'bb_pct': -0.0012,       # Negative = below band predicts bounce
        'hist_vol_20': -0.0003,  # Negative = high vol predicts lower returns
    },
    't_stats': {
        'rsi_14': -3.21,         # Significant (>2 or <-2)
        'adx_14': 1.85,          # Borderline
        'bb_pct': -2.45,         # Significant
        'hist_vol_20': -0.92,    # Not significant
    },
    'cv_r2': 0.02,               # Low R² is typical for returns
    'cv_rmse': 0.025,            # ~2.5% daily error
}
```

### Interpreting Results

**Coefficients**:
- Sign: Direction of relationship
- Magnitude: Strength of effect (standardize features first)
- Consistency across folds: Robust signal

**T-statistics**:
- |t| > 2: Statistically significant at 5% level
- |t| > 3: Highly significant
- |t| < 1: Not meaningful

**R²**:
- For daily returns, R² > 0.01 is good
- R² > 0.05 is excellent (and suspicious—check for leakage)
- Low R² doesn't mean useless—small edge × many trades = profit

## Ridge & Lasso Regularization (🔴 Planned)

### Why Regularization?

- **Ridge (L2)**: Shrinks coefficients toward zero, handles multicollinearity
- **Lasso (L1)**: Sets some coefficients exactly to zero (feature selection)
- **Elastic Net**: Combination of both

### When to Use

| Method | When |
|--------|------|
| OLS | Few features, no multicollinearity |
| Ridge | Many correlated features |
| Lasso | Want automatic feature selection |
| Elastic Net | Many correlated features + sparsity |

### Planned Implementation

```python
def tune_regularization(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'ridge',
    alphas: list = [0.001, 0.01, 0.1, 1, 10],
    n_splits: int = 5,
) -> dict:
    """
    Tune regularization strength using time-series CV.
    
    Returns:
        - best_alpha: Optimal regularization strength
        - cv_scores: Performance at each alpha
        - final_coefficients: With optimal alpha
    """
```

## CAPM Beta Calculation (🔴 Planned)

### What is CAPM?

Capital Asset Pricing Model explains returns as:

```
Return_stock = Alpha + Beta × Return_market + Error
```

- **Beta**: Sensitivity to market movements
- **Alpha**: Excess return not explained by market (skill or luck)

### For Short-Term Trading

| Beta | Meaning | Trading Implication |
|------|---------|---------------------|
| β = 1.0 | Moves with market | Neutral market exposure |
| β > 1.5 | More volatile than market | Amplifies market moves |
| β < 0.5 | Less volatile than market | Defensive, smaller moves |
| β < 0 | Inverse to market | Hedge position |

### Planned Implementation

```python
def compute_capm_beta(
    returns: pd.Series,        # Stock returns
    market_returns: pd.Series, # SPY or benchmark
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Compute CAPM beta and alpha.
    
    Returns:
        - beta: Market sensitivity
        - alpha: Excess return (annualized)
        - r_squared: % of variance explained by market
        - t_stat_beta: Significance of beta
    """
```

### Using Beta

```python
# For portfolio construction
portfolio_beta = sum(weight_i * beta_i)

# For position sizing (beta-adjusted)
adjusted_position = base_position / stock_beta
```

## Fama-French Factor Exposure (🔴 Planned)

### What are Fama-French Factors?

Academic factors that explain stock returns:

| Factor | Measures |
|--------|----------|
| **Mkt-RF** | Market excess return |
| **SMB** | Small minus Big (size factor) |
| **HML** | High minus Low (value factor) |
| **RMW** | Robust minus Weak (profitability) |
| **CMA** | Conservative minus Aggressive (investment) |

### For Short-Term Trading

Understanding factor exposure helps you:
1. **Diversify**: Don't bet on one factor
2. **Explain returns**: Was it skill or factor exposure?
3. **Hedge**: Neutralize unwanted exposures

### Planned Implementation

```python
def compute_factor_exposures(
    returns: pd.Series,
    factors: pd.DataFrame,  # From Ken French library
    model: str = '5-factor',
) -> dict:
    """
    Regress returns on Fama-French factors.
    
    Returns:
        - exposures: Beta to each factor
        - alpha: Unexplained return (potential skill)
        - r_squared: % explained by factors
        - t_stats: Significance of each exposure
    """
```

### Example Output

```python
{
    'exposures': {
        'Mkt-RF': 1.25,   # Higher market beta
        'SMB': 0.85,      # Small-cap tilt (expected for our universe)
        'HML': -0.30,     # Growth tilt (negative value exposure)
        'RMW': 0.15,      # Slight quality tilt
        'CMA': -0.10,     # Neutral investment
    },
    'alpha': 0.12,        # 12% unexplained annualized return
    'r_squared': 0.65,    # 65% explained by factors
}
```

## Logistic Regression for Direction (🔴 Planned)

### Why Classification?

Sometimes you just need to know: **up or down?**

- Simpler target than return magnitude
- Higher accuracy achievable (50% baseline vs 0% for regression)
- Directly maps to trading decision

### Planned Implementation

```python
def train_logistic_classifier(
    X: pd.DataFrame,
    y: pd.Series,         # Returns converted to 0/1 direction
    n_splits: int = 5,
    regularization: str = 'l2',
    C: float = 1.0,       # Inverse regularization strength
) -> dict:
    """
    Train logistic regression for direction prediction.
    
    Returns:
        - coefficients: Feature effects on probability
        - cv_accuracy: Out-of-sample accuracy
        - cv_auc: Area under ROC curve
        - confusion_matrix: TP, FP, TN, FN
    """
```

### Interpreting Logistic Coefficients

```python
# Coefficient interpretation
exp(coefficient) = odds ratio

# Example: RSI coefficient = -0.03
# exp(-0.03) = 0.97
# Each 1-point increase in RSI decreases odds of up-move by 3%
```

## CLI Usage (Planned)

```bash
# Linear regression baseline
python data_processing/train_ml_model.py \
    --model-type linear \
    --output data/cache/models/linear_model.json

# Ridge with tuning
python data_processing/train_ml_model.py \
    --model-type linear \
    --regularization ridge \
    --tune-alpha

# Logistic for direction
python data_processing/train_ml_model.py \
    --model-type logistic \
    --target direction
```

## Best Practices

### For Time-Series CV
1. **Always use temporal splits**: Never random splits for time series
2. **Include a gap**: At least 1 day between train and test
3. **Check fold stability**: Results should be similar across folds
4. **Use enough folds**: 5+ for reliable estimates

### For Linear Models
1. **Standardize features**: Put on same scale before fitting
2. **Check multicollinearity**: Highly correlated features confuse coefficients
3. **Look at t-stats**: Focus on significant features
4. **Compare to LightGBM**: If linear wins, simple model is enough

### For Factor Analysis
1. **Update factor data**: Ken French data updates monthly
2. **Match frequencies**: Daily factors for daily analysis
3. **Interpret alpha carefully**: Could be skill, could be missing factors

---

*Continue to: [LightGBM Models](03-lightgbm-models.md)*

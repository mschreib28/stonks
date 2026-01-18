# LightGBM Models (Tier 1)

**Status**: ✅ Complete  
**Module**: `data_processing/train_ml_model.py`  
**Book Reference**: Machine Learning for Algorithmic Trading, Chapter 6

## Why This Matters for Short-Term Trading

Gradient boosting models like LightGBM can capture **non-linear patterns** that linear models miss. For swing trading, this means:

1. **Complex interactions**: RSI + Volume together might predict better than either alone
2. **Non-linear thresholds**: "RSI < 30 AND falling" vs just "low RSI"
3. **Automatic feature interactions**: No manual engineering needed
4. **State-of-the-art performance**: Often the best ML method for tabular data

## Why LightGBM?

| Factor | LightGBM Advantage |
|--------|-------------------|
| **Speed** | 10-100x faster than XGBoost |
| **Memory** | Uses less RAM via histogram binning |
| **Accuracy** | Comparable or better than alternatives |
| **Features** | Native categorical support, early stopping |
| **Production** | Easy to deploy, small model files |

## Implemented Capabilities

### Classifier (Direction Prediction)

Predicts whether next-day returns will be positive or negative.

```python
def train_lightgbm_classifier(
    X: pd.DataFrame,           # Features
    y: pd.Series,              # Returns (converted to 0/1)
    n_splits: int = 5,         # CV folds
    test_size: int = 60,       # Test period per fold
    params: Optional[dict] = None,
) -> dict:
```

**Output Metrics**:
- **Accuracy**: % of correct predictions
- **AUC-ROC**: Area under ROC curve (>0.5 is better than random)
- **Precision/Recall**: For up and down predictions
- **Feature importance**: Which features matter most

### Regressor (Return Magnitude)

Predicts the actual return value.

```python
def train_lightgbm_regressor(
    X: pd.DataFrame,
    y: pd.Series,              # Actual returns
    n_splits: int = 5,
    test_size: int = 60,
    params: Optional[dict] = None,
) -> dict:
```

**Output Metrics**:
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **R²**: Variance explained (low is normal for returns)
- **Directional Accuracy**: % of times direction was correct

### SHAP Interpretability

Understand why the model makes specific predictions.

```python
def compute_shap_values(
    model,
    X: pd.DataFrame,
    max_samples: int = 1000,
) -> dict:
```

**Output**:
- SHAP values for each feature and sample
- Feature importance (mean absolute SHAP)
- Dependency plots (feature value vs SHAP impact)

## Model Parameters

### Default Configuration

```python
params = {
    'objective': 'binary',         # or 'regression'
    'metric': 'binary_logloss',    # or 'rmse'
    'boosting_type': 'gbdt',       # Gradient boosting
    'num_leaves': 31,              # Complexity control
    'learning_rate': 0.05,         # Conservative learning
    'feature_fraction': 0.8,       # Use 80% of features per tree
    'bagging_fraction': 0.8,       # Use 80% of data per tree
    'bagging_freq': 5,             # Bagging every 5 rounds
    'n_estimators': 100,           # Max trees
}
```

### Parameter Tuning Guide

| Parameter | Low Value | High Value | Effect |
|-----------|-----------|------------|--------|
| `num_leaves` | 15 | 63 | More leaves = more complex |
| `learning_rate` | 0.01 | 0.1 | Lower = slower but more accurate |
| `feature_fraction` | 0.5 | 1.0 | Lower = more regularization |
| `n_estimators` | 50 | 500 | More = risk of overfitting |

### For Small-Cap Swing Trading

Recommended conservative settings:
```python
params = {
    'num_leaves': 15,           # Simpler model
    'learning_rate': 0.03,      # Slower learning
    'feature_fraction': 0.7,    # More regularization
    'min_data_in_leaf': 50,     # Require more samples per leaf
    'n_estimators': 200,        # Let early stopping decide
}
```

## CLI Usage

### Train Classifier

```bash
uv run python data_processing/train_ml_model.py \
    --data-path data/cache/technical_features.parquet \
    --model-type classifier \
    --n-splits 5 \
    --output-dir data/cache/models
```

### Train Regressor

```bash
uv run python data_processing/train_ml_model.py \
    --model-type regressor \
    --n-splits 5
```

### Custom Parameters

```bash
uv run python data_processing/train_ml_model.py \
    --model-type classifier \
    --num-leaves 15 \
    --learning-rate 0.03 \
    --n-estimators 200
```

## Understanding Output

### Classifier Results

```python
{
    'model_type': 'classifier',
    'cv_metrics': {
        'accuracy_mean': 0.54,
        'accuracy_std': 0.02,
        'auc_mean': 0.56,
        'auc_std': 0.03,
    },
    'feature_importance': {
        'gain': {
            'rsi_14': 1250.5,
            'bb_pct': 980.3,
            'adx_14': 750.2,
            ...
        },
        'shap': {
            'rsi_14': 0.0045,
            'bb_pct': 0.0038,
            'adx_14': 0.0025,
            ...
        }
    },
    'confusion_matrix': [[TP, FP], [FN, TN]],
}
```

### Interpreting Metrics

**Accuracy of 54%**:
- Random = 50%, so 54% is better than random
- For trading: Even small edge × many trades = profit
- Don't expect 70%+ for return prediction

**AUC of 0.56**:
- 0.5 = random
- 0.6 = modest predictive power
- 0.7+ = strong (and suspicious—check for leakage)

**Feature Importance**:
- Gain: Total improvement from splits on this feature
- SHAP: Average impact on predictions
- Both should roughly agree on top features

## Feature Importance Analysis

### Gain-Based Importance

Measures how much each feature improves predictions when used in splits.

```python
importance = model.feature_importance(importance_type='gain')
```

**Interpretation**:
- Higher = more useful for making predictions
- Relative, not absolute
- Doesn't tell you direction of effect

### SHAP-Based Importance

Measures each feature's average impact on individual predictions.

```python
shap_values = compute_shap_values(model, X_test)
```

**Interpretation**:
- Positive SHAP: Feature pushes prediction up
- Negative SHAP: Feature pushes prediction down
- Magnitude: How much impact

### Example Analysis

```
Feature Importance (by SHAP):
1. rsi_14: 0.0045 (negative SHAP when high → oversold predicts bounce)
2. bb_pct: 0.0038 (negative when high → mean reversion)
3. adx_14: 0.0025 (positive when high → trend continues)
4. hist_vol_20: 0.0018 (negative → high vol predicts down)
```

**Trading insight**: Model confirms mean reversion on RSI/BB with trend following on ADX.

## Using Predictions

### For Stock Screening

```python
# Load trained model
model = load_model('data/cache/models/classifier.txt')

# Get today's features
features = load_features('data/cache/technical_features.parquet')
latest = features.groupby('ticker').last()

# Predict probabilities
probs = model.predict_proba(latest)[:, 1]  # Probability of up-move

# Rank by probability
ranking = pd.DataFrame({
    'ticker': latest.index,
    'up_probability': probs
}).sort_values('up_probability', ascending=False)
```

### For Entry Signals

```python
# Only trade high-confidence predictions
THRESHOLD = 0.55  # 55% confidence

signals = probs > THRESHOLD
tickers_to_buy = ranking[ranking['up_probability'] > THRESHOLD]
```

### Combining with Other Criteria

```python
# Model prediction + scoring criteria
candidates = scoring_results.merge(
    predictions, 
    on='ticker'
)

# Filter: High tradability AND model confidence
final_picks = candidates[
    (candidates['tradability_score'] > 70) &
    (candidates['up_probability'] > 0.55)
]
```

## Model Validation

### Out-of-Sample Testing

Time-series CV ensures honest evaluation:
```
Fold 1: Train Jan-Jun → Test Jul (2.5 months OOS)
Fold 2: Train Jan-Jul → Test Aug
...
```

### Walk-Forward Analysis

Best practice for production:
1. Train on 6 months
2. Trade for 1 month
3. Retrain including new month
4. Repeat

### Checking for Overfitting

| Warning Sign | Solution |
|--------------|----------|
| Train accuracy >> Test accuracy | More regularization |
| Metric variance across folds is high | More data or simpler model |
| Best model is most complex | Early stopping, fewer leaves |
| Performance degrades in later folds | Regime change, rolling window |

## Best Practices

### 1. Always Use Time-Series CV
Never use random train/test splits. Future information leaks into training.

### 2. Watch for Data Leakage
Common leakage sources:
- Using future prices in features
- Calculating technical indicators with lookahead
- Including target-related columns

### 3. Don't Over-Tune
- Use early stopping
- Prefer simpler models (fewer leaves)
- Validate on truly held-out data

### 4. Combine with Domain Knowledge
- Model says RSI important? Check if mean reversion makes sense
- High importance on weird feature? Might be leakage
- Low importance on expected features? Check feature calculation

### 5. Regular Retraining
Markets change. Retrain monthly or when performance degrades.

## Comparison: LightGBM vs Linear Models

| Aspect | Linear Model | LightGBM |
|--------|--------------|----------|
| **Interpretability** | High (coefficients) | Medium (SHAP) |
| **Non-linear patterns** | No | Yes |
| **Feature interactions** | Manual | Automatic |
| **Speed** | Very fast | Fast |
| **Overfitting risk** | Low | Higher |
| **Small data** | Works well | Needs more data |

**Recommendation**: Train both. If LightGBM doesn't beat linear, use linear (simpler is better).

---

*Continue to: [VectorBT Backtesting](04-vectorbt-backtesting.md)*

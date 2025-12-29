# Model Optimization Guide

## Optimization Strategy Selection

```
Start
  │
  ▼
┌─────────────────┐
│ Few hyperparams │──► Grid Search (exhaustive)
│ (<5 params)     │
└────────┬────────┘
         │ Many params
         ▼
┌─────────────────┐
│ Limited budget  │──► Random Search (faster)
│ (<100 trials)   │
└────────┬────────┘
         │ More budget
         ▼
┌─────────────────┐
│ Expensive evals │──► Bayesian (Optuna/HyperOpt)
│ (DL models)     │
└────────┬────────┘
         │ Very expensive
         ▼
┌─────────────────┐
│ Unlimited       │──► Population (evolutionary)
│ compute         │
└─────────────────┘
```

## Search Method Comparison

| Method | Efficiency | Parallelizable | Best For |
|--------|------------|----------------|----------|
| Grid Search | Low | Yes | Small spaces |
| Random Search | Medium | Yes | High-dim spaces |
| Bayesian (TPE) | High | Limited | Most cases |
| Evolutionary | Medium | Yes | Complex spaces |
| Hyperband | High | Yes | Neural networks |

## Common Hyperparameter Ranges

### Tree-based Models

```yaml
random_forest:
  n_estimators: [100, 500]
  max_depth: [3, 20]
  min_samples_split: [2, 20]
  min_samples_leaf: [1, 10]

xgboost:
  n_estimators: [100, 1000]
  max_depth: [3, 12]
  learning_rate: [0.01, 0.3]
  subsample: [0.6, 1.0]
  colsample_bytree: [0.6, 1.0]

lightgbm:
  n_estimators: [100, 1000]
  num_leaves: [20, 150]
  max_depth: [3, 12]
  learning_rate: [0.01, 0.3]
```

### Neural Networks

```yaml
learning_rate: [1e-5, 1e-2] (log scale)
batch_size: [16, 32, 64, 128, 256]
dropout: [0.0, 0.5]
hidden_units: [32, 64, 128, 256, 512]
weight_decay: [1e-6, 1e-2] (log scale)
```

## Model Compression Techniques

| Technique | Size Reduction | Speed Gain | Accuracy Loss |
|-----------|----------------|------------|---------------|
| Quantization (INT8) | 4x | 2-4x | 0-2% |
| Pruning (30%) | 1.4x | 1.2x | 0-1% |
| Knowledge Distillation | Varies | Varies | 1-3% |
| ONNX Conversion | 1x | 1.5-2x | 0% |
| TensorRT | 1x | 2-5x | <1% |

## Early Stopping Best Practices

```python
# Configuration
patience = 10           # Wait 10 epochs
min_delta = 0.001      # Minimum improvement
monitor = 'val_loss'   # Metric to watch
mode = 'min'           # Lower is better
restore_best = True    # Load best weights
```

## Cross-Validation Strategies

| Strategy | Use When |
|----------|----------|
| KFold | Standard, balanced data |
| StratifiedKFold | Imbalanced classification |
| TimeSeriesSplit | Time-dependent data |
| GroupKFold | Group-based data |
| LeaveOneOut | Very small datasets |

## Optimization Checklist

```markdown
Pre-optimization:
□ Baseline model established
□ Evaluation metric defined
□ Cross-validation strategy chosen
□ Search space defined

During optimization:
□ Early stopping enabled
□ Pruning for failed trials
□ Results logged
□ Resource limits set

Post-optimization:
□ Best model validated
□ Hyperparameter importance analyzed
□ Overfitting checked
□ Model saved and documented
```

## Common Pitfalls

1. **Overfitting to validation set** - Use nested CV
2. **Too narrow search space** - Start broad
3. **Ignoring early stopping** - Wastes compute
4. **No baseline comparison** - Can't measure improvement
5. **Random seed not fixed** - Non-reproducible results

## Tools Comparison

| Tool | Ease | Distributed | Visualization |
|------|------|-------------|---------------|
| Optuna | High | Yes | Built-in |
| Ray Tune | Medium | Excellent | TensorBoard |
| Hyperopt | Medium | Limited | Manual |
| Scikit-optimize | High | No | Manual |
| Keras Tuner | High | No | Built-in |

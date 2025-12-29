# Machine Learning Guide

Comprehensive guide for building ML models with scikit-learn.

## Model Selection Flowchart

```
Is it supervised learning?
├── Yes → Is the target categorical?
│   ├── Yes → Classification
│   │   ├── Binary → LogisticRegression, RandomForestClassifier, XGBoost
│   │   └── Multi-class → RandomForestClassifier, GradientBoosting, Neural Networks
│   └── No → Regression
│       └── LinearRegression, RandomForestRegressor, XGBRegressor
└── No → Unsupervised Learning
    ├── Clustering → KMeans, DBSCAN, Hierarchical
    └── Dimensionality Reduction → PCA, t-SNE, UMAP
```

## Best Practices

### 1. Data Preparation
- Always split data before any preprocessing
- Use stratified splits for imbalanced datasets
- Scale features for distance-based algorithms

### 2. Feature Engineering
- Handle missing values appropriately
- Create meaningful features from domain knowledge
- Remove highly correlated features

### 3. Model Training
- Use cross-validation for reliable estimates
- Start simple, add complexity as needed
- Monitor for overfitting

### 4. Evaluation
- Use appropriate metrics for your problem
- Consider business context, not just accuracy
- Validate on held-out test set

## Common Pitfalls

1. **Data Leakage**: Preprocessing before splitting
2. **Overfitting**: High train, low test performance
3. **Wrong Metric**: Using accuracy on imbalanced data
4. **Ignoring Features**: Not understanding feature importance

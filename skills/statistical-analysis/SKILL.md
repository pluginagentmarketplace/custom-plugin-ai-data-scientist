---
name: statistical-analysis
description: Probability, distributions, hypothesis testing, and statistical inference. Use for A/B testing, experimental design, or statistical validation.
sasmp_version: "1.3.0"
bonded_agent: 02-mathematics-statistics
bond_type: PRIMARY_BOND
---

# Statistical Analysis

Apply statistical methods to understand data and validate findings.

## Quick Start

```python
from scipy import stats
import numpy as np

# Descriptive statistics
data = np.array([1, 2, 3, 4, 5])
print(f"Mean: {np.mean(data)}")
print(f"Std: {np.std(data)}")

# Hypothesis testing
group1 = [23, 25, 27, 29, 31]
group2 = [20, 22, 24, 26, 28]
t_stat, p_value = stats.ttest_ind(group1, group2)
print(f"P-value: {p_value}")
```

## Core Tests

### T-Test (Compare Means)
```python
# One-sample: Compare to population mean
stats.ttest_1samp(data, 100)

# Two-sample: Compare two groups
stats.ttest_ind(group1, group2)

# Paired: Before/after comparison
stats.ttest_rel(before, after)
```

### Chi-Square (Categorical Data)
```python
from scipy.stats import chi2_contingency

observed = np.array([[10, 20], [15, 25]])
chi2, p_value, dof, expected = chi2_contingency(observed)
```

### ANOVA (Multiple Groups)
```python
f_stat, p_value = stats.f_oneway(group1, group2, group3)
```

## Confidence Intervals

```python
from scipy import stats

confidence_level = 0.95
mean = np.mean(data)
se = stats.sem(data)
ci = stats.t.interval(confidence_level, len(data)-1, mean, se)

print(f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
```

## Correlation

```python
# Pearson (linear)
r, p_value = stats.pearsonr(x, y)

# Spearman (rank-based)
rho, p_value = stats.spearmanr(x, y)
```

## Distributions

```python
# Normal
x = np.linspace(-3, 3, 100)
pdf = stats.norm.pdf(x, loc=0, scale=1)

# Sampling
samples = np.random.normal(0, 1, 1000)

# Test normality
stat, p_value = stats.shapiro(data)
```

## A/B Testing Framework

```python
def ab_test(control, treatment, alpha=0.05):
    """
    Run A/B test with statistical significance

    Returns: significant (bool), p_value (float)
    """
    t_stat, p_value = stats.ttest_ind(control, treatment)

    significant = p_value < alpha
    improvement = (np.mean(treatment) - np.mean(control)) / np.mean(control) * 100

    return {
        'significant': significant,
        'p_value': p_value,
        'improvement': f"{improvement:.2f}%"
    }
```

## Interpretation

**P-value < 0.05**: Reject null hypothesis (statistically significant)

**P-value >= 0.05**: Fail to reject null (not significant)

## Common Pitfalls

- Multiple testing without correction
- Small sample sizes
- Ignoring assumptions (normality, independence)
- Confusing correlation with causation
- p-hacking (searching for significance)

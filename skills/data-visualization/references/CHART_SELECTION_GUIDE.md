# Chart Selection Guide

## Quick Selection Matrix

```
What do you want to show?
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ COMPARISON                                                               │
│   Few categories → Bar Chart                                             │
│   Many categories → Horizontal Bar                                       │
│   Over time → Line Chart                                                 │
├─────────────────────────────────────────────────────────────────────────┤
│ DISTRIBUTION                                                             │
│   Single variable → Histogram, Box Plot                                  │
│   Two variables → Scatter Plot                                           │
│   Multiple variables → Pair Plot, Violin Plot                            │
├─────────────────────────────────────────────────────────────────────────┤
│ RELATIONSHIP                                                             │
│   Correlation → Scatter Plot, Heatmap                                    │
│   Part-to-whole → Pie Chart, Stacked Bar                                 │
│   Hierarchy → Treemap, Sunburst                                          │
├─────────────────────────────────────────────────────────────────────────┤
│ TREND                                                                    │
│   Over time → Line Chart, Area Chart                                     │
│   With uncertainty → Line + Confidence Interval                          │
│   Multiple series → Multi-line Plot                                      │
├─────────────────────────────────────────────────────────────────────────┤
│ COMPOSITION                                                              │
│   Static → Pie Chart, Donut                                              │
│   Over time → Stacked Area                                               │
│   Nested → Treemap                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## Chart Types Quick Reference

| Chart | Best For | Avoid When |
|-------|----------|------------|
| Bar | Comparing categories | Too many categories (>15) |
| Line | Trends over time | Non-sequential data |
| Scatter | Relationships | Overlapping points |
| Histogram | Distributions | Small samples |
| Box Plot | Comparing distributions | Non-normal data |
| Heatmap | Correlations, matrices | Too many variables |
| Pie | Part-to-whole (2-6 parts) | Many categories |
| Area | Cumulative trends | Overlapping series |

## Color Guidelines

### When to Use Color

| Purpose | Strategy |
|---------|----------|
| Categorical | Different hue per category |
| Sequential | Light to dark gradient |
| Diverging | Two colors from center |
| Highlighting | One accent color |
| Grouping | Color families |

### Colorblind-Safe Palettes

```python
# Recommended palettes
palettes = {
    'categorical': ['#0077BB', '#EE7733', '#009988', '#CC3311'],
    'sequential': 'viridis',  # or 'plasma', 'cividis'
    'diverging': 'coolwarm'
}

# Avoid: red-green combinations
# Prefer: blue-orange, purple-green
```

## Chart Anatomy Best Practices

```
┌─────────────────────────────────────────────┐
│            TITLE (Descriptive)              │ ← Clear, informative
├─────────────────────────────────────────────┤
│  Y │                                        │
│    │         ╭────────╮                     │
│  L │    ╭────╯        ╰────╮               │
│  A │ ───╯                  ╰───             │
│  B │                                        │
│  E │                                        │
│  L │                                        │
│    └────────────────────────────────────────│
│            X AXIS LABEL                     │ ← Units included
│                                    LEGEND ──│ ← If needed
│            Source: XYZ Dataset              │ ← Optional
└─────────────────────────────────────────────┘
```

## Common Mistakes to Avoid

1. **Truncated Y-axis** - Start at zero for bars
2. **Too many colors** - Limit to 5-7 max
3. **3D effects** - Distort perception
4. **Dual Y-axes** - Confusing
5. **Pie charts with many slices** - Use bar instead
6. **Missing labels** - Always include axes labels
7. **Small fonts** - Ensure readability
8. **No legend** - Explain what colors mean

## Python Quick Recipes

```python
# Distribution
sns.histplot(df['col'], kde=True)

# Comparison
df.plot.bar(x='category', y='value')

# Relationship
sns.scatterplot(x='var1', y='var2', hue='group', data=df)

# Correlation
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')

# Time series
df.plot(x='date', y='value', figsize=(12, 6))

# Subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
```

## Export Settings

```python
# Publication quality
plt.savefig('figure.png', dpi=300, bbox_inches='tight')
plt.savefig('figure.pdf', bbox_inches='tight')  # Vector
plt.savefig('figure.svg', bbox_inches='tight')  # Editable
```

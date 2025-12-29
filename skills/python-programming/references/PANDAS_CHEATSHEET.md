# Pandas Cheatsheet for Data Science

## Data Loading

```python
# CSV
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv', parse_dates=['date'], index_col='id')

# Excel
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')

# SQL
df = pd.read_sql_query("SELECT * FROM table", connection)

# JSON
df = pd.read_json('file.json')

# Parquet (fast, compressed)
df = pd.read_parquet('file.parquet')
```

## Quick Exploration

```python
df.head()           # First 5 rows
df.tail()           # Last 5 rows
df.shape            # (rows, columns)
df.info()           # Types and memory
df.describe()       # Statistics
df.columns          # Column names
df.dtypes           # Data types
df.nunique()        # Unique counts
df.isnull().sum()   # Missing values
```

## Selection & Indexing

```python
# Single column
df['col']              # Series
df[['col']]            # DataFrame

# Multiple columns
df[['col1', 'col2']]

# Rows by index
df.loc[0]              # By label
df.iloc[0]             # By position

# Rows and columns
df.loc[0:5, 'col1':'col3']
df.iloc[0:5, 0:3]

# Conditional
df[df['age'] > 30]
df[(df['age'] > 30) & (df['city'] == 'NYC')]
df.query("age > 30 and city == 'NYC'")
```

## Data Cleaning

```python
# Missing values
df.dropna()                    # Drop rows with any NaN
df.dropna(subset=['col'])      # Drop if NaN in specific column
df.fillna(0)                   # Fill with value
df.fillna(df.mean())           # Fill with mean
df.fillna(method='ffill')      # Forward fill
df.interpolate()               # Interpolate

# Duplicates
df.drop_duplicates()
df.drop_duplicates(subset=['col'], keep='first')

# Type conversion
df['col'] = df['col'].astype(int)
df['date'] = pd.to_datetime(df['date'])
df['cat'] = df['cat'].astype('category')

# String operations
df['col'].str.lower()
df['col'].str.strip()
df['col'].str.replace('old', 'new')
df['col'].str.contains('pattern')
```

## Transformations

```python
# Apply function
df['new'] = df['col'].apply(lambda x: x * 2)
df['new'] = df.apply(lambda row: row['a'] + row['b'], axis=1)

# Map values
df['col'] = df['col'].map({'old': 'new'})

# Replace values
df['col'].replace({'A': 1, 'B': 2})

# Binning
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                         labels=['young', 'adult', 'middle', 'senior'])

# Encoding
pd.get_dummies(df['category'])  # One-hot
df['category'].astype('category').cat.codes  # Label
```

## Aggregation

```python
# Basic
df['col'].sum()
df['col'].mean()
df['col'].median()
df['col'].std()
df['col'].min() / df['col'].max()
df['col'].value_counts()

# GroupBy
df.groupby('category')['value'].mean()
df.groupby('category').agg({
    'value': ['sum', 'mean', 'count'],
    'other': 'max'
})

# Multiple aggregations
df.groupby('cat').agg(
    total=('value', 'sum'),
    average=('value', 'mean'),
    count=('id', 'count')
)

# Pivot table
df.pivot_table(
    values='sales',
    index='region',
    columns='product',
    aggfunc='sum',
    fill_value=0
)
```

## Merging & Joining

```python
# Merge (SQL-like join)
pd.merge(df1, df2, on='key')
pd.merge(df1, df2, on='key', how='left')  # left, right, inner, outer

# Concat
pd.concat([df1, df2], axis=0)  # Stack vertically
pd.concat([df1, df2], axis=1)  # Stack horizontally

# Join on index
df1.join(df2, how='left')
```

## Time Series

```python
# Date parsing
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['dayofweek'] = df['date'].dt.dayofweek
df['quarter'] = df['date'].dt.quarter

# Resampling
df.set_index('date').resample('M').mean()  # Monthly
df.set_index('date').resample('W').sum()   # Weekly

# Rolling windows
df['rolling_mean'] = df['value'].rolling(window=7).mean()
df['expanding_mean'] = df['value'].expanding().mean()
```

## Performance Tips

```python
# 1. Use vectorized operations (not loops!)
# Bad
for i in range(len(df)):
    df.loc[i, 'new'] = df.loc[i, 'old'] * 2

# Good
df['new'] = df['old'] * 2

# 2. Optimize dtypes
df['int_col'] = df['int_col'].astype('int32')  # Not int64
df['cat_col'] = df['cat_col'].astype('category')

# 3. Use Parquet for large files
df.to_parquet('file.parquet')

# 4. Use chunking for huge files
for chunk in pd.read_csv('huge.csv', chunksize=10000):
    process(chunk)

# 5. Check memory usage
df.memory_usage(deep=True)
```

## Common Patterns

```python
# Percentage of total
df['pct'] = df['value'] / df['value'].sum() * 100

# Cumulative sum
df['cumsum'] = df['value'].cumsum()

# Rank
df['rank'] = df['value'].rank(ascending=False)

# Shift (lag)
df['prev_value'] = df['value'].shift(1)
df['change'] = df['value'] - df['value'].shift(1)

# Percentage change
df['pct_change'] = df['value'].pct_change()
```

#!/usr/bin/env python3
"""
Python Data Processing Utilities
Common patterns for data science workflows
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Callable
from pathlib import Path
from functools import wraps
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


class DataProcessor:
    """Comprehensive data processing utilities for pandas DataFrames."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._original = df.copy()
        self._transformations = []

    @classmethod
    def from_csv(cls, path: Union[str, Path], **kwargs) -> 'DataProcessor':
        """Load DataFrame from CSV file."""
        df = pd.read_csv(path, **kwargs)
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return cls(df)

    @classmethod
    def from_parquet(cls, path: Union[str, Path], **kwargs) -> 'DataProcessor':
        """Load DataFrame from Parquet file."""
        df = pd.read_parquet(path, **kwargs)
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return cls(df)

    def info(self) -> Dict:
        """Get comprehensive DataFrame information."""
        return {
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2,
            'null_counts': self.df.isnull().sum().to_dict(),
            'null_percentages': (self.df.isnull().mean() * 100).to_dict()
        }

    @timer
    def clean_column_names(self) -> 'DataProcessor':
        """Standardize column names (lowercase, underscores)."""
        self.df.columns = (
            self.df.columns
            .str.lower()
            .str.strip()
            .str.replace(r'[^\w\s]', '', regex=True)
            .str.replace(r'\s+', '_', regex=True)
        )
        self._transformations.append('clean_column_names')
        return self

    @timer
    def drop_duplicates(self, subset: Optional[List[str]] = None,
                       keep: str = 'first') -> 'DataProcessor':
        """Remove duplicate rows."""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset, keep=keep)
        after = len(self.df)
        logger.info(f"Removed {before - after:,} duplicate rows")
        self._transformations.append(f'drop_duplicates(subset={subset})')
        return self

    @timer
    def handle_missing(self, strategy: str = 'drop',
                      columns: Optional[List[str]] = None,
                      fill_value: Optional[any] = None) -> 'DataProcessor':
        """
        Handle missing values.

        Strategies:
            - 'drop': Remove rows with missing values
            - 'fill': Fill with specified value
            - 'mean': Fill with column mean (numeric only)
            - 'median': Fill with column median (numeric only)
            - 'mode': Fill with column mode
            - 'ffill': Forward fill
            - 'bfill': Backward fill
        """
        cols = columns or self.df.columns.tolist()

        if strategy == 'drop':
            self.df = self.df.dropna(subset=cols)
        elif strategy == 'fill':
            self.df[cols] = self.df[cols].fillna(fill_value)
        elif strategy == 'mean':
            for col in cols:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        elif strategy == 'median':
            for col in cols:
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        elif strategy == 'mode':
            for col in cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode().iloc[0])
        elif strategy in ['ffill', 'bfill']:
            self.df[cols] = self.df[cols].fillna(method=strategy)

        self._transformations.append(f'handle_missing({strategy})')
        return self

    @timer
    def convert_dtypes(self, dtype_map: Dict[str, str]) -> 'DataProcessor':
        """Convert column data types."""
        for col, dtype in dtype_map.items():
            if col in self.df.columns:
                if dtype == 'datetime':
                    self.df[col] = pd.to_datetime(self.df[col])
                elif dtype == 'category':
                    self.df[col] = self.df[col].astype('category')
                else:
                    self.df[col] = self.df[col].astype(dtype)

        self._transformations.append('convert_dtypes')
        return self

    @timer
    def encode_categorical(self, columns: List[str],
                          method: str = 'onehot') -> 'DataProcessor':
        """
        Encode categorical variables.

        Methods:
            - 'onehot': One-hot encoding
            - 'label': Label encoding
            - 'ordinal': Ordinal encoding (preserves order)
        """
        if method == 'onehot':
            self.df = pd.get_dummies(self.df, columns=columns, prefix=columns)
        elif method == 'label':
            for col in columns:
                self.df[col] = self.df[col].astype('category').cat.codes

        self._transformations.append(f'encode_categorical({method})')
        return self

    @timer
    def scale_numeric(self, columns: Optional[List[str]] = None,
                     method: str = 'standard') -> 'DataProcessor':
        """
        Scale numeric columns.

        Methods:
            - 'standard': Zero mean, unit variance
            - 'minmax': Scale to [0, 1]
            - 'robust': Use median and IQR (robust to outliers)
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()

        for col in columns:
            if method == 'standard':
                mean = self.df[col].mean()
                std = self.df[col].std()
                self.df[col] = (self.df[col] - mean) / std if std > 0 else 0
            elif method == 'minmax':
                min_val = self.df[col].min()
                max_val = self.df[col].max()
                range_val = max_val - min_val
                self.df[col] = (self.df[col] - min_val) / range_val if range_val > 0 else 0
            elif method == 'robust':
                median = self.df[col].median()
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                self.df[col] = (self.df[col] - median) / iqr if iqr > 0 else 0

        self._transformations.append(f'scale_numeric({method})')
        return self

    @timer
    def remove_outliers(self, columns: List[str],
                       method: str = 'iqr',
                       threshold: float = 1.5) -> 'DataProcessor':
        """
        Remove outliers from numeric columns.

        Methods:
            - 'iqr': Interquartile range method
            - 'zscore': Z-score method (threshold = number of std devs)
        """
        before = len(self.df)

        for col in columns:
            if method == 'iqr':
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - threshold * iqr
                upper = q3 + threshold * iqr
                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]
            elif method == 'zscore':
                mean = self.df[col].mean()
                std = self.df[col].std()
                z_scores = np.abs((self.df[col] - mean) / std)
                self.df = self.df[z_scores < threshold]

        after = len(self.df)
        logger.info(f"Removed {before - after:,} outlier rows")
        self._transformations.append(f'remove_outliers({method})')
        return self

    def reset(self) -> 'DataProcessor':
        """Reset to original DataFrame."""
        self.df = self._original.copy()
        self._transformations = []
        return self

    def get_transformations(self) -> List[str]:
        """Get list of applied transformations."""
        return self._transformations.copy()

    def to_csv(self, path: Union[str, Path], **kwargs) -> None:
        """Save DataFrame to CSV."""
        self.df.to_csv(path, index=False, **kwargs)
        logger.info(f"Saved {len(self.df):,} rows to {path}")

    def to_parquet(self, path: Union[str, Path], **kwargs) -> None:
        """Save DataFrame to Parquet."""
        self.df.to_parquet(path, index=False, **kwargs)
        logger.info(f"Saved {len(self.df):,} rows to {path}")


def main():
    """Demo data processing pipeline."""
    print("Python Data Processor Demo")
    print("=" * 50)

    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'User ID': range(1, 101),
        'Age': np.random.randint(18, 80, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100),
        'Score': np.random.uniform(0, 100, 100)
    })

    # Add some missing values
    df.loc[5:10, 'Age'] = np.nan
    df.loc[15:20, 'Income'] = np.nan

    # Process data
    processor = (
        DataProcessor(df)
        .clean_column_names()
        .drop_duplicates()
        .handle_missing(strategy='median')
        .remove_outliers(['income'], method='iqr')
        .scale_numeric(['income', 'score'], method='standard')
    )

    print(f"\nOriginal shape: {df.shape}")
    print(f"Processed shape: {processor.df.shape}")
    print(f"\nTransformations applied:")
    for t in processor.get_transformations():
        print(f"  - {t}")

    print("\n[SUCCESS] Data processing complete!")


if __name__ == '__main__':
    main()

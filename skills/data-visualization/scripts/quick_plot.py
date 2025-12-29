#!/usr/bin/env python3
"""
Quick Plotting Utilities for Data Science
One-liner visualizations for rapid EDA
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Optional, List, Union, Tuple
from pathlib import Path


def set_style(style: str = 'professional'):
    """Set global plotting style."""
    styles = {
        'professional': {
            'style': 'whitegrid',
            'context': 'notebook',
            'palette': 'deep'
        },
        'minimal': {
            'style': 'white',
            'context': 'paper',
            'palette': 'muted'
        },
        'dark': {
            'style': 'darkgrid',
            'context': 'talk',
            'palette': 'bright'
        },
        'publication': {
            'style': 'ticks',
            'context': 'paper',
            'palette': 'colorblind'
        }
    }

    config = styles.get(style, styles['professional'])
    sns.set_style(config['style'])
    sns.set_context(config['context'])
    sns.set_palette(config['palette'])


def quick_hist(data: Union[pd.Series, np.ndarray, list],
               title: str = None,
               bins: int = 30,
               kde: bool = True,
               figsize: Tuple[int, int] = (10, 6),
               save: str = None) -> plt.Figure:
    """Quick histogram with optional KDE."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(data, bins=bins, kde=kde, ax=ax)

    ax.set_title(title or 'Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


def quick_scatter(x: Union[pd.Series, np.ndarray],
                  y: Union[pd.Series, np.ndarray],
                  hue: Optional[Union[pd.Series, np.ndarray]] = None,
                  title: str = None,
                  xlabel: str = None,
                  ylabel: str = None,
                  figsize: Tuple[int, int] = (10, 6),
                  save: str = None) -> plt.Figure:
    """Quick scatter plot with optional coloring."""
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(x, y, c=hue, alpha=0.7, cmap='viridis', edgecolors='white')

    if hue is not None:
        plt.colorbar(scatter, label='Value')

    ax.set_title(title or 'Scatter Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel or 'X')
    ax.set_ylabel(ylabel or 'Y')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


def quick_bar(x: Union[pd.Series, list],
              y: Union[pd.Series, list],
              title: str = None,
              horizontal: bool = False,
              figsize: Tuple[int, int] = (10, 6),
              save: str = None) -> plt.Figure:
    """Quick bar chart."""
    fig, ax = plt.subplots(figsize=figsize)

    if horizontal:
        ax.barh(x, y, color=sns.color_palette()[0], edgecolor='white')
        ax.set_xlabel('Value')
    else:
        ax.bar(x, y, color=sns.color_palette()[0], edgecolor='white')
        ax.set_ylabel('Value')
        plt.xticks(rotation=45, ha='right')

    ax.set_title(title or 'Bar Chart', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


def quick_line(x: Union[pd.Series, np.ndarray],
               y: Union[pd.Series, np.ndarray, List],
               labels: List[str] = None,
               title: str = None,
               xlabel: str = None,
               ylabel: str = None,
               figsize: Tuple[int, int] = (12, 6),
               save: str = None) -> plt.Figure:
    """Quick line plot (supports multiple lines)."""
    fig, ax = plt.subplots(figsize=figsize)

    # Handle single or multiple lines
    if isinstance(y, list) and isinstance(y[0], (list, np.ndarray, pd.Series)):
        for i, line_y in enumerate(y):
            label = labels[i] if labels else f'Series {i+1}'
            ax.plot(x, line_y, marker='o', label=label, linewidth=2, markersize=4)
        ax.legend()
    else:
        ax.plot(x, y, marker='o', linewidth=2, markersize=4)

    ax.set_title(title or 'Line Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel or 'X')
    ax.set_ylabel(ylabel or 'Y')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


def quick_box(df: pd.DataFrame,
              x: str = None,
              y: str = None,
              title: str = None,
              figsize: Tuple[int, int] = (10, 6),
              save: str = None) -> plt.Figure:
    """Quick box plot."""
    fig, ax = plt.subplots(figsize=figsize)

    if x and y:
        sns.boxplot(data=df, x=x, y=y, ax=ax)
        plt.xticks(rotation=45, ha='right')
    else:
        sns.boxplot(data=df, ax=ax)

    ax.set_title(title or 'Box Plot', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


def quick_heatmap(data: pd.DataFrame,
                  title: str = None,
                  annot: bool = True,
                  figsize: Tuple[int, int] = (10, 8),
                  save: str = None) -> plt.Figure:
    """Quick correlation heatmap."""
    fig, ax = plt.subplots(figsize=figsize)

    # If not already a correlation matrix, compute it
    if data.shape[0] != data.shape[1]:
        data = data.corr()

    mask = np.triu(np.ones_like(data, dtype=bool))
    sns.heatmap(data, mask=mask, annot=annot, cmap='coolwarm',
                center=0, square=True, linewidths=0.5,
                fmt='.2f', ax=ax)

    ax.set_title(title or 'Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')
    return fig


def quick_pairplot(df: pd.DataFrame,
                   hue: str = None,
                   figsize: Tuple[int, int] = None,
                   save: str = None):
    """Quick pair plot for multivariate exploration."""
    g = sns.pairplot(df, hue=hue, diag_kind='kde',
                     plot_kws={'alpha': 0.6, 'edgecolor': 'white'},
                     height=2.5, aspect=1)

    g.fig.suptitle('Pair Plot', y=1.02, fontsize=14, fontweight='bold')

    if save:
        g.fig.savefig(save, dpi=300, bbox_inches='tight')
    return g


def quick_eda(df: pd.DataFrame,
              figsize: Tuple[int, int] = (16, 12),
              save: str = None) -> plt.Figure:
    """Quick EDA dashboard for a DataFrame."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) == 0:
        print("No numeric columns found!")
        return None

    # Limit to first 6 numeric columns
    cols_to_plot = numeric_cols[:6]
    n_cols = len(cols_to_plot)

    fig, axes = plt.subplots(2, min(n_cols, 3), figsize=figsize)
    axes = axes.flatten()

    # Row 1: Distributions
    for i, col in enumerate(cols_to_plot[:3]):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f'{col} Distribution')

    # Row 2: Box plots
    for i, col in enumerate(cols_to_plot[:3]):
        sns.boxplot(data=df, y=col, ax=axes[i + 3])
        axes[i + 3].set_title(f'{col} Box Plot')

    # Hide unused axes
    for j in range(n_cols, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Quick EDA Dashboard', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=300, bbox_inches='tight')

    return fig


def main():
    """Demo quick plotting utilities."""
    print("Quick Plotting Demo")
    print("=" * 50)

    # Set style
    set_style('professional')

    # Generate sample data
    np.random.seed(42)
    n = 100

    df = pd.DataFrame({
        'x': np.linspace(0, 10, n),
        'y1': np.sin(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n),
        'y2': np.cos(np.linspace(0, 10, n)) + np.random.normal(0, 0.1, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'value': np.random.normal(50, 15, n)
    })

    print("Sample DataFrame created:")
    print(df.head())

    # Demo: Histogram
    quick_hist(df['value'], title='Value Distribution')
    plt.show()

    # Demo: Line plot with multiple series
    quick_line(df['x'], [df['y1'], df['y2']],
              labels=['Sin', 'Cos'],
              title='Multiple Line Plot')
    plt.show()

    # Demo: Correlation heatmap
    quick_heatmap(df[['y1', 'y2', 'value']], title='Correlation Matrix')
    plt.show()

    print("\n[SUCCESS] Quick plotting demo complete!")


if __name__ == '__main__':
    main()

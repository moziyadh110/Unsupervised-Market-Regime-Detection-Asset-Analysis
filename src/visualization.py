"""
Visualization utilities using matplotlib and seaborn.
Creates plots for EDA, clustering, and regime analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def setup_plotting_style():
    """Set up consistent plotting style."""
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 10


def plot_time_series(df: pd.DataFrame, x_col: str = 'DATE', 
                     y_cols: List[str] = None, figsize: tuple = (14, 8)) -> None:
    """
    Plot time series data.
    
    Args:
        df: DataFrame with time series data
        x_col: Column name for x-axis (dates)
        y_cols: List of columns to plot
        figsize: Figure size
    """
    if y_cols is None:
        y_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    plt.figure(figsize=figsize)
    for col in y_cols:
        plt.plot(df[x_col], df[col], label=col, alpha=0.7)
    
    plt.xlabel(x_col)
    plt.ylabel('Value')
    plt.title('Time Series Plot')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, figsize: tuple = (10, 8)) -> None:
    """
    Plot correlation matrix heatmap.
    
    Args:
        df: DataFrame with numeric columns
        figsize: Figure size
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, column: str, bins: int = 50, figsize: tuple = (10, 6)) -> None:
    """
    Plot histogram with distribution.
    
    Args:
        df: DataFrame
        column: Column to plot
        bins: Number of bins
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.hist(df[column].dropna(), bins=bins, edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.show()


def plot_boxplot_by_group(df: pd.DataFrame, x_col: str, y_col: str, 
                          figsize: tuple = (10, 6)) -> None:
    """
    Plot boxplot grouped by category.
    
    Args:
        df: DataFrame
        x_col: Grouping column (category)
        y_col: Value column
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_col, y=y_col, data=df)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} by {x_col}')
    plt.tight_layout()
    plt.show()


def plot_scatter_2d(df: pd.DataFrame, x_col: str, y_col: str, 
                    color_col: str = None, figsize: tuple = (10, 6)) -> None:
    """
    Plot 2D scatter plot with optional color coding.
    
    Args:
        df: DataFrame
        x_col: X-axis column
        y_col: Y-axis column
        color_col: Column to use for color encoding (optional)
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    if color_col:
        scatter = plt.scatter(df[x_col], df[y_col], c=df[color_col], 
                             cmap='viridis', alpha=0.6, s=50)
        plt.colorbar(scatter, label=color_col)
    else:
        plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50)
    
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(labels: np.ndarray, figsize: tuple = (8, 6)) -> None:
    """
    Plot cluster size distribution.
    
    Args:
        labels: Cluster labels array
        figsize: Figure size
    """
    unique, counts = np.unique(labels, return_counts=True)
    
    plt.figure(figsize=figsize)
    bars = plt.bar(unique, counts, color='steelblue', edgecolor='black', alpha=0.7)
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom')
    
    plt.xticks(unique)
    plt.tight_layout()
    plt.show()


def plot_regime_characteristics(df: pd.DataFrame, regime_col: str = 'Regime',
                               figsize: tuple = (14, 10)) -> None:
    """
    Plot key characteristics of each regime.
    
    Args:
        df: DataFrame with regime assignments
        regime_col: Column with regime labels
        figsize: Figure size
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove the regime column itself if present
    numeric_cols = [col for col in numeric_cols if col != regime_col]
    
    # Select top features
    features_to_plot = numeric_cols[:6] if len(numeric_cols) > 6 else numeric_cols
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()
    
    for idx, feature in enumerate(features_to_plot):
        sns.boxplot(x=regime_col, y=feature, data=df, ax=axes[idx])
        axes[idx].set_title(f'{feature} by Regime')
    
    # Hide empty subplots
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def plot_pca_clusters(X_transformed: np.ndarray, labels: np.ndarray,
                     figsize: tuple = (10, 8)) -> None:
    """
    Plot clusters in 2D PCA space.
    
    Args:
        X_transformed: 2D transformed features (from PCA)
        labels: Cluster labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    scatter = plt.scatter(X_transformed[:, 0], X_transformed[:, 1], 
                         c=labels, cmap='viridis', alpha=0.6, s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Clusters in 2D PCA Space')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_elbow_curve(k_values: List[int], scores: List[float], 
                    metric: str = 'Silhouette', figsize: tuple = (10, 6)) -> None:
    """
    Plot elbow curve for optimal k selection.
    
    Args:
        k_values: List of k values tested
        scores: List of scores for each k
        metric: Metric name (e.g., 'Silhouette', 'Inertia')
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(k_values, scores, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel(metric + ' Score')
    plt.title(f'Elbow Curve - {metric}')
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.show()


def save_figure(filename: str, dpi: int = 300) -> None:
    """
    Save current figure to file.
    
    Args:
        filename: Output filename
        dpi: Resolution
    """
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    print(f"✓ Figure saved to {filename}")

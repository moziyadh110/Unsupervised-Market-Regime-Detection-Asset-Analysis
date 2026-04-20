"""
Model evaluation and regime analysis utilities.
Analyzes cluster characteristics and statistical significance.
"""

import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal
from typing import Dict, Tuple


def analyze_regime_profiles(df: pd.DataFrame, regime_col: str, 
                           feature_cols: list = None) -> pd.DataFrame:
    """
    Calculate mean values for each regime.
    
    Args:
        df: DataFrame with regime assignments
        regime_col: Column name with regime/cluster labels
        feature_cols: Features to analyze (None = all numeric columns)
        
    Returns:
        DataFrame with regime profiles
    """
    if feature_cols is None:
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    profiles = df.groupby(regime_col)[feature_cols].mean()
    
    print(f"✓ Regime Profiles:")
    print(profiles.round(4))
    
    return profiles


def analyze_target_by_regime(df: pd.DataFrame, target_col: str, regime_col: str = 'Regime') -> pd.DataFrame:
    """
    Analyze target variable (e.g., gold returns) across regimes.
    
    Args:
        df: DataFrame with regime assignments
        target_col: Target variable to analyze
        regime_col: Column name with regime labels
        
    Returns:
        DataFrame with descriptive statistics per regime
    """
    analysis = df.groupby(regime_col)[target_col].describe()
    
    print(f"\n✓ {target_col} Analysis by Regime:")
    print(analysis.round(4))
    
    return analysis


def anova_test(df: pd.DataFrame, target_col: str, regime_col: str = 'Regime') -> Dict:
    """
    Perform ANOVA to test if target varies significantly across regimes.
    
    Args:
        df: DataFrame with regime assignments
        target_col: Target variable
        regime_col: Column with regime labels
        
    Returns:
        Dictionary with F-statistic and p-value
    """
    # Remove NaN values
    df_clean = df[[target_col, regime_col]].dropna()
    
    # Get groups
    groups = []
    for regime in sorted(df_clean[regime_col].unique()):
        group_data = df_clean[df_clean[regime_col] == regime][target_col].values
        groups.append(group_data)
    
    # Perform ANOVA
    f_stat, p_value = f_oneway(*groups)
    
    results = {
        'f_statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print(f"\n✓ ANOVA Test ({target_col} vs {regime_col}):")
    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significant (α=0.05): {p_value < 0.05}")
    
    return results


def kruskal_wallis_test(df: pd.DataFrame, target_col: str, regime_col: str = 'Regime') -> Dict:
    """
    Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
    
    Args:
        df: DataFrame with regime assignments
        target_col: Target variable
        regime_col: Column with regime labels
        
    Returns:
        Dictionary with H-statistic and p-value
    """
    # Remove NaN values
    df_clean = df[[target_col, regime_col]].dropna()
    
    # Get groups
    groups = []
    for regime in sorted(df_clean[regime_col].unique()):
        group_data = df_clean[df_clean[regime_col] == regime][target_col].values
        groups.append(group_data)
    
    # Perform Kruskal-Wallis
    h_stat, p_value = kruskal(*groups)
    
    results = {
        'h_statistic': h_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    print(f"\n✓ Kruskal-Wallis Test ({target_col} vs {regime_col}):")
    print(f"  H-statistic: {h_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significant (α=0.05): {p_value < 0.05}")
    
    return results


def regime_correlation_matrix(df: pd.DataFrame, regime_label: int = None) -> pd.DataFrame:
    """
    Compute correlation matrix for a specific regime.
    
    Args:
        df: DataFrame with regime assignments (column named 'Regime')
        regime_label: Regime to analyze (None = all data)
        
    Returns:
        Correlation matrix DataFrame
    """
    if regime_label is not None:
        df_subset = df[df['Regime'] == regime_label]
        title = f"Regime {regime_label}"
    else:
        df_subset = df
        title = "All Data"
    
    # Select numeric columns only
    numeric_df = df_subset.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    
    print(f"\n✓ Correlation Matrix ({title}):")
    print(corr.round(3))
    
    return corr


def compare_regime_metrics(df: pd.DataFrame, regime_col: str = 'Regime', 
                          metrics: list = None) -> pd.DataFrame:
    """
    Compare multiple metrics across regimes.
    
    Args:
        df: DataFrame with regime assignments
        regime_col: Column with regime labels
        metrics: List of metric columns to compare
        
    Returns:
        DataFrame comparing metrics by regime
    """
    if metrics is None:
        metrics = df.select_dtypes(include=[np.number]).columns.tolist()
    
    comparison = df.groupby(regime_col)[metrics].agg(['mean', 'std', 'min', 'max'])
    
    print(f"\n✓ Metrics Comparison by Regime:")
    print(comparison.round(4))
    
    return comparison


def get_regime_statistics(df: pd.DataFrame, regime_col: str = 'Regime') -> Dict:
    """
    Get comprehensive statistics about regimes.
    
    Args:
        df: DataFrame with regime assignments
        regime_col: Column with regime labels
        
    Returns:
        Dictionary with regime statistics
    """
    regime_counts = df[regime_col].value_counts().sort_index()
    regime_percentages = (regime_counts / len(df) * 100).round(2)
    
    stats = {
        'total_samples': len(df),
        'n_regimes': df[regime_col].nunique(),
        'regime_counts': regime_counts.to_dict(),
        'regime_percentages': regime_percentages.to_dict()
    }
    
    print(f"\n✓ Regime Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of regimes: {stats['n_regimes']}")
    for regime, count in regime_counts.items():
        pct = regime_percentages[regime]
        print(f"  Regime {regime}: {count} samples ({pct}%)")
    
    return stats

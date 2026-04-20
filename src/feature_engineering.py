"""
Feature engineering utilities.
Creates log transforms, volatility measures, and derived features.
"""

import pandas as pd
import numpy as np
from typing import List


def apply_log_transformation(df: pd.DataFrame, columns: List[str], epsilon: float = 1e-6) -> pd.DataFrame:
    """
    Apply log transformation to specified columns.
    Handles negative/zero values by adding epsilon.
    
    Args:
        df: Input DataFrame
        columns: List of column names to transform
        epsilon: Small value to add to prevent log(0)
        
    Returns:
        DataFrame with log-transformed columns (as new columns with '_log' suffix)
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            try:
                # Ensure positive values before log
                min_val = df_copy[col].min()
                if min_val <= 0:
                    values = df_copy[col] + abs(min_val) + epsilon
                else:
                    values = df_copy[col]
                
                df_copy[f'{col}_log'] = np.log(values)
                print(f"✓ Created log transformation: {col}_log")
            except Exception as e:
                print(f"✗ Error transforming '{col}': {e}")
    
    return df_copy


def compute_rolling_volatility(df: pd.DataFrame, column: str, window: int = 21) -> pd.DataFrame:
    """
    Calculate rolling standard deviation (volatility).
    
    Args:
        df: Input DataFrame
        column: Column to calculate volatility for
        window: Rolling window size (default 21 = 1 month of trading days)
        
    Returns:
        DataFrame with new volatility column
    """
    df_copy = df.copy()
    if column in df_copy.columns:
        df_copy[f'{column}_VOL'] = df_copy[column].rolling(window=window).std()
        print(f"✓ Created volatility column: {column}_VOL (window={window})")
    else:
        print(f"✗ Column '{column}' not found")
    
    return df_copy


def compute_rolling_returns(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Calculate daily percentage returns.
    
    Args:
        df: Input DataFrame (must be sorted by date)
        column: Column to calculate returns for
        
    Returns:
        DataFrame with new returns column
    """
    df_copy = df.copy()
    if column in df_copy.columns:
        df_copy[f'{column}_RET'] = df_copy[column].pct_change() * 100
        print(f"✓ Created returns column: {column}_RET")
    else:
        print(f"✗ Column '{column}' not found")
    
    return df_copy


def compute_real_yield(df: pd.DataFrame, yield_col: str = '10Y_TREASURY_YIELD', 
                       inflation_col: str = 'CPI_INFLATION') -> pd.DataFrame:
    """
    Calculate real yield = Nominal yield - Inflation rate.
    
    Args:
        df: Input DataFrame
        yield_col: Column with nominal yields
        inflation_col: Column with inflation rates
        
    Returns:
        DataFrame with real yield column
    """
    df_copy = df.copy()
    if yield_col in df_copy.columns and inflation_col in df_copy.columns:
        df_copy['REAL_YIELD'] = df_copy[yield_col] - df_copy[inflation_col]
        print(f"✓ Created REAL_YIELD = {yield_col} - {inflation_col}")
    else:
        print(f"✗ Required columns not found")
    
    return df_copy


def compute_rolling_average(df: pd.DataFrame, column: str, window: int = 5) -> pd.DataFrame:
    """
    Calculate rolling mean (moving average).
    
    Args:
        df: Input DataFrame
        column: Column to calculate rolling average for
        window: Window size
        
    Returns:
        DataFrame with new rolling average column
    """
    df_copy = df.copy()
    if column in df_copy.columns:
        df_copy[f'{column}_{window}D_AVG'] = df_copy[column].rolling(window=window).mean()
        print(f"✓ Created {window}-day average: {column}_{window}D_AVG")
    else:
        print(f"✗ Column '{column}' not found")
    
    return df_copy


def create_price_change_percentage(df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    """
    Create percentage change feature for price columns.
    
    Args:
        df: Input DataFrame
        price_col: Price column name
        
    Returns:
        DataFrame with price change percentage column
    """
    df_copy = df.copy()
    if price_col in df_copy.columns:
        df_copy[f'{price_col}_CHANGE_%'] = df_copy[price_col].pct_change() * 100
        print(f"✓ Created price change %: {price_col}_CHANGE_%")
    else:
        print(f"✗ Column '{price_col}' not found")
    
    return df_copy


def select_numeric_columns(df: pd.DataFrame, exclude: List[str] = None) -> pd.DataFrame:
    """
    Select only numeric columns, optionally excluding specified columns.
    
    Args:
        df: Input DataFrame
        exclude: List of column names to exclude
        
    Returns:
        DataFrame with only numeric columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if exclude:
        numeric_cols = [col for col in numeric_cols if col not in exclude]
    
    print(f"✓ Selected {len(numeric_cols)} numeric columns")
    return df[numeric_cols]


def handle_missing_in_features(df: pd.DataFrame, strategy: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in feature columns.
    
    Args:
        df: Input DataFrame
        strategy: 'drop' or 'forward_fill' or 'backward_fill'
        
    Returns:
        DataFrame with missing values handled
    """
    df_copy = df.copy()
    
    if strategy == 'drop':
        n_before = len(df_copy)
        df_copy = df_copy.dropna()
        n_removed = n_before - len(df_copy)
        print(f"✓ Dropped {n_removed} rows with missing values")
    elif strategy == 'forward_fill':
        df_copy = df_copy.fillna(method='ffill')
        print(f"✓ Applied forward fill strategy")
    elif strategy == 'backward_fill':
        df_copy = df_copy.fillna(method='bfill')
        print(f"✓ Applied backward fill strategy")
    
    return df_copy

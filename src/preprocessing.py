"""
Data cleaning and preprocessing utilities.
Handles type conversion, imputation, and data normalization.
"""

import pandas as pd
import numpy as np
from typing import List, Dict


def convert_columns_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Convert specified columns to numeric type, handling commas and invalid values.
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert
        
    Returns:
        DataFrame with converted columns
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            try:
                # Remove commas if present
                df_copy[col] = df_copy[col].astype(str).str.replace(',', '', regex=False)
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
                print(f"✓ Converted '{col}' to numeric")
            except Exception as e:
                print(f"✗ Error converting '{col}': {e}")
    return df_copy


def impute_missing_values(df: pd.DataFrame, column: str, indices: List[int]) -> pd.DataFrame:
    """
    Impute missing values using rolling average (forward and backward neighbors).
    
    Args:
        df: Input DataFrame
        column: Column name with missing values
        indices: List of row indices to impute
        
    Returns:
        DataFrame with imputed values
    """
    df_copy = df.copy()
    for idx in indices:
        if idx > 0 and idx < len(df_copy) - 1:
            prev_val = df_copy.loc[idx-1, column]
            next_val = df_copy.loc[idx+1, column]
            imputed_val = (prev_val + next_val) / 2
            df_copy.loc[idx, column] = imputed_val
            print(f"✓ Imputed row {idx}: {imputed_val:.2f}")
    return df_copy


def remove_duplicates(df: pd.DataFrame, subset: List[str] = None, keep: str = 'first') -> pd.DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset: Column names to consider for duplicates (None = all columns)
        keep: Which duplicates to keep ('first', 'last', or False)
        
    Returns:
        DataFrame with duplicates removed
    """
    n_before = len(df)
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = n_before - len(df_clean)
    
    if n_removed > 0:
        print(f"✓ Removed {n_removed} duplicate rows")
    else:
        print(f"✓ No duplicates found")
    
    return df_clean


def standardize_date_column(df: pd.DataFrame, column: str = 'DATE', dayfirst: bool = True) -> pd.DataFrame:
    """
    Convert a column to datetime format.
    
    Args:
        df: Input DataFrame
        column: Column name containing dates
        dayfirst: If True, parse as DD/MM/YYYY; else MM/DD/YYYY
        
    Returns:
        DataFrame with datetime column
    """
    df_copy = df.copy()
    try:
        df_copy[column] = pd.to_datetime(df_copy[column], dayfirst=dayfirst)
        print(f"✓ Standardized '{column}' to datetime")
    except Exception as e:
        print(f"✗ Error converting '{column}': {e}")
    return df_copy


def sort_by_date(df: pd.DataFrame, column: str = 'DATE', ascending: bool = True) -> pd.DataFrame:
    """
    Sort DataFrame by date column.
    
    Args:
        df: Input DataFrame
        column: Date column name
        ascending: If True, sort oldest to newest
        
    Returns:
        Sorted DataFrame
    """
    df_sorted = df.sort_values(by=column, ascending=ascending)
    print(f"✓ Sorted by '{column}' (ascending={ascending})")
    return df_sorted


def drop_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Drop specified columns from DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to drop
        
    Returns:
        DataFrame with columns removed
    """
    df_copy = df.copy()
    cols_to_drop = [col for col in columns if col in df_copy.columns]
    if cols_to_drop:
        df_copy = df_copy.drop(columns=cols_to_drop)
        print(f"✓ Dropped columns: {cols_to_drop}")
    return df_copy


def drop_null_rows(df: pd.DataFrame, subset: List[str] = None) -> pd.DataFrame:
    """
    Drop rows with missing values.
    
    Args:
        df: Input DataFrame
        subset: Column names to check for nulls (None = all columns)
        
    Returns:
        DataFrame with null rows removed
    """
    n_before = len(df)
    df_clean = df.dropna(subset=subset)
    n_removed = n_before - len(df_clean)
    
    if n_removed > 0:
        print(f"✓ Removed {n_removed} rows with null values")
    else:
        print(f"✓ No null values found")
    
    return df_clean


def detect_and_handle_outliers(df: pd.DataFrame, column: str, threshold: float = 3.0) -> tuple:
    """
    Detect outliers using z-score method.
    
    Args:
        df: Input DataFrame
        column: Column to check for outliers
        threshold: Z-score threshold (default 3.0)
        
    Returns:
        Tuple of (cleaned DataFrame, outliers DataFrame)
    """
    from scipy.stats import zscore
    
    z_scores = np.abs(zscore(df[column].dropna()))
    outlier_indices = np.where(z_scores > threshold)[0]
    
    outliers = df.iloc[outlier_indices]
    df_clean = df[~df.index.isin(outliers.index)]
    
    if len(outliers) > 0:
        print(f"✓ Found {len(outliers)} outliers in '{column}' (z-score > {threshold})")
    else:
        print(f"✓ No outliers found in '{column}'")
    
    return df_clean, outliers

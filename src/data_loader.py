"""
Data loading and basic preprocessing utilities.
Handles CSV imports and initial data validation.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_csv(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with the loaded data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Loaded {file_path}")
        print(f"  Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"✗ File not found: {file_path}")
        return None
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        return None


def display_basic_info(df: pd.DataFrame) -> None:
    """
    Display basic information about a DataFrame.
    
    Args:
        df: Input DataFrame
    """
    print("\n" + "="*60)
    print("DATA OVERVIEW")
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print("="*60 + "\n")


def check_missing_values(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Find rows with missing values in a specific column.
    
    Args:
        df: Input DataFrame
        column: Column name to check
        
    Returns:
        DataFrame with rows containing NaN in that column
    """
    missing_rows = df[df[column].isna()]
    if len(missing_rows) > 0:
        print(f"\n✓ Found {len(missing_rows)} rows with NaN in '{column}':")
        print(missing_rows[['DATE', column]])
    else:
        print(f"\n✓ No missing values in '{column}'")
    return missing_rows


def save_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        file_path: Output file path
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"✓ Saved to {file_path}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")

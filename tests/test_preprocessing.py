"""
Unit tests for preprocessing functions.
Run with: pytest tests/test_preprocessing.py -v
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import (
    convert_columns_to_numeric,
    impute_missing_values,
    remove_duplicates,
    standardize_date_column,
    sort_by_date,
    drop_columns,
    drop_null_rows
)


class TestConvertColumnsToNumeric:
    """Test numeric column conversion."""
    
    def test_convert_single_column(self):
        df = pd.DataFrame({'A': ['1', '2', '3'], 'B': ['a', 'b', 'c']})
        result = convert_columns_to_numeric(df, ['A'])
        assert result['A'].dtype in [np.float64, np.int64]
        assert result['B'].dtype == object
    
    def test_convert_with_commas(self):
        df = pd.DataFrame({'A': ['1,000', '2,000', '3,000']})
        result = convert_columns_to_numeric(df, ['A'])
        assert result['A'].iloc[0] == 1000.0
    
    def test_nonexistent_column(self):
        df = pd.DataFrame({'A': ['1', '2', '3']})
        result = convert_columns_to_numeric(df, ['B'])
        assert len(result.columns) == 1
        assert result['A'].dtype == object


class TestImputeMissingValues:
    """Test missing value imputation."""
    
    def test_impute_rolling_average(self):
        df = pd.DataFrame({'col': [10.0, np.nan, 30.0, 40.0, 50.0]})
        result = impute_missing_values(df, 'col', [1])
        expected = (10.0 + 30.0) / 2
        assert result.loc[1, 'col'] == expected
    
    def test_multiple_imputation(self):
        df = pd.DataFrame({'col': [10.0, np.nan, 30.0, np.nan, 50.0]})
        result = impute_missing_values(df, 'col', [1, 3])
        assert not result['col'].isna().any()
    
    def test_out_of_bounds_index(self):
        df = pd.DataFrame({'col': [10, 20, 30]})
        result = impute_missing_values(df, 'col', [0, 2])
        assert len(result) == len(df)


class TestRemoveDuplicates:
    """Test duplicate removal."""
    
    def test_remove_exact_duplicates(self):
        df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
        result = remove_duplicates(df)
        assert len(result) == 2
    
    def test_keep_first(self):
        df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 4]})
        result = remove_duplicates(df, keep='first')
        assert result.iloc[0, 0] == 1
        assert len(result) == 2
    
    def test_no_duplicates(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = remove_duplicates(df)
        assert len(result) == len(df)


class TestStandardizeDateColumn:
    """Test date column standardization."""
    
    def test_convert_to_datetime(self):
        df = pd.DataFrame({'DATE': ['01/01/2020', '02/01/2020']})
        result = standardize_date_column(df, 'DATE', dayfirst=True)
        assert pd.api.types.is_datetime64_any_dtype(result['DATE'])
    
    def test_date_parsing_dayfirst(self):
        df = pd.DataFrame({'DATE': ['15/01/2020']})
        result = standardize_date_column(df, 'DATE', dayfirst=True)
        assert result['DATE'].iloc[0].month == 1
        assert result['DATE'].iloc[0].day == 15


class TestSortByDate:
    """Test date sorting."""
    
    def test_sort_ascending(self):
        df = pd.DataFrame({
            'DATE': pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01']),
            'value': [3, 1, 2]
        })
        result = sort_by_date(df, 'DATE', ascending=True)
        assert result['value'].iloc[0] == 1
        assert result['value'].iloc[-1] == 3
    
    def test_sort_descending(self):
        df = pd.DataFrame({
            'DATE': pd.to_datetime(['2020-03-01', '2020-01-01', '2020-02-01']),
            'value': [3, 1, 2]
        })
        result = sort_by_date(df, 'DATE', ascending=False)
        assert result['value'].iloc[0] == 3


class TestDropColumns:
    """Test column dropping."""
    
    def test_drop_existing_column(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = drop_columns(df, ['B'])
        assert 'B' not in result.columns
        assert len(result.columns) == 2
    
    def test_drop_nonexistent_column(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        result = drop_columns(df, ['C'])
        assert len(result.columns) == 2
    
    def test_drop_multiple_columns(self):
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4], 'C': [5, 6]})
        result = drop_columns(df, ['A', 'C'])
        assert len(result.columns) == 1
        assert 'B' in result.columns


class TestDropNullRows:
    """Test null row removal."""
    
    def test_drop_rows_with_nulls(self):
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [4.0, 5.0, 6.0]
        })
        result = drop_null_rows(df)
        assert len(result) == 2
        assert not result.isna().any().any()
    
    def test_drop_subset_nulls(self):
        df = pd.DataFrame({
            'A': [1.0, np.nan, 3.0],
            'B': [np.nan, 5.0, 6.0]
        })
        result = drop_null_rows(df, subset=['A'])
        assert len(result) == 2
        assert np.isnan(result.iloc[0, 1])
    
    def test_no_nulls(self):
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        result = drop_null_rows(df)
        assert len(result) == len(df)

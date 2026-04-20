"""
Unit tests for feature engineering functions.
Run with: pytest tests/test_feature_engineering.py -v
"""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering import (
    apply_log_transformation,
    compute_rolling_volatility,
    compute_rolling_returns,
    compute_real_yield,
    compute_rolling_average
)


class TestLogTransformation:
    """Test log transformation."""
    
    def test_basic_log_transform(self):
        df = pd.DataFrame({'VIX': [10.0, 20.0, 30.0]})
        result = apply_log_transformation(df, ['VIX'])
        assert 'VIX_log' in result.columns
        assert not result['VIX_log'].isna().any()
    
    def test_handle_zero_values(self):
        df = pd.DataFrame({'col': [0.0, 1.0, 2.0]})
        result = apply_log_transformation(df, ['col'])
        assert not result['col_log'].isna().any()
    
    def test_handle_negative_values(self):
        df = pd.DataFrame({'col': [-1.0, 0.0, 1.0]})
        result = apply_log_transformation(df, ['col'])
        assert not result['col_log'].isna().any()


class TestRollingVolatility:
    """Test volatility calculation."""
    
    def test_compute_volatility(self):
        df = pd.DataFrame({'price': [100.0, 102.0, 101.0, 103.0, 105.0]})
        result = compute_rolling_volatility(df, 'price', window=2)
        assert 'price_VOL' in result.columns
        assert result['price_VOL'].iloc[1:].notna().any()
    
    def test_volatility_shape(self):
        df = pd.DataFrame({'price': list(range(100, 110))})
        result = compute_rolling_volatility(df, 'price', window=5)
        assert len(result) == len(df)
    
    def test_invalid_column(self):
        df = pd.DataFrame({'A': [1.0, 2.0, 3.0]})
        result = compute_rolling_volatility(df, 'B', window=2)
        assert 'B_VOL' not in result.columns


class TestRollingReturns:
    """Test returns calculation."""
    
    def test_compute_returns(self):
        df = pd.DataFrame({'price': [100.0, 105.0, 110.0]})
        result = compute_rolling_returns(df, 'price')
        assert 'price_RET' in result.columns
        assert len(result) == len(df)
    
    def test_returns_first_value_nan(self):
        df = pd.DataFrame({'price': [100.0, 105.0, 110.0]})
        result = compute_rolling_returns(df, 'price')
        assert np.isnan(result['price_RET'].iloc[0])
    
    def test_returns_calculation_correct(self):
        df = pd.DataFrame({'price': [100.0, 110.0]})
        result = compute_rolling_returns(df, 'price')
        expected = (110.0 - 100.0) / 100.0 * 100
        assert np.isclose(result['price_RET'].iloc[1], expected)


class TestRealYield:
    """Test real yield calculation."""
    
    def test_compute_real_yield(self):
        df = pd.DataFrame({
            '10Y_TREASURY_YIELD': [2.0, 2.5, 3.0],
            'CPI_INFLATION': [1.0, 1.5, 2.0]
        })
        result = compute_real_yield(df, '10Y_TREASURY_YIELD', 'CPI_INFLATION')
        assert 'REAL_YIELD' in result.columns
        assert result['REAL_YIELD'].iloc[0] == 1.0
    
    def test_negative_real_yield(self):
        df = pd.DataFrame({
            '10Y_TREASURY_YIELD': [1.0],
            'CPI_INFLATION': [2.0]
        })
        result = compute_real_yield(df, '10Y_TREASURY_YIELD', 'CPI_INFLATION')
        assert result['REAL_YIELD'].iloc[0] == -1.0
    
    def test_missing_columns(self):
        df = pd.DataFrame({'A': [1.0, 2.0]})
        result = compute_real_yield(df, 'B', 'C')
        assert 'REAL_YIELD' not in result.columns


class TestRollingAverage:
    """Test moving average calculation."""
    
    def test_compute_moving_average(self):
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0, 4.0, 5.0]})
        result = compute_rolling_average(df, 'col', window=2)
        assert 'col_2D_AVG' in result.columns
        assert len(result) == len(df)
    
    def test_moving_average_values(self):
        df = pd.DataFrame({'col': [2.0, 4.0, 6.0]})
        result = compute_rolling_average(df, 'col', window=2)
        assert np.isclose(result['col_2D_AVG'].iloc[1], 3.0)
    
    def test_invalid_column(self):
        df = pd.DataFrame({'A': [1.0, 2.0]})
        result = compute_rolling_average(df, 'B')
        assert 'B_5D_AVG' not in result.columns

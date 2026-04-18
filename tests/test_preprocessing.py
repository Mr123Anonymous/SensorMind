"""
Basic unit test for data preprocessing.
"""

import pytest
import numpy as np
import pandas as pd
from src.data.preprocessing import (
    handle_missing_values,
    handle_outliers,
    normalize_features,
    create_lag_features,
    create_cyclical_features,
)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "value": np.random.randn(100) + 100,
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
    }, index=dates)
    return df


def test_handle_missing_values(sample_dataframe):
    """Test missing value handling."""
    df = sample_dataframe.copy()
    df.loc[5, "value"] = np.nan
    df.loc[10, "feature_1"] = np.nan
    
    # Interpolate
    df_filled = handle_missing_values(df, method="interpolate")
    assert df_filled.isnull().sum().sum() == 0
    
    # Forward fill
    df_filled = handle_missing_values(df, method="forward_fill")
    assert df_filled.isnull().sum().sum() == 0


def test_handle_outliers(sample_dataframe):
    """Test outlier handling."""
    df = sample_dataframe.copy()
    df.loc[0, "value"] = 1000  # Add outlier
    
    df_cleaned = handle_outliers(df, method="iqr")
    assert df_cleaned["value"].max() < 1000  # Outlier should be clipped


def test_normalize_features(sample_dataframe):
    """Test feature normalization."""
    df = sample_dataframe.copy()
    
    # Standard scaling
    df_normalized, scaler = normalize_features(df, method="standard")
    assert np.abs(df_normalized["value"].mean()) < 0.01  # Mean close to 0
    assert np.abs(df_normalized["value"].std() - 1.0) < 0.01  # Std close to 1


def test_create_lag_features(sample_dataframe):
    """Test lag feature creation."""
    df = sample_dataframe.copy()
    
    df_with_lags = create_lag_features(df, "value", lags=[1, 7, 24])
    
    # Should have 3 additional columns
    assert len(df_with_lags.columns) == len(df.columns) + 3
    assert "value_lag_1" in df_with_lags.columns
    assert "value_lag_7" in df_with_lags.columns
    assert "value_lag_24" in df_with_lags.columns


def test_create_cyclical_features(sample_dataframe):
    """Test cyclical feature creation."""
    df = sample_dataframe.copy()
    
    df_with_cyclical = create_cyclical_features(df)
    
    # Should have sin/cos features for hour, day of week, month
    expected_features = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    for feature in expected_features:
        assert feature in df_with_cyclical.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Feature engineering utilities for Faclon ML Portfolio.
Creates custom features optimized for each task type.
"""

from typing import Optional
import pandas as pd
import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


def create_time_series_features(
    df: pd.DataFrame,
    target_column: str,
    lags: list = None,
    rolling_windows: list = None,
    rolling_stats: list = None,
    include_cyclical: bool = True
) -> pd.DataFrame:
    """
    Create comprehensive time-series features.
    
    Args:
        df: Input dataframe
        target_column: Column to create features for
        lags: List of lag values
        rolling_windows: List of rolling window sizes
        rolling_stats: List of rolling statistics
        include_cyclical: Whether to create cyclical features
    
    Returns:
        Dataframe with engineered features
    """
    df = df.copy()
    
    # Create lag features
    if lags is not None:
        for lag in lags:
            df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    
    # Create rolling features
    if rolling_windows is not None and rolling_stats is not None:
        for window in rolling_windows:
            for stat in rolling_stats:
                col_name = f"{target_column}_rolling_{window}_{stat}"
                if stat == "mean":
                    df[col_name] = df[target_column].rolling(window).mean()
                elif stat == "std":
                    df[col_name] = df[target_column].rolling(window).std()
                elif stat == "min":
                    df[col_name] = df[target_column].rolling(window).min()
                elif stat == "max":
                    df[col_name] = df[target_column].rolling(window).max()
    
    # Create cyclical features
    if include_cyclical and isinstance(df.index, pd.DatetimeIndex):
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    
    logger.info(f"Created {len(df.columns) - 1} engineered features")
    
    return df.dropna()

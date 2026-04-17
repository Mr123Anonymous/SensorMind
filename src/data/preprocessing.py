"""
Data preprocessing utilities for Faclon ML Portfolio.
Handles cleaning, normalization, and feature engineering.
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from src.config import PREPROCESSING_CONFIG, MODEL_CONFIG, FEATURE_CONFIG
from src.utils import setup_logger

logger = setup_logger(__name__)


def handle_missing_values(
    df: pd.DataFrame,
    method: str = "interpolate",
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in dataframe.
    
    Args:
        df: Input dataframe
        method: 'interpolate', 'forward_fill', 'drop', or 'value'
        fill_value: Value to fill with (if method='value')
    
    Returns:
        Dataframe with missing values handled
    """
    df = df.copy()
    
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        logger.info(f"Found missing values:\n{missing_counts[missing_counts > 0]}")
        
        if method == "interpolate":
            df = df.interpolate(method="linear", limit_direction="both")
        elif method == "forward_fill":
            df = df.fillna(method="ffill").fillna(method="bfill")
        elif method == "drop":
            df = df.dropna()
        elif method == "value" and fill_value is not None:
            df = df.fillna(fill_value)
        
        logger.info(f"Remaining missing values: {df.isnull().sum().sum()}")
    
    return df


def handle_outliers(
    df: pd.DataFrame,
    method: str = "iqr",
    threshold: float = 3,
    columns: Optional[list] = None
) -> pd.DataFrame:
    """
    Handle outliers in dataframe.
    
    Args:
        df: Input dataframe
        method: 'iqr' or 'zscore'
        threshold: Z-score threshold (for zscore method)
        columns: List of columns to process (if None, process all)
    
    Returns:
        Dataframe with outliers clipped or removed
    """
    df = df.copy()
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    for col in columns:
        if method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower_bound, upper_bound)
            
        elif method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df[col] = df[col][z_scores <= threshold]
    
    logger.info(f"Processed outliers using {method} method")
    return df


def normalize_features(
    df: pd.DataFrame,
    method: str = "standard",
    fit_data: Optional[pd.DataFrame] = None
) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Normalize numerical features.
    
    Args:
        df: Input dataframe
        method: 'standard' or 'minmax'
        fit_data: Data to fit scaler (if None, fit on current data)
    
    Returns:
        Tuple of (normalized_dataframe, scaler)
    """
    df = df.copy()
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    if fit_data is not None:
        scaler.fit(fit_data[numeric_columns])
    else:
        scaler.fit(df[numeric_columns])
    
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    logger.info(f"Normalized features using {method} method")
    
    return df, scaler


def create_lag_features(
    df: pd.DataFrame,
    target_column: str,
    lags: list = None
) -> pd.DataFrame:
    """
    Create lag features for time-series data.
    
    Args:
        df: Input dataframe
        target_column: Column to create lags for
        lags: List of lag values (if None, uses config)
    
    Returns:
        Dataframe with lag features added
    """
    df = df.copy()
    
    if lags is None:
        lags = FEATURE_CONFIG["lag_features"]
    
    for lag in lags:
        df[f"{target_column}_lag_{lag}"] = df[target_column].shift(lag)
    
    logger.info(f"Created {len(lags)} lag features for {target_column}")
    
    return df.dropna()


def create_rolling_features(
    df: pd.DataFrame,
    target_column: str,
    windows: list = None,
    stats: list = None
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: Input dataframe
        target_column: Column to create rolling features for
        windows: List of window sizes (if None, uses config)
        stats: List of statistics ('mean', 'std', 'min', 'max')
    
    Returns:
        Dataframe with rolling features added
    """
    df = df.copy()
    
    if windows is None:
        windows = FEATURE_CONFIG["rolling_windows"]
    if stats is None:
        stats = FEATURE_CONFIG["rolling_stats"]
    
    for window in windows:
        for stat in stats:
            col_name = f"{target_column}_rolling_{window}_{stat}"
            if stat == "mean":
                df[col_name] = df[target_column].rolling(window).mean()
            elif stat == "std":
                df[col_name] = df[target_column].rolling(window).std()
            elif stat == "min":
                df[col_name] = df[target_column].rolling(window).min()
            elif stat == "max":
                df[col_name] = df[target_column].rolling(window).max()
    
    logger.info(f"Created rolling features with windows {windows} and stats {stats}")
    
    return df.dropna()


def create_cyclical_features(
    df: pd.DataFrame,
    index_column: str = None
) -> pd.DataFrame:
    """
    Create cyclical features from datetime index (sin/cos encoding).
    
    Args:
        df: Input dataframe with datetime index
        index_column: Column name if not using index
    
    Returns:
        Dataframe with cyclical features added
    """
    df = df.copy()
    
    if index_column is not None:
        df[index_column] = pd.to_datetime(df[index_column])
        date_series = df[index_column]
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Dataframe must have DatetimeIndex or specify index_column")
        date_series = df.index
    
    # Hour
    df["hour_sin"] = np.sin(2 * np.pi * date_series.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * date_series.hour / 24)
    
    # Day of week
    df["dow_sin"] = np.sin(2 * np.pi * date_series.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * date_series.dayofweek / 7)
    
    # Month
    df["month_sin"] = np.sin(2 * np.pi * date_series.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * date_series.month / 12)
    
    logger.info("Created cyclical features (hour, day of week, month)")
    
    return df


def preprocess_forecast_data(
    df: pd.DataFrame,
    target_column: str = "generation",
    test_size: float = 0.1,
    val_size: float = 0.1,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess data for time-series forecasting.
    
    Args:
        df: Input dataframe with time-series data
        target_column: Name of target column
        test_size: Proportion of test set
        val_size: Proportion of validation set
        scale: Whether to scale features
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    logger.info("Preprocessing data for forecasting...")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Create cyclical features
    df = create_cyclical_features(df)
    
    # Create lag features
    df = create_lag_features(df, target_column)
    
    # Create rolling features
    df = create_rolling_features(df, target_column)
    
    # Separate features and target
    y = df[target_column].values
    X = df.drop(columns=[target_column]).values
    
    # Split data (temporal split)
    train_size = 1 - test_size - val_size
    n = len(df)
    train_idx = int(n * train_size)
    val_idx = int(n * (train_size + val_size))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def preprocess_anomaly_data(
    df: pd.DataFrame,
    labels_df: Optional[pd.DataFrame] = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Preprocess data for anomaly detection.
    
    Args:
        df: Input dataframe with time-series data
        labels_df: DataFrame with anomaly labels (optional)
        test_size: Proportion of test set
        val_size: Proportion of validation set
        scale: Whether to scale features
    
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    logger.info("Preprocessing data for anomaly detection...")
    
    # Handle missing values
    df = handle_missing_values(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Get features
    X = df.values
    
    # Get labels if provided
    if labels_df is not None:
        y = labels_df.values.ravel()
    else:
        # Assume no labels (unsupervised)
        y = np.zeros(len(df))
    
    # Split data
    train_size = 1 - test_size - val_size
    n = len(df)
    train_idx = int(n * train_size)
    val_idx = int(n * (train_size + val_size))
    
    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    # Scale features
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

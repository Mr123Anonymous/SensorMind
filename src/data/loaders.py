"""
Data loading utilities for Faclon ML Portfolio.
Handles loading of datasets from various sources.
"""

from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import numpy as np

from src.config import DATA_RAW_PATH
from src.utils import setup_logger

logger = setup_logger(__name__)


def load_pgcb_dataset(
    file_path: Optional[Path] = None,
    download: bool = False
) -> pd.DataFrame:
    """
    Load PGCB (Power Generation Control Board) dataset.
    Bangladesh electricity generation data.
    
    Args:
        file_path: Path to CSV file (if None, uses default location)
        download: Whether to download dataset if not found
    
    Returns:
        DataFrame with PGCB data
    """
    if file_path is None:
        file_path = DATA_RAW_PATH / "pgcb_generation.csv"
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        if download:
            logger.info("Downloading PGCB dataset...")
            # TODO: Implement automatic download from UCI
            raise NotImplementedError("Automatic download not yet implemented")
        else:
            logger.warning(f"Dataset not found at {file_path}")
            logger.info("Please download from: https://archive.ics.uci.edu/dataset/1175")
            return None
    
    logger.info(f"Loading PGCB dataset from {file_path}")
    df = pd.read_csv(file_path)
    
    # Basic validation
    logger.info(f"Loaded dataset shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    logger.info(f"Missing values:\n{df.isnull().sum()}")
    
    return df


def load_anomaly_dataset(
    file_path: Optional[Path] = None,
    download: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load Yahoo time-series anomaly detection dataset.
    Pre-labeled time-series with known anomalies.
    
    Args:
        file_path: Path to CSV file (if None, uses default location)
        download: Whether to download dataset if not found
    
    Returns:
        Tuple of (data_df, labels_df)
    """
    if file_path is None:
        data_file = DATA_RAW_PATH / "anomaly_data.csv"
        labels_file = DATA_RAW_PATH / "anomaly_labels.csv"
    else:
        data_file = Path(file_path)
        labels_file = data_file.parent / "labels.csv"
    
    if not data_file.exists():
        if download:
            logger.info("Downloading anomaly dataset...")
            # TODO: Implement automatic download from HuggingFace
            raise NotImplementedError("Automatic download not yet implemented")
        else:
            logger.warning(f"Dataset not found at {data_file}")
            logger.info("Please download from: https://huggingface.co/datasets/YahooResearch/...")
            return None, None
    
    logger.info(f"Loading anomaly dataset from {data_file}")
    data_df = pd.read_csv(data_file)
    labels_df = pd.read_csv(labels_file) if labels_file.exists() else None
    
    logger.info(f"Data shape: {data_df.shape}")
    if labels_df is not None:
        logger.info(f"Labels shape: {labels_df.shape}")
    
    return data_df, labels_df


def load_synthetic_time_series(
    n_samples: int = 1000,
    n_features: int = 5,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic time-series data for testing.
    
    Args:
        n_samples: Number of time steps
        n_features: Number of features
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with synthetic time-series
    """
    np.random.seed(seed)
    
    # Create time index
    dates = pd.date_range(start="2023-01-01", periods=n_samples, freq="H")
    
    # Generate synthetic data with trend, seasonality, and noise
    t = np.arange(n_samples)
    data = {}
    
    for i in range(n_features):
        # Trend
        trend = 0.001 * t
        # Seasonality (24-hour pattern)
        seasonality = 2 * np.sin(2 * np.pi * t / 24)
        # Weekly pattern
        weekly = 1 * np.sin(2 * np.pi * t / (24 * 7))
        # Noise
        noise = np.random.normal(0, 0.5, n_samples)
        
        data[f"feature_{i}"] = trend + seasonality + weekly + noise + np.random.normal(100 * i, 10)
    
    df = pd.DataFrame(data, index=dates)
    return df


if __name__ == "__main__":
    # Test loading
    logger.info("Testing data loading utilities...")
    
    # Generate synthetic data for testing
    synthetic_data = load_synthetic_time_series()
    print(synthetic_data.head())

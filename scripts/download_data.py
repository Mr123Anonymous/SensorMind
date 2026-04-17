"""
Script to download and prepare datasets.
Run this before starting model training.
"""

from pathlib import Path
import logging
from src.data.loaders import load_synthetic_time_series, load_pgcb_dataset
from src.config import DATA_RAW_PATH
from src.utils import setup_logger

logger = setup_logger(__name__)


def download_pgcb_dataset():
    """
    Download PGCB Grid dataset.
    Manual download required: https://archive.ics.uci.edu/dataset/1175
    """
    logger.info("PGCB Dataset Download Instructions:")
    logger.info("1. Visit: https://archive.ics.uci.edu/dataset/1175")
    logger.info("2. Download 'df_generation_electricity_hourly.csv'")
    logger.info(f"3. Save to: {DATA_RAW_PATH}/pgcb_generation.csv")
    logger.info("")


def download_anomaly_dataset():
    """
    Download Yahoo Anomaly Dataset.
    Manual download required: https://huggingface.co/datasets/YahooResearch/...
    """
    logger.info("Yahoo Anomaly Dataset Download Instructions:")
    logger.info("1. Visit: https://huggingface.co/datasets/YahooResearch/ydata-labeled-time-series-anomalies-v1_0")
    logger.info("2. Download the dataset")
    logger.info(f"3. Extract to: {DATA_RAW_PATH}/")
    logger.info("")


def generate_synthetic_data():
    """Generate synthetic data for testing."""
    logger.info("Generating synthetic time-series data...")
    
    # Generate synthetic forecast data
    synthetic_forecast = load_synthetic_time_series(
        n_samples=1000,
        n_features=5,
        seed=42
    )
    synthetic_forecast.to_csv(
        DATA_RAW_PATH / "synthetic_forecast.csv"
    )
    logger.info(f"✓ Saved synthetic forecast data: {DATA_RAW_PATH}/synthetic_forecast.csv")
    
    # Generate synthetic anomaly data
    synthetic_anomaly = load_synthetic_time_series(
        n_samples=500,
        n_features=3,
        seed=123
    )
    synthetic_anomaly.to_csv(
        DATA_RAW_PATH / "synthetic_anomaly.csv"
    )
    logger.info(f"✓ Saved synthetic anomaly data: {DATA_RAW_PATH}/synthetic_anomaly.csv")


if __name__ == "__main__":
    DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Faclon ML Portfolio - Data Download Script")
    logger.info("=" * 60)
    logger.info("")
    
    # Download instructions
    download_pgcb_dataset()
    download_anomaly_dataset()
    
    # Generate synthetic data
    logger.info("")
    generate_synthetic_data()
    
    logger.info("")
    logger.info("✓ Data preparation complete!")
    logger.info("Next step: Run 'jupyter notebook' to start EDA")

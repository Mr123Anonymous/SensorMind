"""
Configuration module for SensorMind ML Portfolio.
Centralized settings for reproducibility and easy parameter management.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW_PATH = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed"
MODELS_PATH = PROJECT_ROOT / "models"
REPORTS_PATH = PROJECT_ROOT / "reports"
FIGURES_PATH = REPORTS_PATH / "figures"
NOTEBOOKS_PATH = PROJECT_ROOT / "notebooks"
# Backward-compatible alias used in early notebook cells.
PLOTS_PATH = FIGURES_PATH

# Create directories if they don't exist
for path in [DATA_RAW_PATH, DATA_PROCESSED_PATH, MODELS_PATH, REPORTS_PATH, FIGURES_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Random seeds for reproducibility
RANDOM_SEED = 42
import random
import numpy as np
try:
    import torch
except ImportError:  # Optional for lightweight notebook environments.
    torch = None

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch is not None:
    torch.manual_seed(RANDOM_SEED)

# Model training configurations
MODEL_CONFIG: Dict[str, Any] = {
    # Time-Series Forecasting (PGCB)
    "forecast": {
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1,
        "forecast_horizon": 24,  # Predict next 24 hours
        "lookback_window": 168,  # Use 7 days of history
        "arima_order": (1, 1, 1),  # (p, d, q)
        "seasonal_order": (1, 1, 1, 24),  # (P, D, Q, s)
        "xgboost_params": {
            "max_depth": 5,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "seed": RANDOM_SEED,
        },
        "lstm_config": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
        },
    },
    # Anomaly Detection
    "anomaly": {
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "contamination": 0.05,  # Expected % of anomalies
        "autoencoder_config": {
            "encoder_dims": [64, 32, 16],
            "decoder_dims": [16, 32, 64],
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 50,
            "dropout": 0.2,
        },
        "isolation_forest_params": {
            "n_estimators": 100,
            "contamination": 0.05,
            "seed": RANDOM_SEED,
        },
    },
    # NLP Sentiment Classification
    "nlp": {
        "train_size": 0.8,
        "val_size": 0.1,
        "test_size": 0.1,
        "max_seq_length": 512,
        "bert_model": "bert-base-uncased",
        "learning_rate": 2e-5,
        "batch_size": 32,
        "epochs": 3,
    },
}

# Feature engineering parameters
FEATURE_CONFIG: Dict[str, Any] = {
    "lag_features": [1, 7, 24, 168],  # 1 hour, 1 day, 1 day, 1 week
    "rolling_windows": [24, 168],  # 1 day, 1 week
    "rolling_stats": ["mean", "std", "min", "max"],
}

# Dataset URLs (for automatic download)
DATASET_URLS: Dict[str, str] = {
    "pgcb": "https://archive.ics.uci.edu/ml/machine-learning-databases/...",
    "anomaly": "https://huggingface.co/datasets/YahooResearch/...",
}

# Logging configuration
LOGGING_CONFIG: Dict[str, Any] = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "stream": "ext://sys.stdout",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

# MLflow configuration
MLFLOW_CONFIG: Dict[str, Any] = {
    "experiment_name": "SensorMind-ml-portfolio",
    "tracking_uri": "sqlite:///mlruns/mlflow.db",
    "registry_uri": "sqlite:///mlruns/registry.db",
}

# Streamlit configuration for app
STREAMLIT_CONFIG: Dict[str, Any] = {
    "page_title": "SensorMind ML Portfolio",
    "page_icon": ":bar_chart:",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#1f77b4",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730",
        "font": "sans serif",
    },
}

# Data preprocessing parameters
PREPROCESSING_CONFIG: Dict[str, Any] = {
    "handle_missing": "interpolate",  # 'interpolate', 'forward_fill', 'drop'
    "outlier_method": "iqr",  # 'iqr', 'zscore'
    "outlier_threshold": 3,  # For zscore
    "scaling_method": "standard",  # 'standard', 'minmax'
    "normalize_features": True,
}

# API configuration
API_CONFIG: Dict[str, Any] = {
    "host": os.environ.get("API_HOST", "0.0.0.0"),
    "port": int(os.environ.get("API_PORT", 8000)),
    "reload": True,
    "workers": 1,
}


def get_config_section(section: str) -> Dict[str, Any]:
    """Get configuration for a specific section."""
    section_map = {
        "forecast": MODEL_CONFIG["forecast"],
        "anomaly": MODEL_CONFIG["anomaly"],
        "nlp": MODEL_CONFIG["nlp"],
        "features": FEATURE_CONFIG,
        "mlflow": MLFLOW_CONFIG,
        "api": API_CONFIG,
    }
    return section_map.get(section, {})


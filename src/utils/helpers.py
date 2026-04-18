"""
Helper utilities for logging, metrics, and common operations.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, mean_absolute_percentage_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)


def setup_logger(
    name: str,
    log_file: Path = None,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Setup a logger with console and optional file output.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level (default: INFO)
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_logger(name: str) -> logging.Logger:
    """Create a simple logger."""
    return logging.getLogger(name)


def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
    
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    
    # R2 score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "MAPE": float(mape),
        "R2": float(r2),
    }


def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray = None,
    average: str = "weighted"
) -> Dict[str, Any]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
        average: Averaging method for multi-class ('weighted', 'macro', 'micro')
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }
    
    # ROC-AUC (for binary classification or multi-class OvR)
    try:
        if y_proba is not None:
            if y_proba.shape[1] == 2:
                metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["ROC_AUC"] = float(roc_auc_score(
                    y_true, y_proba, multi_class="ovr", average=average
                ))
    except Exception as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    # Confusion matrix
    metrics["Confusion_Matrix"] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def get_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str = "regression",
    y_proba: np.ndarray = None
) -> Dict[str, Any]:
    """
    Get metrics based on task type.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        task_type: 'regression' or 'classification'
        y_proba: Predicted probabilities (for classification)
    
    Returns:
        Dictionary of metrics
    """
    if task_type == "regression":
        return get_regression_metrics(y_true, y_pred)
    elif task_type == "classification":
        return get_classification_metrics(y_true, y_pred, y_proba)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")


def save_metrics(
    metrics: Dict[str, Any],
    output_path: Path,
    model_name: str = "model"
) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Path to save JSON
        model_name: Name of the model
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "metrics": metrics,
    }
    
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    task_type: str = "regression",
    save_path: Path = None
) -> None:
    """
    Plot comparison of metrics across models.
    
    Args:
        metrics_dict: Dictionary mapping model names to metrics
        task_type: 'regression' or 'classification'
        save_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt
    
    df = pd.DataFrame(metrics_dict).T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind="bar", ax=ax)
    ax.set_title(f"Model Comparison - {task_type.capitalize()} Metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("Model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    plt.close()


def ensure_path_exists(path: Path) -> Path:
    """Ensure a directory path exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataframe(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Load a dataframe from various formats.
    
    Args:
        file_path: Path to file
        **kwargs: Additional arguments for read functions
    
    Returns:
        Loaded dataframe
    """
    file_path = Path(file_path)
    
    if file_path.suffix == ".csv":
        return pd.read_csv(file_path, **kwargs)
    elif file_path.suffix == ".parquet":
        return pd.read_parquet(file_path, **kwargs)
    elif file_path.suffix == ".json":
        return pd.read_json(file_path, **kwargs)
    elif file_path.suffix in [".xls", ".xlsx"]:
        return pd.read_excel(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")


def save_dataframe(
    df: pd.DataFrame,
    file_path: Path,
    format: str = "csv",
    **kwargs
) -> None:
    """
    Save a dataframe to various formats.
    
    Args:
        df: Dataframe to save
        file_path: Path to save file
        format: Format ('csv', 'parquet', 'json')
        **kwargs: Additional arguments for write functions
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "csv":
        df.to_csv(file_path, index=False, **kwargs)
    elif format == "parquet":
        df.to_parquet(file_path, index=False, **kwargs)
    elif format == "json":
        df.to_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")

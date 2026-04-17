"""
Model evaluation module.
Calculate and compare metrics across models.
"""

import logging
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

from src.utils import setup_logger

logger = setup_logger(__name__)


def evaluate_forecast_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate forecast model.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        model_name: Model name for logging
    
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
    }
    
    logger.info(f"\n{model_name} Forecast Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


def evaluate_anomaly_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model"
) -> Dict[str, float]:
    """
    Evaluate anomaly detection model.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        model_name: Model name for logging
    
    Returns:
        Dictionary of metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }
    
    logger.info(f"\n{model_name} Anomaly Detection Metrics:")
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
    
    return metrics


def compare_models(
    results_dict: Dict[str, Dict[str, float]],
    task_type: str = "forecast"
) -> str:
    """
    Compare multiple models.
    
    Args:
        results_dict: Dictionary mapping model names to metrics
        task_type: 'forecast' or 'anomaly'
    
    Returns:
        Formatted comparison string
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Model Comparison - {task_type.upper()}")
    logger.info(f"{'='*60}")
    
    # Find best model
    if task_type == "forecast":
        best_metric = "RMSE"
        best_model = min(results_dict.items(), key=lambda x: x[1][best_metric])
    else:
        best_metric = "F1"
        best_model = max(results_dict.items(), key=lambda x: x[1][best_metric])
    
    logger.info(f"\nBest Model: {best_model[0]} ({best_metric}: {best_model[1][best_metric]:.4f})")
    
    return best_model[0]

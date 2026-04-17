"""
Model prediction module.
Inference with trained models.
"""

import logging
from pathlib import Path
from typing import Union, Dict, Any
import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)


def predict_forecast(
    model: Any,
    X: np.ndarray,
    scaler: Any = None
) -> np.ndarray:
    """
    Make forecast predictions.
    
    Args:
        model: Trained model
        X: Input features
        scaler: Feature scaler (for inverse transform)
    
    Returns:
        Predictions
    """
    logger.info("Making forecast predictions...")
    predictions = model.predict(X)
    
    if scaler is not None:
        predictions = scaler.inverse_transform(predictions)
    
    return predictions


def predict_anomaly(
    model: Any,
    X: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Detect anomalies.
    
    Args:
        model: Trained model
        X: Input features
        threshold: Anomaly threshold
    
    Returns:
        Dictionary with predictions and scores
    """
    logger.info("Detecting anomalies...")
    
    predictions = model.predict(X)
    
    results = {
        "anomaly": predictions,
        "is_anomaly": (predictions > threshold).astype(int),
    }
    
    return results


def predict_sentiment(
    model: Any,
    texts: list
) -> Dict[str, Any]:
    """
    Predict sentiment.
    
    Args:
        model: Trained NLP model
        texts: List of texts
    
    Returns:
        Dictionary with predictions and probabilities
    """
    logger.info(f"Predicting sentiment for {len(texts)} texts...")
    
    predictions = model.predict(texts)
    
    results = {
        "predictions": predictions,
        "texts": texts,
    }
    
    return results

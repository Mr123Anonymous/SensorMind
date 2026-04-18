"""
Model prediction module.
Inference with trained models.
"""

from typing import Dict, Any
import numpy as np
import torch

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
    if hasattr(model, "predict"):
        predictions = model.predict(X)
    else:
        # Support torch modules used in the project baselines.
        model.eval()
        with torch.no_grad():
            predictions = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()

    predictions = np.asarray(predictions).reshape(-1)
    
    if scaler is not None and hasattr(scaler, "inverse_transform"):
        # Inverse transform requires 2D arrays.
        predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).reshape(-1)
    
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

    if hasattr(model, "score_samples"):
        # IsolationForest and similar models.
        raw_predictions = model.predict(X)
        anomaly_scores = -model.score_samples(X)
        is_anomaly = (raw_predictions == -1).astype(int)
    elif hasattr(model, "predict"):
        raw_predictions = model.predict(X)
        anomaly_scores = np.asarray(raw_predictions).reshape(-1)
        is_anomaly = (anomaly_scores > threshold).astype(int)
    else:
        # Autoencoder-like torch model fallback.
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32)
            recon = model(inputs)
            anomaly_scores = torch.mean((recon - inputs) ** 2, dim=1).cpu().numpy()
        raw_predictions = anomaly_scores
        is_anomaly = (anomaly_scores > threshold).astype(int)

    results = {
        "anomaly": np.asarray(raw_predictions),
        "anomaly_score": np.asarray(anomaly_scores),
        "is_anomaly": np.asarray(is_anomaly),
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
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
    
    results = {
        "predictions": predictions,
        "texts": texts,
        "probabilities": probabilities,
    }
    
    return results

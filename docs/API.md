# API / Inference Interfaces

This project currently exposes model inference through Streamlit pages and Python modules.

## Forecast Inference
Module: src/models/predict.py
Function: predict_forecast(model, X, scaler=None)

Input:
- model: trained sklearn or torch model
- X: numpy array [n_samples, n_features]
- scaler: optional scaler with inverse_transform

Output:
- numpy array of predictions shape [n_samples]

## Anomaly Inference
Module: src/models/predict.py
Function: predict_anomaly(model, X, threshold=0.5)

Input:
- model: trained anomaly model (IsolationForest or torch autoencoder)
- X: numpy array [n_samples, n_features]
- threshold: score threshold

Output dictionary:
- anomaly: raw model output
- anomaly_score: normalized anomaly scores
- is_anomaly: binary predictions (0/1)

## Sentiment Inference
Module: src/models/predict.py
Function: predict_sentiment(model, texts)

Input:
- model: trained text pipeline
- texts: list[str]

Output dictionary:
- predictions: list/array of class labels
- probabilities: optional probability matrix when supported
- texts: original input texts

## Future FastAPI Extension (Planned)
Potential endpoints:
- POST /predict/forecast
- POST /predict/anomaly
- POST /predict/sentiment
- GET /health

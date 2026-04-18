"""Tests for Phase 3 model training baselines."""

from __future__ import annotations

import numpy as np

from src.models.train import train_anomaly_models, train_forecast_models, train_nlp_models


def _build_forecast_split(n_samples: int = 120, n_features: int = 6):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = X[:, 0] * 0.5 - X[:, 1] * 0.25 + rng.normal(scale=0.1, size=n_samples)

    train_end = 80
    val_end = 100
    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


def _build_anomaly_split(n_samples: int = 150, n_features: int = 5):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(n_samples, n_features))
    y = np.zeros(n_samples, dtype=int)
    anomaly_idx = rng.choice(n_samples, size=20, replace=False)
    X[anomaly_idx] += rng.normal(loc=3.0, scale=0.4, size=(len(anomaly_idx), n_features))
    y[anomaly_idx] = 1

    train_end = 100
    val_end = 125
    return X[:train_end], X[train_end:val_end], X[val_end:], y[:train_end], y[train_end:val_end], y[val_end:]


def test_train_forecast_models_returns_summary():
    X_train, X_val, X_test, y_train, y_val, y_test = _build_forecast_split()
    result = train_forecast_models(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        save_artifacts=False,
        mlp_epochs=3,
    )

    assert result.task == "forecast"
    assert result.best_model in {"ridge", "random_forest", "mlp"}
    assert set(result.metrics.keys()) == {"ridge", "random_forest", "mlp"}
    assert result.metrics[result.best_model]["RMSE"] >= 0


def test_train_anomaly_models_returns_summary():
    X_train, X_val, X_test, y_train, y_val, y_test = _build_anomaly_split()
    result = train_anomaly_models(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        save_artifacts=False,
        ae_epochs=3,
    )

    assert result.task == "anomaly"
    assert result.best_model in {"isolation_forest", "autoencoder"}
    assert set(result.metrics.keys()) == {"isolation_forest", "autoencoder"}
    assert result.metrics[result.best_model]["F1"] >= 0


def test_train_nlp_models_returns_summary():
    result = train_nlp_models(save_artifacts=False, n_samples=80)

    assert result.task == "nlp"
    assert result.best_model == "tfidf_logistic_regression"
    assert "tfidf_logistic_regression" in result.metrics
    assert result.metrics["tfidf_logistic_regression"]["Accuracy"] >= 0.5

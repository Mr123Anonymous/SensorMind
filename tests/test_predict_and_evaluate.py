"""Phase 4 tests for prediction and evaluation helpers."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.models.evaluate import evaluate_anomaly_model, evaluate_forecast_model
from src.models.predict import predict_anomaly, predict_forecast, predict_sentiment


def test_predict_forecast_with_sklearn_model() -> None:
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 3.0, 5.0, 7.0])
    model = LinearRegression().fit(X, y)

    pred = predict_forecast(model, np.array([[4.0], [5.0]]))
    assert pred.shape == (2,)
    assert np.allclose(pred, np.array([9.0, 11.0]), atol=1e-5)


def test_predict_anomaly_with_isolation_forest() -> None:
    rng = np.random.default_rng(42)
    normal = rng.normal(size=(100, 2))
    outlier = np.array([[7.0, 7.0]])
    X = np.vstack([normal, outlier])

    model = IsolationForest(contamination=0.05, random_state=42).fit(normal)
    result = predict_anomaly(model, X)

    assert set(result.keys()) == {"anomaly", "anomaly_score", "is_anomaly"}
    assert result["is_anomaly"].shape[0] == X.shape[0]
    assert result["is_anomaly"][-1] == 1


def test_predict_sentiment_with_pipeline() -> None:
    texts = ["great result", "bad outcome", "excellent model", "poor prediction"]
    labels = [1, 0, 1, 0]
    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    model.fit(texts, labels)

    output = predict_sentiment(model, ["great model", "poor model"])
    assert len(output["predictions"]) == 2
    assert output["probabilities"] is not None


def test_evaluate_helpers_return_expected_metrics() -> None:
    forecast_metrics = evaluate_forecast_model(
        np.array([1.0, 2.0, 3.0]),
        np.array([1.1, 1.9, 3.2]),
        model_name="test_forecast",
    )
    assert {"MSE", "RMSE", "MAE", "MAPE"}.issubset(forecast_metrics.keys())

    anomaly_metrics = evaluate_anomaly_model(
        np.array([0, 0, 1, 1]),
        np.array([0, 1, 1, 1]),
        model_name="test_anomaly",
    )
    assert {"Accuracy", "Precision", "Recall", "F1"}.issubset(anomaly_metrics.keys())

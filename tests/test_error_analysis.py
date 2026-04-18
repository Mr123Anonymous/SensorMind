"""Phase 4 tests for generated error-analysis report artifacts."""

from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

from src.analysis.error_analysis import generate_phase4_reports


def test_generate_phase4_reports_creates_outputs() -> None:
    data_processed_path = Path("tests/.tmp_phase4/data_processed")
    models_path = Path("tests/.tmp_phase4/models")
    reports_path = Path("tests/.tmp_phase4/reports")

    data_processed_path.mkdir(parents=True, exist_ok=True)
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    forecast_x_train = rng.normal(size=(24, 4))
    forecast_x_val = rng.normal(size=(6, 4))
    forecast_x_test = rng.normal(size=(6, 4))
    forecast_y_train = forecast_x_train[:, 0] * 0.5 - forecast_x_train[:, 1] * 0.2 + 1.0
    forecast_y_val = forecast_x_val[:, 0] * 0.5 - forecast_x_val[:, 1] * 0.2 + 1.0
    forecast_y_test = forecast_x_test[:, 0] * 0.5 - forecast_x_test[:, 1] * 0.2 + 1.0

    anomaly_x_train = rng.normal(size=(30, 3))
    anomaly_x_val = rng.normal(size=(10, 3))
    anomaly_x_test = rng.normal(size=(10, 3))
    anomaly_y_train = np.zeros(30, dtype=int)
    anomaly_y_val = np.zeros(10, dtype=int)
    anomaly_y_test = np.zeros(10, dtype=int)
    anomaly_y_test[-2:] = 1
    anomaly_x_test[-2:] += 4.0

    np.save(data_processed_path / "forecast_X_train.npy", forecast_x_train)
    np.save(data_processed_path / "forecast_X_val.npy", forecast_x_val)
    np.save(data_processed_path / "forecast_X_test.npy", forecast_x_test)
    np.save(data_processed_path / "forecast_y_train.npy", forecast_y_train)
    np.save(data_processed_path / "forecast_y_val.npy", forecast_y_val)
    np.save(data_processed_path / "forecast_y_test.npy", forecast_y_test)

    np.save(data_processed_path / "anomaly_X_train.npy", anomaly_x_train)
    np.save(data_processed_path / "anomaly_X_val.npy", anomaly_x_val)
    np.save(data_processed_path / "anomaly_X_test.npy", anomaly_x_test)
    np.save(data_processed_path / "anomaly_y_train.npy", anomaly_y_train)
    np.save(data_processed_path / "anomaly_y_val.npy", anomaly_y_val)
    np.save(data_processed_path / "anomaly_y_test.npy", anomaly_y_test)

    forecast_model = LinearRegression().fit(forecast_x_train, forecast_y_train)
    joblib.dump(forecast_model, models_path / "forecast_ridge.joblib")

    anomaly_model = IsolationForest(contamination=0.1, random_state=42).fit(anomaly_x_train)
    joblib.dump(anomaly_model, models_path / "anomaly_isolation_forest.joblib")

    summary = {
        "forecast": {
            "best_model": "ridge",
            "metrics": {"ridge": {"RMSE": 0.0, "MAE": 0.0, "MAPE": 0.0}},
        },
        "anomaly": {
            "best_model": "isolation_forest",
            "metrics": {"isolation_forest": {"Accuracy": 1.0, "Precision": 1.0, "Recall": 1.0, "F1": 1.0}},
        },
        "nlp": {
            "best_model": "tfidf_logistic_regression",
            "metrics": {"tfidf_logistic_regression": {"Accuracy": 1.0, "F1": 1.0}},
        },
    }
    with open(data_processed_path / "phase3_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    report = generate_phase4_reports(
        output_dir=reports_path,
        data_processed_path=data_processed_path,
        models_path=models_path,
    )

    assert report["phase"] == "phase4"
    assert "forecast" in report
    assert "anomaly" in report
    assert "nlp" in report

    assert (reports_path / "error_analysis.json").exists()
    assert (reports_path / "error_analysis.md").exists()

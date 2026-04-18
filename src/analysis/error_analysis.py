"""Error analysis utilities for Phase 4.

Builds report artifacts from trained model outputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import json

import numpy as np
import joblib
import torch
from sklearn.metrics import confusion_matrix

from src.config import DATA_PROCESSED_PATH, MODELS_PATH, REPORTS_PATH
from src.models.train import Autoencoder, ForecastMLP
from src.models.predict import predict_anomaly, predict_forecast
from src.utils import setup_logger

logger = setup_logger(__name__)


def _load_phase2_split(
    prefix: str,
    data_processed_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    arrays = []
    for name in names:
        path = data_processed_path / f"{prefix}_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing phase 2 artifact: {path}. Run scripts/run_phase2.py first."
            )

    for name in names:
        arrays.append(np.load(data_processed_path / f"{prefix}_{name}.npy", allow_pickle=False))
    return tuple(arrays)  # type: ignore[return-value]


def _load_forecast_model(model_name: str, models_path: Path) -> Any:
    if model_name == "mlp":
        payload = torch.load(models_path / "forecast_mlp.pt", map_location="cpu")
        model = ForecastMLP(input_dim=int(payload["input_dim"]), hidden_dim=int(payload.get("hidden_dim", 64)))
        model.load_state_dict(payload["state_dict"])
        return model
    return joblib.load(models_path / f"forecast_{model_name}.joblib")


def _load_anomaly_model(model_name: str, models_path: Path) -> Tuple[Any, float]:
    if model_name == "autoencoder":
        payload = torch.load(models_path / "anomaly_autoencoder.pt", map_location="cpu")
        model = Autoencoder(input_dim=int(payload["input_dim"]))
        model.load_state_dict(payload["state_dict"])
        threshold = float(payload.get("threshold", 0.5))
        return model, threshold

    model = joblib.load(models_path / "anomaly_isolation_forest.joblib")
    return model, 0.5


def _top_error_samples(y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 10) -> Dict[str, Any]:
    residuals = y_true - y_pred
    abs_residuals = np.abs(residuals)
    top_idx = np.argsort(abs_residuals)[-top_k:][::-1]
    return {
        "mae": float(np.mean(abs_residuals)),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "p90_abs_error": float(np.quantile(abs_residuals, 0.90)),
        "worst_indices": top_idx.tolist(),
        "worst_abs_errors": abs_residuals[top_idx].tolist(),
    }


def generate_phase4_reports(
    output_dir: Path | None = None,
    data_processed_path: Path | None = None,
    models_path: Path | None = None,
) -> Dict[str, Any]:
    """Generate JSON + Markdown error analysis reports.

    Returns a dictionary with detailed findings.
    """
    data_processed_path = data_processed_path or DATA_PROCESSED_PATH
    models_path = models_path or MODELS_PATH
    output_dir = output_dir or REPORTS_PATH
    output_dir.mkdir(parents=True, exist_ok=True)

    phase3_summary_path = data_processed_path / "phase3_summary.json"
    if not phase3_summary_path.exists():
        raise FileNotFoundError(
            f"Missing phase 3 summary: {phase3_summary_path}. Run scripts/run_phase3.py first."
        )

    with open(phase3_summary_path, "r", encoding="utf-8") as f:
        phase3_summary = json.load(f)

    _, _, fx_test, _, _, fy_test = _load_phase2_split("forecast", data_processed_path)
    _, _, ax_test, _, _, ay_test = _load_phase2_split("anomaly", data_processed_path)

    forecast_model_name = phase3_summary["forecast"]["best_model"]
    forecast_model = _load_forecast_model(forecast_model_name, models_path)
    forecast_pred = predict_forecast(forecast_model, fx_test)
    forecast_analysis = _top_error_samples(fy_test, forecast_pred)

    anomaly_model_name = phase3_summary["anomaly"]["best_model"]
    anomaly_model, threshold = _load_anomaly_model(anomaly_model_name, models_path)
    anomaly_output = predict_anomaly(anomaly_model, ax_test, threshold=threshold)
    anomaly_pred = anomaly_output["is_anomaly"].astype(int)
    tn, fp, fn, tp = confusion_matrix(ay_test, anomaly_pred, labels=[0, 1]).ravel()

    anomaly_analysis = {
        "threshold": float(threshold),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "fp_indices": np.where((ay_test == 0) & (anomaly_pred == 1))[0].tolist()[:20],
        "fn_indices": np.where((ay_test == 1) & (anomaly_pred == 0))[0].tolist()[:20],
    }

    nlp_metrics = phase3_summary["nlp"]["metrics"][phase3_summary["nlp"]["best_model"]]

    report = {
        "phase": "phase4",
        "forecast": {
            "best_model": forecast_model_name,
            "analysis": forecast_analysis,
        },
        "anomaly": {
            "best_model": anomaly_model_name,
            "analysis": anomaly_analysis,
        },
        "nlp": {
            "best_model": phase3_summary["nlp"]["best_model"],
            "metrics": nlp_metrics,
            "analysis": {
                "note": "Synthetic text baseline reached high scores; validate on real corpus before production.",
                "risk": "Potential overfitting to templated synthetic phrases.",
            },
        },
    }

    json_path = output_dir / "error_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# Phase 4 Error Analysis",
        "",
        "## Forecasting",
        f"- Best model: {forecast_model_name}",
        f"- MAE: {forecast_analysis['mae']:.4f}",
        f"- P90 absolute error: {forecast_analysis['p90_abs_error']:.4f}",
        f"- Residual mean: {forecast_analysis['residual_mean']:.4f}",
        f"- Residual std: {forecast_analysis['residual_std']:.4f}",
        f"- Worst sample indices: {forecast_analysis['worst_indices'][:10]}",
        "",
        "## Anomaly Detection",
        f"- Best model: {anomaly_model_name}",
        f"- Threshold: {anomaly_analysis['threshold']:.6f}",
        f"- TP: {anomaly_analysis['true_positives']}, FP: {anomaly_analysis['false_positives']}, FN: {anomaly_analysis['false_negatives']}, TN: {anomaly_analysis['true_negatives']}",
        f"- False-positive examples: {anomaly_analysis['fp_indices'][:10]}",
        f"- False-negative examples: {anomaly_analysis['fn_indices'][:10]}",
        "",
        "## NLP",
        f"- Best model: {phase3_summary['nlp']['best_model']}",
        f"- Accuracy: {nlp_metrics.get('Accuracy', 0):.4f}",
        f"- F1: {nlp_metrics.get('F1', 0):.4f}",
        f"- Risk note: {report['nlp']['analysis']['risk']}",
    ]

    md_path = output_dir / "error_analysis.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    logger.info("Phase 4 error-analysis reports written to %s and %s", json_path, md_path)
    return report

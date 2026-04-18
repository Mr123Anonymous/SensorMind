"""Model training module.

This module implements the first working model baselines for Phase 3:

* Forecasting: Ridge, Random Forest, and a small PyTorch MLP regressor
* Anomaly detection: Isolation Forest and a PyTorch autoencoder
* NLP: TF-IDF + Logistic Regression on a synthetic sentiment corpus

The models are intentionally lightweight so they can run locally and in CI,
while still demonstrating the end-to-end ML workflow for a resume project.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import json

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import DATA_PROCESSED_PATH, MLFLOW_CONFIG, MODEL_CONFIG, MODELS_PATH, RANDOM_SEED
from src.models.evaluate import evaluate_anomaly_model, evaluate_forecast_model
from src.utils import get_classification_metrics, save_metrics, setup_logger

try:
    import mlflow
except Exception:  # pragma: no cover - optional at runtime
    mlflow = None

logger = setup_logger(__name__)


torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.set_num_threads(1)


@dataclass
class TrainingResult:
    """Metadata describing a trained model family."""

    task: str
    best_model: str
    metrics: Dict[str, Dict[str, Any]]
    artifacts: Dict[str, str]


class ForecastMLP(nn.Module):
    """Small feed-forward regressor for tabular time-series features."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class Autoencoder(nn.Module):
    """Simple autoencoder for anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = 8) -> None:
        super().__init__()
        hidden_dim = max(16, input_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(inputs)
        return self.decoder(encoded)


def _ensure_models_dir() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)


def _log_mlflow_run(
    task: str,
    model_name: str,
    metrics: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    artifact_path: Optional[Path] = None,
) -> None:
    """Log one model run to MLflow if the dependency is available."""
    if mlflow is None:
        return

    try:
        tracking_dir = DATA_PROCESSED_PATH.parent / "mlruns"
        tracking_dir.mkdir(parents=True, exist_ok=True)

        mlflow.set_tracking_uri(MLFLOW_CONFIG.get("tracking_uri", f"file://{tracking_dir.resolve()}"))
        mlflow.set_experiment(MLFLOW_CONFIG.get("experiment_name", "SensorMind-ml-portfolio"))
        with mlflow.start_run(run_name=f"{task}_{model_name}"):
            mlflow.log_param("task", task)
            mlflow.log_param("model_name", model_name)
            if params:
                for key, value in params.items():
                    mlflow.log_param(key, value)

            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, float(value))

            if artifact_path is not None and artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))
    except Exception as exc:  # pragma: no cover - non-fatal telemetry path
        logger.warning("MLflow logging skipped for %s/%s: %s", task, model_name, exc)


def _safe_load_numpy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _load_phase2_split(prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load arrays from the Phase 2 preprocessing pipeline, running it if needed."""
    expected = {
        "X_train": DATA_PROCESSED_PATH / f"{prefix}_X_train.npy",
        "X_val": DATA_PROCESSED_PATH / f"{prefix}_X_val.npy",
        "X_test": DATA_PROCESSED_PATH / f"{prefix}_X_test.npy",
        "y_train": DATA_PROCESSED_PATH / f"{prefix}_y_train.npy",
        "y_val": DATA_PROCESSED_PATH / f"{prefix}_y_val.npy",
        "y_test": DATA_PROCESSED_PATH / f"{prefix}_y_test.npy",
    }

    if not all(path.exists() for path in expected.values()):
        logger.info("Phase 2 artifacts not found for %s; regenerating them.", prefix)
        from scripts.run_phase2 import main as run_phase2_main

        run_phase2_main()

    arrays = tuple(_safe_load_numpy(expected[name]) for name in ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"])
    return arrays  # type: ignore[return-value]


def _save_metrics_report(task: str, metrics: Dict[str, Dict[str, Any]], best_model: str) -> Path:
    report_path = DATA_PROCESSED_PATH / f"phase3_{task}_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({"task": task, "best_model": best_model, "metrics": metrics}, handle, indent=2)
    return report_path


def _train_torch_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> ForecastMLP:
    model = ForecastMLP(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_predictions = model(X_val_tensor)
            val_loss = loss_fn(val_predictions, y_val_tensor).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

        logger.info("Forecast MLP epoch %d/%d - val_loss=%.5f", epoch + 1, epochs, val_loss)

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def _predict_torch_regressor(model: ForecastMLP, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        predictions = model(torch.tensor(X, dtype=torch.float32)).cpu().numpy()
    return predictions


def _train_autoencoder(
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> Tuple[Autoencoder, float]:
    model = Autoencoder(input_dim=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for (batch_x,) in train_loader:
            optimizer.zero_grad()
            reconstruction = model(batch_x)
            loss = loss_fn(reconstruction, batch_x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_x)

        logger.info("Autoencoder epoch %d/%d - train_loss=%.5f", epoch + 1, epochs, epoch_loss / len(train_dataset))

    train_scores = _reconstruction_errors(model, X_train)
    candidate_thresholds = np.quantile(train_scores, [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99])

    if len(np.unique(y_val)) > 1:
        val_scores = _reconstruction_errors(model, X_val)
        best_threshold = candidate_thresholds[0]
        best_f1 = -1.0
        for threshold in candidate_thresholds:
            predictions = (val_scores > threshold).astype(int)
            f1 = f1_score(y_val, predictions, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
    else:
        best_threshold = float(candidate_thresholds[4])

    return model, best_threshold


def _reconstruction_errors(model: Autoencoder, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        reconstructions = model(inputs)
        errors = torch.mean((reconstructions - inputs) ** 2, dim=1).cpu().numpy()
    return errors


def _train_text_classifier(
    n_samples: int = 400,
) -> Tuple[Pipeline, pd.DataFrame]:
    positive_templates = [
        "the model is performing well",
        "deployment went smoothly",
        "the dashboard looks clean and useful",
        "the forecast is accurate and stable",
        "the pipeline runs reliably",
        "we improved model performance",
        "the system response is fast",
        "great results from the new baseline",
    ]
    negative_templates = [
        "the model is performing poorly",
        "deployment failed with an error",
        "the dashboard is confusing and slow",
        "the forecast is inaccurate and unstable",
        "the pipeline breaks frequently",
        "we need to fix model drift",
        "the system response is too slow",
        "results from the baseline are weak",
    ]

    rng = np.random.default_rng(RANDOM_SEED)
    texts: list[str] = []
    labels: list[int] = []
    for _ in range(n_samples // 2):
        texts.append(rng.choice(positive_templates))
        labels.append(1)
        texts.append(rng.choice(negative_templates))
        labels.append(0)

    dataset = pd.DataFrame({"text": texts, "label": labels})
    dataset = dataset.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    train_texts, test_texts, train_labels, test_labels = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=dataset["label"],
    )

    classifier = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ("clf", LogisticRegression(max_iter=500, random_state=RANDOM_SEED)),
        ]
    )
    classifier.fit(train_texts, train_labels)

    predictions = classifier.predict(test_texts)
    metrics = pd.DataFrame(
        {
            "text": test_texts,
            "label": test_labels,
            "prediction": predictions,
        }
    )
    return classifier, metrics


def train_forecast_models(
    X_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    save_artifacts: bool = True,
    mlp_epochs: Optional[int] = None,
) -> TrainingResult:
    """Train time-series forecasting models and return the best model summary."""
    logger.info("Training forecasting models...")

    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = _load_phase2_split("forecast")
    assert X_val is not None and X_test is not None and y_train is not None and y_val is not None and y_test is not None

    forecast_config = MODEL_CONFIG["forecast"]
    epochs = mlp_epochs or min(25, forecast_config["lstm_config"]["epochs"])

    trained_models: Dict[str, Any] = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=100,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
    }

    trained_models["ridge"].fit(X_train, y_train)
    trained_models["random_forest"].fit(X_train, y_train)
    trained_models["mlp"] = _train_torch_regressor(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        learning_rate=forecast_config["lstm_config"]["learning_rate"],
        batch_size=min(32, forecast_config["lstm_config"]["batch_size"]),
    )

    metrics: Dict[str, Dict[str, Any]] = {}
    artifact_paths: Dict[str, str] = {}
    for name, model in trained_models.items():
        if name == "mlp":
            predictions = _predict_torch_regressor(model, X_test)
        else:
            predictions = model.predict(X_test)

        metrics[name] = evaluate_forecast_model(y_test, predictions, model_name=name)

        if save_artifacts:
            artifact_path = MODELS_PATH / f"forecast_{name}.joblib"
            if name == "mlp":
                artifact_path = MODELS_PATH / "forecast_mlp.pt"
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "input_dim": X_train.shape[1],
                        "hidden_dim": 64,
                    },
                    artifact_path,
                )
            else:
                joblib.dump(model, artifact_path)
            artifact_paths[name] = str(artifact_path)

            _log_mlflow_run(
                task="forecast",
                model_name=name,
                metrics=metrics[name],
                params={"epochs": epochs if name == "mlp" else 0},
                artifact_path=artifact_path,
            )

    best_model = min(metrics.items(), key=lambda item: item[1]["RMSE"])[0]

    if save_artifacts:
        save_metrics(metrics[best_model], MODELS_PATH / "forecast_best_metrics.json", model_name=best_model)
        _save_metrics_report("forecast", metrics, best_model)

    logger.info("Best forecasting model: %s", best_model)
    return TrainingResult(task="forecast", best_model=best_model, metrics=metrics, artifacts=artifact_paths)


def train_anomaly_models(
    X_train: Optional[np.ndarray] = None,
    X_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    save_artifacts: bool = True,
    ae_epochs: Optional[int] = None,
) -> TrainingResult:
    """Train anomaly detection models and return the best model summary."""
    logger.info("Training anomaly detection models...")

    if X_train is None:
        X_train, X_val, X_test, y_train, y_val, y_test = _load_phase2_split("anomaly")
    assert X_val is not None and X_test is not None and y_train is not None and y_val is not None and y_test is not None

    anomaly_config = MODEL_CONFIG["anomaly"]
    isolation_forest = IsolationForest(
        n_estimators=anomaly_config["isolation_forest_params"]["n_estimators"],
        contamination=anomaly_config["isolation_forest_params"]["contamination"],
        random_state=RANDOM_SEED,
    )
    isolation_forest.fit(X_train)
    iso_predictions = (isolation_forest.predict(X_test) == -1).astype(int)
    iso_metrics = evaluate_anomaly_model(y_test, iso_predictions, model_name="isolation_forest")

    ae_epochs = ae_epochs or min(25, anomaly_config["autoencoder_config"]["epochs"])
    autoencoder, threshold = _train_autoencoder(
        X_train=X_train[y_train == 0] if np.any(y_train == 0) else X_train,
        X_val=X_val,
        y_val=y_val,
        epochs=ae_epochs,
        learning_rate=anomaly_config["autoencoder_config"]["learning_rate"],
        batch_size=min(32, anomaly_config["autoencoder_config"]["batch_size"]),
    )
    ae_scores = _reconstruction_errors(autoencoder, X_test)
    ae_predictions = (ae_scores > threshold).astype(int)
    ae_metrics = evaluate_anomaly_model(y_test, ae_predictions, model_name="autoencoder")

    metrics = {
        "isolation_forest": iso_metrics,
        "autoencoder": ae_metrics,
    }
    best_model = max(metrics.items(), key=lambda item: item[1]["F1"])[0]
    artifact_paths: Dict[str, str] = {}

    if save_artifacts:
        iso_path = MODELS_PATH / "anomaly_isolation_forest.joblib"
        joblib.dump(isolation_forest, iso_path)
        artifact_paths["isolation_forest"] = str(iso_path)
        _log_mlflow_run(
            task="anomaly",
            model_name="isolation_forest",
            metrics=iso_metrics,
            params={
                "contamination": anomaly_config["isolation_forest_params"]["contamination"],
                "n_estimators": anomaly_config["isolation_forest_params"]["n_estimators"],
            },
            artifact_path=iso_path,
        )

        ae_path = MODELS_PATH / "anomaly_autoencoder.pt"
        torch.save(
            {
                "state_dict": autoencoder.state_dict(),
                "input_dim": X_train.shape[1],
                "threshold": threshold,
            },
            ae_path,
        )
        artifact_paths["autoencoder"] = str(ae_path)
        _log_mlflow_run(
            task="anomaly",
            model_name="autoencoder",
            metrics=ae_metrics,
            params={"epochs": ae_epochs, "threshold": threshold},
            artifact_path=ae_path,
        )

        save_metrics(metrics[best_model], MODELS_PATH / "anomaly_best_metrics.json", model_name=best_model)
        _save_metrics_report("anomaly", metrics, best_model)

    logger.info("Best anomaly model: %s", best_model)
    return TrainingResult(task="anomaly", best_model=best_model, metrics=metrics, artifacts=artifact_paths)


def train_nlp_models(
    save_artifacts: bool = True,
    n_samples: int = 400,
) -> TrainingResult:
    """Train NLP sentiment baselines and return the best model summary."""
    logger.info("Training NLP models...")

    classifier, evaluation_frame = _train_text_classifier(n_samples=n_samples)
    y_true = evaluation_frame["label"].to_numpy()
    y_pred = evaluation_frame["prediction"].to_numpy()
    metrics = get_classification_metrics(y_true, y_pred)

    best_model = "tfidf_logistic_regression"
    artifact_paths: Dict[str, str] = {}
    if save_artifacts:
        artifact_path = MODELS_PATH / "nlp_tfidf_logreg.joblib"
        joblib.dump(classifier, artifact_path)
        artifact_paths[best_model] = str(artifact_path)
        save_metrics(metrics, MODELS_PATH / "nlp_best_metrics.json", model_name=best_model)
        _save_metrics_report("nlp", {best_model: metrics}, best_model)
        _log_mlflow_run(
            task="nlp",
            model_name=best_model,
            metrics=metrics,
            params={"n_samples": n_samples},
            artifact_path=artifact_path,
        )

    logger.info("Best NLP model: %s", best_model)
    return TrainingResult(task="nlp", best_model=best_model, metrics={best_model: metrics}, artifacts=artifact_paths)


def train_all_models(save_artifacts: bool = True) -> Dict[str, TrainingResult]:
    """Train the full Phase 3 baseline suite."""
    _ensure_models_dir()
    return {
        "forecast": train_forecast_models(save_artifacts=save_artifacts),
        "anomaly": train_anomaly_models(save_artifacts=save_artifacts),
        "nlp": train_nlp_models(save_artifacts=save_artifacts),
    }


def main() -> None:
    """Train all Phase 3 models and persist artifacts."""
    _ensure_models_dir()

    logger.info("=" * 60)
    logger.info("SensorMind ML Portfolio - Phase 3 Model Training")
    logger.info("=" * 60)
    logger.info("")

    results = train_all_models(save_artifacts=True)

    summary = {
        task: {
            "best_model": result.best_model,
            "artifacts": result.artifacts,
            "metrics": result.metrics,
        }
        for task, result in results.items()
    }
    summary_path = DATA_PROCESSED_PATH / "phase3_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("")
    logger.info("[OK] Phase 3 training complete!")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()



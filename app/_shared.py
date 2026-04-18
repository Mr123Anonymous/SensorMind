"""Shared utilities for Streamlit pages."""

from __future__ import annotations

from pathlib import Path
import warnings
from typing import Any, Dict, Tuple
import json

import joblib
import numpy as np
import torch
import streamlit as st

warnings.filterwarnings(
    "ignore",
    message=r'Field "model_server_url" has conflict with protected namespace "model_".*',
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Valid config keys have changed in V2:.*schema_extra.*json_schema_extra.*",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED_PATH = ROOT / "data" / "processed"
MODELS_PATH = ROOT / "models"
REPORTS_PATH = ROOT / "reports"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_phase3_summary() -> Dict[str, Any]:
    return _load_json(DATA_PROCESSED_PATH / "phase3_summary.json")


@st.cache_data(show_spinner=False)
def load_error_analysis() -> Dict[str, Any]:
    return _load_json(REPORTS_PATH / "error_analysis.json")


@st.cache_data(show_spinner=False)
def load_split(prefix: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ["X_train", "X_val", "X_test", "y_train", "y_val", "y_test"]
    arrays = []
    for name in names:
        path = DATA_PROCESSED_PATH / f"{prefix}_{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing data artifact: {path}")
        arrays.append(np.load(path, allow_pickle=False))
    return tuple(arrays)  # type: ignore[return-value]


@st.cache_resource(show_spinner=False)
def load_forecast_model(model_name: str) -> Any:
    if model_name == "mlp":
        from src.models.train import ForecastMLP

        payload = torch.load(MODELS_PATH / "forecast_mlp.pt", map_location="cpu")
        model = ForecastMLP(input_dim=int(payload["input_dim"]), hidden_dim=int(payload.get("hidden_dim", 64)))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model

    return joblib.load(MODELS_PATH / f"forecast_{model_name}.joblib")


@st.cache_resource(show_spinner=False)
def load_anomaly_model(model_name: str) -> Tuple[Any, float]:
    if model_name == "autoencoder":
        from src.models.train import Autoencoder

        payload = torch.load(MODELS_PATH / "anomaly_autoencoder.pt", map_location="cpu")
        model = Autoencoder(input_dim=int(payload["input_dim"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        threshold = float(payload.get("threshold", 0.5))
        return model, threshold

    model = joblib.load(MODELS_PATH / "anomaly_isolation_forest.joblib")
    return model, 0.5


@st.cache_resource(show_spinner=False)
def load_nlp_model() -> Any:
    return joblib.load(MODELS_PATH / "nlp_tfidf_logreg.joblib")


def has_phase5_artifacts() -> bool:
    required = [
        DATA_PROCESSED_PATH / "phase3_summary.json",
        MODELS_PATH / "forecast_ridge.joblib",
        MODELS_PATH / "anomaly_autoencoder.pt",
        MODELS_PATH / "nlp_tfidf_logreg.joblib",
    ]
    return all(path.exists() for path in required)

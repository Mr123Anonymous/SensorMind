"""Run Phase 2 data preparation pipeline.

This script:
1. Loads raw data (real if available, synthetic fallback).
2. Runs preprocessing for forecasting and anomaly tracks.
3. Persists processed datasets and NumPy artifacts for model training.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

# Ensure project root is importable when executed as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_PROCESSED_PATH, DATA_RAW_PATH
from src.data.loaders import load_pgcb_dataset, load_synthetic_time_series
from src.data.preprocessing import preprocess_forecast_data, preprocess_anomaly_data
from src.utils import setup_logger

logger = setup_logger(__name__)


def _ensure_datetime_index(df: pd.DataFrame, timestamp_col: str | None = None) -> pd.DataFrame:
    """Ensure dataframe uses a DatetimeIndex for time-series feature generation."""
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        return out

    if timestamp_col and timestamp_col in out.columns:
        out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
        out = out.set_index(timestamp_col)
    else:
        # Fallback: infer by position if first column looks like timestamps.
        first_col = out.columns[0]
        maybe_ts = pd.to_datetime(out[first_col], errors="coerce")
        if maybe_ts.notna().mean() > 0.95:
            out[first_col] = maybe_ts
            out = out.set_index(first_col)
        else:
            out.index = pd.date_range(start="2024-01-01", periods=len(out), freq="h")

    return out.sort_index()


def _save_numpy_artifacts(prefix: str, arrays: dict[str, np.ndarray]) -> None:
    """Persist NumPy arrays to processed directory."""
    for name, arr in arrays.items():
        np.save(DATA_PROCESSED_PATH / f"{prefix}_{name}.npy", arr)


def run_forecasting_pipeline() -> dict[str, int]:
    """Run preprocessing for forecasting task and persist artifacts."""
    df = load_pgcb_dataset(download=False)

    if df is None:
        logger.info("PGCB raw data not found. Using synthetic forecasting data.")
        df = load_synthetic_time_series(n_samples=24 * 365, n_features=5, seed=42)
        df.columns = ["generation", "demand", "frequency", "voltage", "temperature"]
        df["generation"] = df["generation"].abs() + 80

    df = _ensure_datetime_index(df)

    if "generation" not in df.columns:
        # Use first numeric column as target if schema differs.
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found for forecasting target.")
        df = df.rename(columns={numeric_cols[0]: "generation"})

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_forecast_data(
        df=df,
        target_column="generation",
        test_size=0.1,
        val_size=0.1,
        scale=True,
    )

    _save_numpy_artifacts(
        "forecast",
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
    )

    # Keep a processed tabular sample for inspection.
    pd.DataFrame(X_train[:500]).to_csv(DATA_PROCESSED_PATH / "forecast_X_train_sample.csv", index=False)

    return {
        "forecast_train": int(len(y_train)),
        "forecast_val": int(len(y_val)),
        "forecast_test": int(len(y_test)),
    }


def run_anomaly_pipeline() -> dict[str, int]:
    """Run preprocessing for anomaly detection and persist artifacts."""
    anomaly_csv = DATA_RAW_PATH / "anomaly_data.csv"
    labels_csv = DATA_RAW_PATH / "anomaly_labels.csv"

    if anomaly_csv.exists():
        anomaly_df = pd.read_csv(anomaly_csv)
        labels_df = pd.read_csv(labels_csv) if labels_csv.exists() else None
        anomaly_df = _ensure_datetime_index(anomaly_df)
    else:
        logger.info("Anomaly raw data not found. Using synthetic anomaly data.")
        base = load_synthetic_time_series(n_samples=5000, n_features=3, seed=7)
        base.columns = ["sensor_1", "sensor_2", "sensor_3"]

        labels = np.zeros(len(base), dtype=int)
        spike_idx = np.random.default_rng(7).choice(len(base), size=250, replace=False)
        base.iloc[spike_idx, 0] += np.random.default_rng(8).normal(8, 2, size=len(spike_idx))
        labels[spike_idx] = 1

        anomaly_df = base
        labels_df = pd.DataFrame({"is_anomaly": labels})

    X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_anomaly_data(
        df=anomaly_df,
        labels_df=labels_df,
        test_size=0.15,
        val_size=0.15,
        scale=True,
    )

    _save_numpy_artifacts(
        "anomaly",
        {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
    )

    return {
        "anomaly_train": int(len(y_train)),
        "anomaly_val": int(len(y_val)),
        "anomaly_test": int(len(y_test)),
    }


def main() -> None:
    """Execute full Phase 2 pipeline and write summary."""
    DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 2 pipeline...")
    forecast_counts = run_forecasting_pipeline()
    anomaly_counts = run_anomaly_pipeline()

    summary = {
        "status": "phase_2_initiated",
        **forecast_counts,
        **anomaly_counts,
    }

    summary_path = Path(DATA_PROCESSED_PATH) / "phase2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase 2 pipeline completed.")
    logger.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()

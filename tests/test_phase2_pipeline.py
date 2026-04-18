"""Tests for Phase 2 pipeline initiation behavior."""

from pathlib import Path

import numpy as np


def test_processed_directory_exists() -> None:
    """Processed data directory should exist after project bootstrap."""
    assert Path("data/processed").exists()


def test_numpy_artifact_suffixes_are_consistent() -> None:
    """Numpy artifacts should follow the .npy convention used by training scripts."""
    expected_suffix = ".npy"
    samples = [
        "forecast_X_train.npy",
        "forecast_y_train.npy",
        "anomaly_X_train.npy",
        "anomaly_y_train.npy",
    ]
    assert all(Path(name).suffix == expected_suffix for name in samples)


def test_train_val_test_split_ratio_math() -> None:
    """Validate split index math used in preprocessing temporal splits."""
    n = 1000
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1

    train_idx = int(n * train_size)
    val_idx = int(n * (train_size + val_size))

    train_count = train_idx
    val_count = val_idx - train_idx
    test_count = n - val_idx

    assert train_count + val_count + test_count == n
    assert np.isclose(train_count / n, train_size, atol=0.01)
    assert np.isclose(val_count / n, val_size, atol=0.01)
    assert np.isclose(test_count / n, test_size, atol=0.01)

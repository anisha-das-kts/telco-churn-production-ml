"""Tests for model runtime benchmarking."""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier

from src.training.benchmark import (
    benchmark_model,
)


def create_fitted_model() -> DummyClassifier:
    """Create a small fitted classifier."""

    model = DummyClassifier(
        strategy="prior",
    )

    model.fit(
        [[0], [1], [0], [1]],
        [0, 1, 0, 1],
    )

    return model


def test_benchmark_returns_runtime_metrics() -> None:
    """Benchmark should return usable promotion metrics."""

    model = create_fitted_model()

    features = pd.DataFrame(
        {
            "feature": [
                0,
                1,
                0,
                1,
            ]
        }
    )

    result = benchmark_model(
        model=model,
        features=features,
        measured_requests=20,
        warmup_requests=2,
    )

    assert result["measured_requests"] == 20
    assert result["successful_requests"] == 20
    assert result["failed_requests"] == 0
    assert result["error_rate_percent"] == 0.0
    assert result["p95_latency_ms"] >= 0.0
    assert (
        result["throughput_requests_per_second"]
        > 0.0
    )


def test_empty_features_are_rejected() -> None:
    """An empty benchmark dataset should be rejected."""

    model = create_fitted_model()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        benchmark_model(
            model=model,
            features=pd.DataFrame(),
        )


def test_invalid_request_count_is_rejected() -> None:
    """Benchmark request count must be positive."""

    model = create_fitted_model()

    features = pd.DataFrame(
        {"feature": [0, 1]}
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        benchmark_model(
            model=model,
            features=features,
            measured_requests=0,
        )


def test_partial_failures_are_measured() -> None:
    """Failed inference calls should affect error rate."""

    class FlakyModel:
        def __init__(self) -> None:
            self.call_count = 0

        def predict_proba(self, features):
            self.call_count += 1

            if self.call_count % 2 == 0:
                raise RuntimeError(
                    "Simulated inference failure"
                )

            return np.array(
                [[0.4, 0.6]]
            )

    result = benchmark_model(
        model=FlakyModel(),
        features=pd.DataFrame(
            {"feature": [0, 1]}
        ),
        measured_requests=4,
        warmup_requests=0,
    )

    assert result["successful_requests"] == 2
    assert result["failed_requests"] == 2
    assert result["error_rate_percent"] == 50.0
    
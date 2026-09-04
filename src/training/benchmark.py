"""Runtime benchmark utilities for model-promotion checks."""

from time import perf_counter

import numpy as np
import pandas as pd


def benchmark_model(
    model,
    features: pd.DataFrame,
    measured_requests: int = 100,
    warmup_requests: int = 5,
) -> dict:
    """Benchmark single-record model inference."""

    if features.empty:
        raise ValueError(
            "Benchmark features cannot be empty"
        )

    if measured_requests <= 0:
        raise ValueError(
            "measured_requests must be greater than zero"
        )

    if warmup_requests < 0:
        raise ValueError(
            "warmup_requests cannot be negative"
        )

    row_count = len(features)

    for request_number in range(
        warmup_requests
    ):
        row_index = request_number % row_count
        model_input = features.iloc[
            [row_index]
        ]

        model.predict_proba(model_input)

    latencies_ms = []
    failed_requests = 0

    benchmark_start = perf_counter()

    for request_number in range(
        measured_requests
    ):
        row_index = request_number % row_count
        model_input = features.iloc[
            [row_index]
        ]

        request_start = perf_counter()

        try:
            model.predict_proba(model_input)
        except Exception:
            failed_requests += 1
        else:
            latency_ms = (
                perf_counter() - request_start
            ) * 1000

            latencies_ms.append(
                float(latency_ms)
            )

    total_duration_seconds = (
        perf_counter() - benchmark_start
    )

    successful_requests = len(latencies_ms)

    if successful_requests == 0:
        raise RuntimeError(
            "Every measured inference request failed"
        )

    error_rate_percent = (
        failed_requests
        / measured_requests
        * 100
    )

    return {
        "warmup_requests": int(
            warmup_requests
        ),
        "measured_requests": int(
            measured_requests
        ),
        "successful_requests": int(
            successful_requests
        ),
        "failed_requests": int(
            failed_requests
        ),
        "error_rate_percent": float(
            error_rate_percent
        ),
        "total_duration_seconds": float(
            total_duration_seconds
        ),
        "average_latency_ms": float(
            np.mean(latencies_ms)
        ),
        "median_latency_ms": float(
            np.median(latencies_ms)
        ),
        "p95_latency_ms": float(
            np.percentile(
                latencies_ms,
                95,
            )
        ),
        "minimum_latency_ms": float(
            np.min(latencies_ms)
        ),
        "maximum_latency_ms": float(
            np.max(latencies_ms)
        ),
        "throughput_requests_per_second": float(
            measured_requests
            / total_duration_seconds
        ),
    }
    
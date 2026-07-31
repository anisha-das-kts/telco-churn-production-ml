"""Measure latency, throughput and error rate of the prediction API."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "eval"
    / "api_benchmark.json"
)

DEFAULT_API_URL = "http://127.0.0.1:8000/predict"

LATENCY_TARGET_MS = 200.0


PREDICTION_PAYLOAD = {
    "gender": "Female",
    "senior_citizen": 0,
    "partner": "Yes",
    "dependents": "No",
    "tenure": 5,
    "phone_service": "Yes",
    "multiple_lines": "No",
    "internet_service": "Fiber optic",
    "online_security": "No",
    "online_backup": "No",
    "device_protection": "No",
    "tech_support": "No",
    "streaming_tv": "Yes",
    "streaming_movies": "Yes",
    "contract": "Month-to-month",
    "paperless_billing": "Yes",
    "payment_method": "Electronic check",
    "monthly_charges": 89.50,
    "total_charges": 447.50,
}


def check_api_health(session: requests.Session) -> None:
    """Confirm that the API is available before benchmarking."""

    health_url = "http://127.0.0.1:8000/health"

    try:
        response = session.get(
            health_url,
            timeout=5,
        )
        response.raise_for_status()

    except requests.RequestException as error:
        raise RuntimeError(
            "The FastAPI service is unavailable. "
            "Start it with: "
            "uvicorn src.serving.app:app --reload"
        ) from error


def send_request(
    session: requests.Session,
    api_url: str,
) -> tuple[float, bool, dict | None]:
    """Send one request and return latency and success status."""

    start_time = perf_counter()

    try:
        response = session.post(
            api_url,
            json=PREDICTION_PAYLOAD,
            timeout=10,
        )

        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        successful = response.status_code == 200

        response_body = (
            response.json()
            if successful
            else None
        )

        return (
            float(latency_ms),
            successful,
            response_body,
        )

    except requests.RequestException:
        latency_ms = (
            perf_counter() - start_time
        ) * 1000

        return float(latency_ms), False, None


def run_benchmark(
    request_count: int,
    warmup_count: int,
    api_url: str,
) -> dict:
    """Execute warm-up and measured API requests."""

    session = requests.Session()

    check_api_health(session)

    print(
        f"Warming up the API with "
        f"{warmup_count} request(s)..."
    )

    for _ in range(warmup_count):
        send_request(
            session=session,
            api_url=api_url,
        )

    print(
        f"Running benchmark with "
        f"{request_count} measured request(s)..."
    )

    latencies = []
    successful_requests = 0
    failed_requests = 0
    sample_response = None

    benchmark_start = perf_counter()

    for _ in range(request_count):
        latency_ms, successful, response_body = (
            send_request(
                session=session,
                api_url=api_url,
            )
        )

        latencies.append(latency_ms)

        if successful:
            successful_requests += 1
            sample_response = response_body
        else:
            failed_requests += 1

    total_duration_seconds = (
        perf_counter() - benchmark_start
    )

    latency_array = np.asarray(
        latencies,
        dtype=float,
    )

    average_latency_ms = float(
        np.mean(latency_array)
    )
    median_latency_ms = float(
        np.median(latency_array)
    )
    p95_latency_ms = float(
        np.percentile(latency_array, 95)
    )
    minimum_latency_ms = float(
        np.min(latency_array)
    )
    maximum_latency_ms = float(
        np.max(latency_array)
    )

    throughput_requests_per_second = float(
        request_count
        / max(total_duration_seconds, 0.000001)
    )

    error_rate_percent = float(
        failed_requests
        / max(request_count, 1)
        * 100
    )

    report = {
        "benchmark_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "api_url": api_url,
        "warmup_requests": warmup_count,
        "measured_requests": request_count,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "error_rate_percent": error_rate_percent,
        "total_duration_seconds": float(
            total_duration_seconds
        ),
        "average_latency_ms": average_latency_ms,
        "median_latency_ms": median_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "minimum_latency_ms": minimum_latency_ms,
        "maximum_latency_ms": maximum_latency_ms,
        "throughput_requests_per_second": (
            throughput_requests_per_second
        ),
        "latency_target_ms": LATENCY_TARGET_MS,
        "meets_p95_latency_target": bool(
            p95_latency_ms <= LATENCY_TARGET_MS
        ),
        "sample_response": sample_response,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as report_file:
        json.dump(
            report,
            report_file,
            indent=4,
        )

    return report


def print_report(report: dict) -> None:
    """Print the important benchmark results."""

    print()
    print("API PERFORMANCE BENCHMARK")
    print("=" * 45)
    print(
        f"Measured requests: "
        f"{report['measured_requests']}"
    )
    print(
        f"Successful requests: "
        f"{report['successful_requests']}"
    )
    print(
        f"Failed requests: "
        f"{report['failed_requests']}"
    )
    print(
        f"Error rate: "
        f"{report['error_rate_percent']:.2f}%"
    )
    print(
        f"Average latency: "
        f"{report['average_latency_ms']:.2f} ms"
    )
    print(
        f"Median latency: "
        f"{report['median_latency_ms']:.2f} ms"
    )
    print(
        f"P95 latency: "
        f"{report['p95_latency_ms']:.2f} ms"
    )
    print(
        f"Minimum latency: "
        f"{report['minimum_latency_ms']:.2f} ms"
    )
    print(
        f"Maximum latency: "
        f"{report['maximum_latency_ms']:.2f} ms"
    )
    print(
        f"Throughput: "
        f"{report['throughput_requests_per_second']:.2f} "
        f"requests/second"
    )
    print(
        f"P95 target: "
        f"{report['latency_target_ms']:.2f} ms"
    )
    print(
        f"Meets latency target: "
        f"{report['meets_p95_latency_target']}"
    )
    print("=" * 45)
    print(f"Report saved to: {REPORT_PATH}")


def main() -> None:
    """Read command-line arguments and run the benchmark."""

    parser = argparse.ArgumentParser(
        description="Benchmark the churn prediction API",
    )

    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of measured API requests",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up requests",
    )

    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_API_URL,
        help="Prediction endpoint URL",
    )

    arguments = parser.parse_args()

    if arguments.requests <= 0:
        raise ValueError(
            "--requests must be greater than zero"
        )

    if arguments.warmup < 0:
        raise ValueError(
            "--warmup cannot be negative"
        )

    report = run_benchmark(
        request_count=arguments.requests,
        warmup_count=arguments.warmup,
        api_url=arguments.url,
    )

    print_report(report)


if __name__ == "__main__":
    main()
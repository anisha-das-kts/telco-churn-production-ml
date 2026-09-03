"""Model-promotion guardrails for challenger evaluation."""

from typing import Any


REQUIRED_MODEL_METRICS = {
    "roc_auc",
    "recall",
}

REQUIRED_RUNTIME_METRICS = {
    "p95_latency_ms",
    "error_rate_percent",
}


def require_numeric_metric(
    metrics: dict,
    metric_name: str,
    metric_group: str,
) -> float:
    """Read and validate one required numeric metric."""

    if metric_name not in metrics:
        raise ValueError(
            f"{metric_group} is missing required metric: "
            f"{metric_name}"
        )

    value: Any = metrics[metric_name]

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        raise ValueError(
            f"{metric_group}.{metric_name} must be numeric"
        )

    return float(value)


def validate_required_metrics(
    candidate_metrics: dict,
    champion_metrics: dict,
    runtime_metrics: dict,
) -> None:
    """Validate required model and runtime metrics."""

    for metric_name in REQUIRED_MODEL_METRICS:
        require_numeric_metric(
            candidate_metrics,
            metric_name,
            "candidate_metrics",
        )

        require_numeric_metric(
            champion_metrics,
            metric_name,
            "champion_metrics",
        )

    for metric_name in REQUIRED_RUNTIME_METRICS:
        require_numeric_metric(
            runtime_metrics,
            metric_name,
            "runtime_metrics",
        )


def evaluate_promotion_guardrails(
    candidate_metrics: dict,
    champion_metrics: dict,
    runtime_metrics: dict,
    minimum_auc: float,
    minimum_auc_improvement: float,
    minimum_recall: float,
    maximum_p95_latency_ms: float,
    maximum_error_rate_percent: float,
) -> dict:
    """Return an auditable promotion or rejection decision."""

    validate_required_metrics(
        candidate_metrics=candidate_metrics,
        champion_metrics=champion_metrics,
        runtime_metrics=runtime_metrics,
    )

    candidate_auc = require_numeric_metric(
        candidate_metrics,
        "roc_auc",
        "candidate_metrics",
    )

    candidate_recall = require_numeric_metric(
        candidate_metrics,
        "recall",
        "candidate_metrics",
    )

    champion_auc = require_numeric_metric(
        champion_metrics,
        "roc_auc",
        "champion_metrics",
    )

    p95_latency_ms = require_numeric_metric(
        runtime_metrics,
        "p95_latency_ms",
        "runtime_metrics",
    )

    error_rate_percent = require_numeric_metric(
        runtime_metrics,
        "error_rate_percent",
        "runtime_metrics",
    )

    required_challenger_auc = (
        champion_auc + minimum_auc_improvement
    )

    checks = {
        "minimum_auc_passed": bool(
            candidate_auc >= minimum_auc
        ),
        "champion_comparison_passed": bool(
            candidate_auc >= required_challenger_auc
        ),
        "minimum_recall_passed": bool(
            candidate_recall >= minimum_recall
        ),
        "latency_guardrail_passed": bool(
            p95_latency_ms
            <= maximum_p95_latency_ms
        ),
        "error_rate_guardrail_passed": bool(
            error_rate_percent
            <= maximum_error_rate_percent
        ),
    }

    all_guardrails_passed = all(
        checks.values()
    )

    failed_guardrails = [
        check_name
        for check_name, passed in checks.items()
        if not passed
    ]

    return {
        "decision": (
            "PROMOTE"
            if all_guardrails_passed
            else "REJECT"
        ),
        "all_guardrails_passed": (
            all_guardrails_passed
        ),
        "checks": checks,
        "failed_guardrails": failed_guardrails,
        "candidate_metrics": {
            "roc_auc": candidate_auc,
            "recall": candidate_recall,
        },
        "champion_metrics": {
            "roc_auc": champion_auc,
            "recall": require_numeric_metric(
                champion_metrics,
                "recall",
                "champion_metrics",
            ),
        },
        "runtime_metrics": {
            "p95_latency_ms": p95_latency_ms,
            "error_rate_percent": (
                error_rate_percent
            ),
        },
        "thresholds": {
            "minimum_auc": float(minimum_auc),
            "minimum_auc_improvement": float(
                minimum_auc_improvement
            ),
            "minimum_recall": float(
                minimum_recall
            ),
            "maximum_p95_latency_ms": float(
                maximum_p95_latency_ms
            ),
            "maximum_error_rate_percent": float(
                maximum_error_rate_percent
            ),
        },
    }
    
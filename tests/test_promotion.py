"""Tests for model-promotion guardrails."""

import pytest

from src.training.promotion import (
    evaluate_promotion_guardrails,
)


DEFAULT_ARGUMENTS = {
    "candidate_metrics": {
        "roc_auc": 0.86,
        "recall": 0.82,
    },
    "champion_metrics": {
        "roc_auc": 0.85,
        "recall": 0.80,
    },
    "runtime_metrics": {
        "p95_latency_ms": 50.0,
        "error_rate_percent": 0.0,
    },
    "minimum_auc": 0.80,
    "minimum_auc_improvement": 0.00,
    "minimum_recall": 0.70,
    "maximum_p95_latency_ms": 200.0,
    "maximum_error_rate_percent": 1.0,
}


def make_arguments(**updates) -> dict:
    """Create promotion arguments with selected updates."""

    arguments = DEFAULT_ARGUMENTS.copy()

    arguments["candidate_metrics"] = (
        DEFAULT_ARGUMENTS[
            "candidate_metrics"
        ].copy()
    )

    arguments["champion_metrics"] = (
        DEFAULT_ARGUMENTS[
            "champion_metrics"
        ].copy()
    )

    arguments["runtime_metrics"] = (
        DEFAULT_ARGUMENTS[
            "runtime_metrics"
        ].copy()
    )

    arguments.update(updates)

    return arguments


def test_candidate_is_promoted_when_all_checks_pass() -> None:
    """A qualified challenger should be promoted."""

    decision = evaluate_promotion_guardrails(
        **make_arguments()
    )

    assert decision["decision"] == "PROMOTE"
    assert decision["all_guardrails_passed"] is True
    assert decision["failed_guardrails"] == []


def test_candidate_below_minimum_auc_is_rejected() -> None:
    """Candidate must satisfy the absolute AUC threshold."""

    decision = evaluate_promotion_guardrails(
        **make_arguments(
            candidate_metrics={
                "roc_auc": 0.79,
                "recall": 0.82,
            }
        )
    )

    assert decision["decision"] == "REJECT"
    assert (
        "minimum_auc_passed"
        in decision["failed_guardrails"]
    )


def test_candidate_below_champion_auc_is_rejected() -> None:
    """Candidate must match or improve champion AUC."""

    decision = evaluate_promotion_guardrails(
        **make_arguments(
            candidate_metrics={
                "roc_auc": 0.84,
                "recall": 0.82,
            }
        )
    )

    assert decision["decision"] == "REJECT"
    assert (
        "champion_comparison_passed"
        in decision["failed_guardrails"]
    )


def test_candidate_with_low_recall_is_rejected() -> None:
    """Candidate must satisfy the recall guardrail."""

    decision = evaluate_promotion_guardrails(
        **make_arguments(
            candidate_metrics={
                "roc_auc": 0.86,
                "recall": 0.65,
            }
        )
    )

    assert decision["decision"] == "REJECT"
    assert (
        "minimum_recall_passed"
        in decision["failed_guardrails"]
    )


def test_candidate_with_high_latency_is_rejected() -> None:
    """Candidate must satisfy the latency guardrail."""

    decision = evaluate_promotion_guardrails(
        **make_arguments(
            runtime_metrics={
                "p95_latency_ms": 250.0,
                "error_rate_percent": 0.0,
            }
        )
    )

    assert decision["decision"] == "REJECT"
    assert (
        "latency_guardrail_passed"
        in decision["failed_guardrails"]
    )


def test_candidate_with_high_error_rate_is_rejected() -> None:
    """Candidate must satisfy the error-rate guardrail."""

    decision = evaluate_promotion_guardrails(
        **make_arguments(
            runtime_metrics={
                "p95_latency_ms": 50.0,
                "error_rate_percent": 2.0,
            }
        )
    )

    assert decision["decision"] == "REJECT"
    assert (
        "error_rate_guardrail_passed"
        in decision["failed_guardrails"]
    )


def test_missing_metric_raises_error() -> None:
    """Missing evidence must block a promotion decision."""

    with pytest.raises(
        ValueError,
        match="missing required metric",
    ):
        evaluate_promotion_guardrails(
            **make_arguments(
                runtime_metrics={
                    "p95_latency_ms": 50.0,
                }
            )
        )
        
"""Tests for model retraining-trigger logic."""

from src.monitoring.retraining_trigger import (
    evaluate_retraining_signals,
)


DEFAULT_ARGUMENTS = {
    "days_since_last_training": 10,
    "new_labeled_rows": 100,
    "recent_auc": 0.84,
    "reference_auc": 0.85,
    "maximum_z_shift": 0.10,
    "retraining_interval_days": 30,
    "minimum_new_labeled_rows": 500,
    "auc_drop_threshold": 0.05,
    "drift_threshold": 0.50,
    "critical_issues": [],
}


def make_arguments(**updates) -> dict:
    """Create test arguments with selected overrides."""

    arguments = DEFAULT_ARGUMENTS.copy()
    arguments.update(updates)
    return arguments


def test_no_signal_continues_monitoring() -> None:
    """No threshold violation should avoid retraining."""

    decision = evaluate_retraining_signals(
        **make_arguments()
    )

    assert decision["retrain_required"] is False
    assert (
        decision["recommended_action"]
        == "CONTINUE_MONITORING"
    )


def test_drift_triggers_retraining() -> None:
    """Feature drift should trigger retraining."""

    decision = evaluate_retraining_signals(
        **make_arguments(maximum_z_shift=1.10)
    )

    assert decision["retrain_required"] is True
    assert decision["signals"]["drift_signal"] is True


def test_auc_drop_triggers_retraining() -> None:
    """Material model-performance degradation should trigger."""

    decision = evaluate_retraining_signals(
        **make_arguments(recent_auc=0.75)
    )

    assert decision["retrain_required"] is True
    assert (
        decision["signals"]["performance_signal"]
        is True
    )


def test_schedule_requires_enough_new_labels() -> None:
    """Elapsed time alone should not trigger retraining."""

    decision = evaluate_retraining_signals(
        **make_arguments(
            days_since_last_training=31,
            new_labeled_rows=100,
        )
    )

    assert decision["signals"]["scheduled_signal"] is False


def test_schedule_and_labels_trigger_retraining() -> None:
    """Schedule plus sufficient labels should trigger."""

    decision = evaluate_retraining_signals(
        **make_arguments(
            days_since_last_training=31,
            new_labeled_rows=500,
        )
    )

    assert decision["retrain_required"] is True
    assert (
        decision["signals"]["scheduled_signal"]
        is True
    )


def test_critical_quality_issue_blocks_retraining() -> None:
    """Bad data should be investigated before retraining."""

    decision = evaluate_retraining_signals(
        **make_arguments(
            maximum_z_shift=1.10,
            critical_issues=[
                "Required column MonthlyCharges is missing"
            ],
        )
    )

    assert decision["retrain_required"] is False
    assert (
        decision["recommended_action"]
        == "BLOCK_RETRAINING_AND_INVESTIGATE_DATA"
    )
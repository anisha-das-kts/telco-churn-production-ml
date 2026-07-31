"""Retraining decision logic using schedule, performance and drift."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

MONITORING_REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
    / "monitoring_report.json"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "model.joblib"
)

DECISION_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
    / "retraining_decision.json"
)


def evaluate_retraining_signals(
    days_since_last_training: int,
    new_labeled_rows: int,
    recent_auc: float | None,
    reference_auc: float,
    maximum_z_shift: float,
    retraining_interval_days: int,
    minimum_new_labeled_rows: int,
    auc_drop_threshold: float,
    drift_threshold: float,
    critical_issues: list[str] | None = None,
) -> dict:
    """Evaluate three independent retraining signals."""

    critical_issues = critical_issues or []

    scheduled_signal = bool(
        days_since_last_training
        >= retraining_interval_days
        and new_labeled_rows
        >= minimum_new_labeled_rows
    )

    performance_signal = bool(
        recent_auc is not None
        and recent_auc
        < reference_auc - auc_drop_threshold
    )

    drift_signal = bool(
        maximum_z_shift > drift_threshold
    )

    reasons = []

    if scheduled_signal:
        reasons.append(
            "Scheduled interval reached with sufficient "
            "new labelled observations"
        )

    if performance_signal:
        reasons.append(
            "Recent ROC AUC dropped beyond the configured "
            "performance threshold"
        )

    if drift_signal:
        reasons.append(
            "Numerical feature drift exceeded the "
            "configured threshold"
        )

    # Retraining must not run on data with critical quality issues.
    if critical_issues:
        retrain_required = False
        action = "BLOCK_RETRAINING_AND_INVESTIGATE_DATA"
    elif reasons:
        retrain_required = True
        action = "START_RETRAINING_PIPELINE"
    else:
        retrain_required = False
        action = "CONTINUE_MONITORING"

    return {
        "retrain_required": retrain_required,
        "recommended_action": action,
        "signals": {
            "scheduled_signal": scheduled_signal,
            "performance_signal": performance_signal,
            "drift_signal": drift_signal,
        },
        "reasons": reasons,
        "critical_issues": critical_issues,
    }


def load_inputs() -> tuple[dict, dict, dict]:
    """Load configuration, monitoring and production metrics."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    with MONITORING_REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as report_file:
        monitoring_report = json.load(report_file)

    model_artifact = joblib.load(
        PRODUCTION_MODEL_PATH
    )

    return config, monitoring_report, model_artifact


def make_retraining_decision(
    days_since_last_training: int,
    new_labeled_rows: int,
    recent_auc: float | None,
) -> dict:
    """Create and save a retraining decision."""

    config, monitoring_report, model_artifact = (
        load_inputs()
    )

    reference_auc = float(
        model_artifact["metrics"]["roc_auc"]
    )

    maximum_z_shift = float(
        monitoring_report["maximum_z_shift"]
    )

    decision = evaluate_retraining_signals(
        days_since_last_training=(
            days_since_last_training
        ),
        new_labeled_rows=new_labeled_rows,
        recent_auc=recent_auc,
        reference_auc=reference_auc,
        maximum_z_shift=maximum_z_shift,
        retraining_interval_days=int(
            config["monitoring"][
                "retraining_interval_days"
            ]
        ),
        minimum_new_labeled_rows=int(
            config["monitoring"][
                "minimum_new_labeled_rows"
            ]
        ),
        auc_drop_threshold=float(
            config["monitoring"][
                "auc_drop_threshold"
            ]
        ),
        drift_threshold=float(
            config["monitoring"][
                "numerical_z_shift_threshold"
            ]
        ),
        critical_issues=monitoring_report.get(
            "critical_issues",
            [],
        ),
    )

    decision.update(
        {
            "decision_timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "input_signals": {
                "days_since_last_training": (
                    days_since_last_training
                ),
                "new_labeled_rows": new_labeled_rows,
                "recent_auc": recent_auc,
                "reference_auc": reference_auc,
                "maximum_z_shift": maximum_z_shift,
            },
            "thresholds": {
                "retraining_interval_days": int(
                    config["monitoring"][
                        "retraining_interval_days"
                    ]
                ),
                "minimum_new_labeled_rows": int(
                    config["monitoring"][
                        "minimum_new_labeled_rows"
                    ]
                ),
                "auc_drop_threshold": float(
                    config["monitoring"][
                        "auc_drop_threshold"
                    ]
                ),
                "drift_threshold": float(
                    config["monitoring"][
                        "numerical_z_shift_threshold"
                    ]
                ),
            },
        }
    )

    DECISION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with DECISION_PATH.open(
        "w",
        encoding="utf-8",
    ) as decision_file:
        json.dump(
            decision,
            decision_file,
            indent=4,
        )

    return decision


def print_decision(decision: dict) -> None:
    """Print the retraining decision."""

    print()
    print("MODEL RETRAINING DECISION")
    print("=" * 55)
    print(
        f"Retraining required: "
        f"{decision['retrain_required']}"
    )
    print(
        f"Recommended action: "
        f"{decision['recommended_action']}"
    )
    print(
        f"Scheduled signal: "
        f"{decision['signals']['scheduled_signal']}"
    )
    print(
        f"Performance signal: "
        f"{decision['signals']['performance_signal']}"
    )
    print(
        f"Drift signal: "
        f"{decision['signals']['drift_signal']}"
    )

    for reason in decision["reasons"]:
        print(f"REASON | {reason}")

    for issue in decision["critical_issues"]:
        print(f"BLOCKER | {issue}")

    print("=" * 55)
    print(f"Decision saved to: {DECISION_PATH}")


def main() -> None:
    """Read signals and evaluate retraining."""

    parser = argparse.ArgumentParser(
        description="Evaluate model retraining signals",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=10,
        help="Days since the production model was trained",
    )

    parser.add_argument(
        "--new-labeled-rows",
        type=int,
        default=100,
        help="Number of newly labelled observations",
    )

    parser.add_argument(
        "--recent-auc",
        type=float,
        default=None,
        help="ROC AUC calculated on recent labelled feedback",
    )

    arguments = parser.parse_args()

    decision = make_retraining_decision(
        days_since_last_training=arguments.days,
        new_labeled_rows=arguments.new_labeled_rows,
        recent_auc=arguments.recent_auc,
    )

    print_decision(decision)


if __name__ == "__main__":
    main()
"""Lightweight data-quality and numerical drift monitoring."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from src.features.build_features import (
    RAW_FEATURE_COLUMNS,
    build_features,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

REFERENCE_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_data.csv"
)

RECENT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "recent_customer_batch.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
    / "monitoring_report.json"
)


MONITORED_NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "service_count",
    "security_support_count",
    "charges_per_service",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load monitoring thresholds."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return yaml.safe_load(config_file)


def calculate_numerical_drift(
    reference_data: pd.DataFrame,
    recent_data: pd.DataFrame,
    threshold: float,
) -> tuple[dict, list[str]]:
    """
    Compare recent and reference means.

    z_shift is the absolute mean difference divided by the
    reference standard deviation.
    """

    drift_results = {}
    warnings = []

    for feature in MONITORED_NUMERIC_FEATURES:
        reference_values = pd.to_numeric(
            reference_data[feature],
            errors="coerce",
        ).dropna()

        recent_values = pd.to_numeric(
            recent_data[feature],
            errors="coerce",
        ).dropna()

        reference_mean = float(
            reference_values.mean()
        )
        reference_std = float(
            reference_values.std(ddof=0)
        )
        recent_mean = float(
            recent_values.mean()
        )
        recent_std = float(
            recent_values.std(ddof=0)
        )

        if reference_std == 0:
            z_shift = 0.0
        else:
            z_shift = float(
                abs(recent_mean - reference_mean)
                / reference_std
            )

        drift_detected = bool(
            z_shift > threshold
        )

        drift_results[feature] = {
            "reference_mean": reference_mean,
            "reference_std": reference_std,
            "recent_mean": recent_mean,
            "recent_std": recent_std,
            "z_shift": z_shift,
            "threshold": threshold,
            "drift_detected": drift_detected,
        }

        if drift_detected:
            warnings.append(
                f"DRIFT: {feature} mean shifted by "
                f"{z_shift:.3f} reference standard "
                f"deviations"
            )

    return drift_results, warnings


def check_data_quality(
    recent_data: pd.DataFrame,
    maximum_missing_rate: float,
) -> tuple[dict, list[str], list[str]]:
    """Check schema, missingness and numeric ranges."""

    warnings = []
    critical_issues = []

    missing_columns = sorted(
        set(RAW_FEATURE_COLUMNS)
        - set(recent_data.columns)
    )

    if missing_columns:
        critical_issues.append(
            f"Missing required columns: {missing_columns}"
        )

        return (
            {
                "missing_columns": missing_columns,
            },
            warnings,
            critical_issues,
        )

    missing_rates = {
        column: float(
            recent_data[column].isna().mean()
        )
        for column in RAW_FEATURE_COLUMNS
    }

    for column, missing_rate in missing_rates.items():
        if missing_rate > maximum_missing_rate:
            warnings.append(
                f"QUALITY: {column} missing rate "
                f"{missing_rate:.2%} exceeds "
                f"{maximum_missing_rate:.2%}"
            )

    tenure = pd.to_numeric(
        recent_data["tenure"],
        errors="coerce",
    )

    monthly_charges = pd.to_numeric(
        recent_data["MonthlyCharges"],
        errors="coerce",
    )

    total_charges = pd.to_numeric(
        recent_data["TotalCharges"],
        errors="coerce",
    )

    negative_tenure = int(
        (tenure.dropna() < 0).sum()
    )
    negative_monthly_charges = int(
        (monthly_charges.dropna() < 0).sum()
    )
    negative_total_charges = int(
        (total_charges.dropna() < 0).sum()
    )

    if negative_tenure > 0:
        critical_issues.append(
            f"Found {negative_tenure} negative tenure values"
        )

    if negative_monthly_charges > 0:
        critical_issues.append(
            f"Found {negative_monthly_charges} "
            "negative MonthlyCharges values"
        )

    if negative_total_charges > 0:
        critical_issues.append(
            f"Found {negative_total_charges} "
            "negative TotalCharges values"
        )

    quality_results = {
        "row_count": int(len(recent_data)),
        "missing_columns": missing_columns,
        "missing_rates": missing_rates,
        "negative_tenure": negative_tenure,
        "negative_monthly_charges": (
            negative_monthly_charges
        ),
        "negative_total_charges": (
            negative_total_charges
        ),
    }

    return (
        quality_results,
        warnings,
        critical_issues,
    )


def run_monitoring() -> dict:
    """Run data-quality and drift checks."""

    config = load_config()

    maximum_missing_rate = float(
        config["monitoring"]["maximum_missing_rate"]
    )

    drift_threshold = float(
        config["monitoring"][
            "numerical_z_shift_threshold"
        ]
    )

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Reference data not found: "
            f"{REFERENCE_DATA_PATH}"
        )

    if not RECENT_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Recent batch not found: "
            f"{RECENT_DATA_PATH}"
        )

    logger.info(
        "Loading reference data from %s",
        REFERENCE_DATA_PATH,
    )
    reference_raw = pd.read_csv(
        REFERENCE_DATA_PATH
    )

    logger.info(
        "Loading recent data from %s",
        RECENT_DATA_PATH,
    )
    recent_raw = pd.read_csv(
        RECENT_DATA_PATH
    )

    quality_results, quality_warnings, critical_issues = (
        check_data_quality(
            recent_data=recent_raw,
            maximum_missing_rate=maximum_missing_rate,
        )
    )

    if critical_issues:
        numerical_drift = {}
        drift_warnings = []
    else:
        reference_features = build_features(
            reference_raw
        )
        recent_features = build_features(
            recent_raw
        )

        numerical_drift, drift_warnings = (
            calculate_numerical_drift(
                reference_data=reference_features,
                recent_data=recent_features,
                threshold=drift_threshold,
            )
        )

    warnings = quality_warnings + drift_warnings

    if critical_issues:
        status = "CRITICAL"
    elif warnings:
        status = "WARNING"
    else:
        status = "HEALTHY"

    z_shifts = [
        result["z_shift"]
        for result in numerical_drift.values()
    ]

    maximum_z_shift = (
        float(max(z_shifts))
        if z_shifts
        else 0.0
    )

    report = {
        "report_timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "status": status,
        "reference_data_path": str(
            REFERENCE_DATA_PATH
        ),
        "recent_data_path": str(
            RECENT_DATA_PATH
        ),
        "reference_rows": int(
            len(reference_raw)
        ),
        "recent_rows": int(
            len(recent_raw)
        ),
        "thresholds": {
            "maximum_missing_rate": (
                maximum_missing_rate
            ),
            "numerical_z_shift_threshold": (
                drift_threshold
            ),
        },
        "data_quality": quality_results,
        "numerical_drift": numerical_drift,
        "maximum_z_shift": maximum_z_shift,
        "warnings": warnings,
        "critical_issues": critical_issues,
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


def print_monitoring_summary(report: dict) -> None:
    """Print monitoring results for demonstration."""

    print()
    print("ML DATA MONITORING REPORT")
    print("=" * 55)
    print(f"Status: {report['status']}")
    print(
        f"Reference rows: {report['reference_rows']}"
    )
    print(f"Recent rows: {report['recent_rows']}")
    print(
        f"Maximum numerical z-shift: "
        f"{report['maximum_z_shift']:.3f}"
    )
    print(
        f"Warnings detected: "
        f"{len(report['warnings'])}"
    )
    print(
        f"Critical issues: "
        f"{len(report['critical_issues'])}"
    )

    for warning in report["warnings"]:
        print(f"WARNING | {warning}")

    for issue in report["critical_issues"]:
        print(f"CRITICAL | {issue}")

    print("=" * 55)
    print(f"Report saved to: {REPORT_PATH}")


def main() -> None:
    """Run monitoring and print its summary."""

    report = run_monitoring()
    print_monitoring_summary(report)


if __name__ == "__main__":
    main()
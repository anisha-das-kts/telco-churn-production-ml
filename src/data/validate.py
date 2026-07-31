"""Initial validation checks for the raw Telco Churn dataset."""

import json
import logging
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

REPORT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "monitoring"
    / "initial_data_quality_report.json"
)

EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_raw_data() -> dict:
    """Validate schema, missing values, duplicates and value ranges."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset was not found: {DATA_PATH}")

    logger.info("Reading dataset from %s", DATA_PATH)

    dataframe = pd.read_csv(DATA_PATH)

    missing_columns = sorted(
        set(EXPECTED_COLUMNS) - set(dataframe.columns)
    )
    unexpected_columns = sorted(
        set(dataframe.columns) - set(EXPECTED_COLUMNS)
    )

    exact_duplicate_rows = int(dataframe.duplicated().sum())
    duplicate_customer_ids = int(
        dataframe["customerID"].duplicated().sum()
    )

    blank_total_charges = int(
        dataframe["TotalCharges"]
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    numeric_total_charges = pd.to_numeric(
        dataframe["TotalCharges"],
        errors="coerce",
    )

    invalid_total_charges = int(
        numeric_total_charges.isna().sum()
    )

    negative_tenure = int((dataframe["tenure"] < 0).sum())
    negative_monthly_charges = int(
        (dataframe["MonthlyCharges"] < 0).sum()
    )
    negative_total_charges = int(
        (numeric_total_charges.dropna() < 0).sum()
    )

    allowed_target_values = {"Yes", "No"}
    actual_target_values = set(
        dataframe["Churn"].dropna().unique()
    )
    invalid_target_values = sorted(
        actual_target_values - allowed_target_values
    )

    missing_values = {
        column: int(count)
        for column, count in dataframe.isna().sum().items()
    }

    target_distribution = {
        str(label): int(count)
        for label, count in dataframe["Churn"]
        .value_counts(dropna=False)
        .items()
    }

    critical_issues = []

    if missing_columns:
        critical_issues.append(
            f"Missing required columns: {missing_columns}"
        )

    if exact_duplicate_rows > 0:
        critical_issues.append(
            f"Found {exact_duplicate_rows} exact duplicate rows"
        )

    if duplicate_customer_ids > 0:
        critical_issues.append(
            f"Found {duplicate_customer_ids} duplicate customer IDs"
        )

    if invalid_target_values:
        critical_issues.append(
            f"Invalid target values: {invalid_target_values}"
        )

    if negative_tenure > 0:
        critical_issues.append(
            f"Found {negative_tenure} negative tenure values"
        )

    if negative_monthly_charges > 0:
        critical_issues.append(
            "Found negative MonthlyCharges values"
        )

    if negative_total_charges > 0:
        critical_issues.append(
            "Found negative TotalCharges values"
        )

    warnings = []

    if invalid_total_charges > 0:
        warnings.append(
            f"{invalid_total_charges} TotalCharges values "
            "cannot be converted to numbers and require cleaning"
        )

    if unexpected_columns:
        warnings.append(
            f"Unexpected columns found: {unexpected_columns}"
        )

    if critical_issues:
        validation_status = "FAIL"
    elif warnings:
        validation_status = "PASS_WITH_WARNINGS"
    else:
        validation_status = "PASS"

    report = {
        "validation_status": validation_status,
        "dataset_path": str(DATA_PATH),
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "missing_columns": missing_columns,
        "unexpected_columns": unexpected_columns,
        "exact_duplicate_rows": exact_duplicate_rows,
        "duplicate_customer_ids": duplicate_customer_ids,
        "blank_total_charges": blank_total_charges,
        "invalid_total_charges": invalid_total_charges,
        "negative_tenure": negative_tenure,
        "negative_monthly_charges": negative_monthly_charges,
        "negative_total_charges": negative_total_charges,
        "missing_values": missing_values,
        "target_distribution": target_distribution,
        "critical_issues": critical_issues,
        "warnings": warnings,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    return report


def main() -> None:
    """Run data validation and print the summary."""

    report = validate_raw_data()

    logger.info("Validation status: %s", report["validation_status"])
    logger.info("Rows: %s", report["row_count"])
    logger.info("Columns: %s", report["column_count"])
    logger.info(
        "Exact duplicate rows: %s",
        report["exact_duplicate_rows"],
    )
    logger.info(
        "Duplicate customer IDs: %s",
        report["duplicate_customer_ids"],
    )
    logger.info(
        "Invalid TotalCharges values: %s",
        report["invalid_total_charges"],
    )
    logger.info(
        "Target distribution: %s",
        report["target_distribution"],
    )

    for warning in report["warnings"]:
        logger.warning(warning)

    for issue in report["critical_issues"]:
        logger.error(issue)

    logger.info("Quality report saved to %s", REPORT_PATH)


if __name__ == "__main__":
    main()
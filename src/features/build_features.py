"""Shared cleaning and feature engineering for training and inference."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

PREVIEW_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "feature_preview.csv"
)


RAW_FEATURE_COLUMNS = [
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
]


ENGINEERED_FEATURE_COLUMNS = [
    "avg_monthly_spend",
    "service_count",
    "security_support_count",
    "streaming_service_count",
    "charges_per_service",
    "tenure_group",
    "is_month_to_month",
    "has_auto_payment",
    "has_internet",
    "support_gap",
    "high_charge_short_tenure",
    "contract_tenure_interaction",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def validate_feature_inputs(dataframe: pd.DataFrame) -> None:
    """Check whether all fields required for feature engineering exist."""

    missing_columns = sorted(
        set(RAW_FEATURE_COLUMNS) - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns required for feature engineering: "
            f"{missing_columns}"
        )


def yes_indicator(series: pd.Series) -> pd.Series:
    """Convert a Yes/No-style service column into a binary indicator."""

    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .eq("yes")
        .astype(int)
    )


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw fields and construct production features.

    This same function will be called during training, API inference,
    batch prediction and monitoring to reduce training-serving skew.
    """

    validate_feature_inputs(dataframe)

    features = dataframe.copy()

    # Standardise numeric fields.
    features["tenure"] = pd.to_numeric(
        features["tenure"],
        errors="coerce",
    )

    features["MonthlyCharges"] = pd.to_numeric(
        features["MonthlyCharges"],
        errors="coerce",
    )

    features["TotalCharges"] = pd.to_numeric(
        features["TotalCharges"],
        errors="coerce",
    )

    # In this dataset, blank TotalCharges values belong to customers with
    # zero tenure. Estimate missing lifetime charges as monthly charge
    # multiplied by tenure. This produces zero for a new customer.
    estimated_total_charges = (
        features["MonthlyCharges"] * features["tenure"]
    )

    features["TotalCharges"] = features["TotalCharges"].fillna(
        estimated_total_charges
    )

    # Prevent invalid negative values from silently entering the model.
    if (features["tenure"].dropna() < 0).any():
        raise ValueError("tenure cannot contain negative values")

    if (features["MonthlyCharges"].dropna() < 0).any():
        raise ValueError("MonthlyCharges cannot contain negative values")

    if (features["TotalCharges"].dropna() < 0).any():
        raise ValueError("TotalCharges cannot contain negative values")

    # Feature 1: average historical monthly spending.
    safe_tenure = features["tenure"].clip(lower=1)

    features["avg_monthly_spend"] = (
        features["TotalCharges"] / safe_tenure
    )

    # Service indicators.
    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    service_indicators = pd.DataFrame(
        {
            column: yes_indicator(features[column])
            for column in service_columns
        },
        index=features.index,
    )

    # Feature 2: number of subscribed services.
    features["service_count"] = service_indicators.sum(axis=1)

    # Feature 3: number of security/support products.
    features["security_support_count"] = (
        yes_indicator(features["OnlineSecurity"])
        + yes_indicator(features["DeviceProtection"])
        + yes_indicator(features["TechSupport"])
    )

    # Feature 4: number of streaming services.
    features["streaming_service_count"] = (
        yes_indicator(features["StreamingTV"])
        + yes_indicator(features["StreamingMovies"])
    )

    # Feature 5: monthly charge normalised by active service count.
    safe_service_count = features["service_count"].clip(lower=1)

    features["charges_per_service"] = (
        features["MonthlyCharges"] / safe_service_count
    )

    # Feature 6: non-linear customer-tenure category.
    features["tenure_group"] = pd.cut(
        features["tenure"],
        bins=[-1, 6, 12, 24, 48, np.inf],
        labels=[
            "new",
            "early",
            "developing",
            "established",
            "loyal",
        ],
    ).astype("string")

    # Feature 7: flexible month-to-month contract indicator.
    features["is_month_to_month"] = (
        features["Contract"]
        .astype("string")
        .str.strip()
        .str.lower()
        .eq("month-to-month")
        .astype(int)
    )

    # Feature 8: automatic payment indicator.
    features["has_auto_payment"] = (
        features["PaymentMethod"]
        .astype("string")
        .str.lower()
        .str.contains("automatic", na=False)
        .astype(int)
    )

    # Feature 9: whether the customer uses internet service.
    features["has_internet"] = (
        features["InternetService"]
        .astype("string")
        .str.strip()
        .str.lower()
        .ne("no")
        .astype(int)
    )

    # Feature 10: internet customer without technical support.
    features["support_gap"] = (
        features["has_internet"].eq(1)
        & yes_indicator(features["TechSupport"]).eq(0)
    ).astype(int)

    # Feature 11: expensive account still in its early tenure.
    # The fixed business thresholds can be reproduced during serving.
    features["high_charge_short_tenure"] = (
        features["MonthlyCharges"].gt(80)
        & features["tenure"].lt(12)
    ).astype(int)

    # Feature 12: interaction between contract and tenure stage.
    features["contract_tenure_interaction"] = (
        features["Contract"].astype("string").str.strip()
        + "_"
        + features["tenure_group"].fillna("unknown")
    )

    return features


def main() -> None:
    """Build features from the raw dataset and save a preview."""

    logger.info("Reading raw data from %s", RAW_DATA_PATH)

    raw_data = pd.read_csv(RAW_DATA_PATH)

    engineered_data = build_features(raw_data)

    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

    engineered_data.head(25).to_csv(
        PREVIEW_PATH,
        index=False,
    )

    engineered_missing_values = {
        column: int(engineered_data[column].isna().sum())
        for column in ENGINEERED_FEATURE_COLUMNS
    }

    logger.info("Input shape: %s", raw_data.shape)
    logger.info("Output shape: %s", engineered_data.shape)
    logger.info(
        "Engineered features: %s",
        ENGINEERED_FEATURE_COLUMNS,
    )
    logger.info(
        "Engineered-feature missing values: %s",
        engineered_missing_values,
    )
    logger.info(
        "Remaining missing TotalCharges values: %s",
        int(engineered_data["TotalCharges"].isna().sum()),
    )
    logger.info("Preview saved to %s", PREVIEW_PATH)


if __name__ == "__main__":
    main()
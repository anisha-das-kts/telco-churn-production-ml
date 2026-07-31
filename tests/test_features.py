"""Tests for shared cleaning and feature engineering."""

import numpy as np
import pandas as pd
import pytest

from src.features.build_features import (
    ENGINEERED_FEATURE_COLUMNS,
    build_features,
)


def create_sample_data() -> pd.DataFrame:
    """Create representative customer records for unit testing."""

    return pd.DataFrame(
        [
            {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 0,
                "PhoneService": "No",
                "MultipleLines": "No phone service",
                "InternetService": "No",
                "OnlineSecurity": "No internet service",
                "OnlineBackup": "No internet service",
                "DeviceProtection": "No internet service",
                "TechSupport": "No internet service",
                "StreamingTV": "No internet service",
                "StreamingMovies": "No internet service",
                "Contract": "Month-to-month",
                "PaperlessBilling": "No",
                "PaymentMethod": "Mailed check",
                "MonthlyCharges": 20.0,
                "TotalCharges": " ",
            },
            {
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 24,
                "PhoneService": "Yes",
                "MultipleLines": "Yes",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "Yes",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Two year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Bank transfer (automatic)",
                "MonthlyCharges": 100.0,
                "TotalCharges": "2400.0",
            },
        ]
    )


def test_all_engineered_features_are_created() -> None:
    """All documented engineered columns should be present."""

    result = build_features(create_sample_data())

    assert set(ENGINEERED_FEATURE_COLUMNS).issubset(
        result.columns
    )

    assert len(ENGINEERED_FEATURE_COLUMNS) >= 5


def test_blank_total_charges_is_cleaned() -> None:
    """Blank TotalCharges should not remain missing after cleaning."""

    result = build_features(create_sample_data())

    assert result["TotalCharges"].isna().sum() == 0
    assert result.loc[0, "TotalCharges"] == 0.0


def test_ratio_features_are_finite() -> None:
    """Zero tenure or zero services must not cause division by zero."""

    result = build_features(create_sample_data())

    assert np.isfinite(result["avg_monthly_spend"]).all()
    assert np.isfinite(result["charges_per_service"]).all()


def test_service_count_is_correct() -> None:
    """Service count should reflect active Yes-valued services."""

    result = build_features(create_sample_data())

    assert result.loc[0, "service_count"] == 0
    assert result.loc[1, "service_count"] == 8


def test_auto_payment_indicator_is_correct() -> None:
    """Automatic payment methods should be detected."""

    result = build_features(create_sample_data())

    assert result.loc[0, "has_auto_payment"] == 0
    assert result.loc[1, "has_auto_payment"] == 1


def test_support_gap_is_correct() -> None:
    """Internet customers without support should have a support gap."""

    data = create_sample_data()

    data.loc[1, "TechSupport"] = "No"

    result = build_features(data)

    assert result.loc[0, "support_gap"] == 0
    assert result.loc[1, "support_gap"] == 1


def test_negative_tenure_is_rejected() -> None:
    """Negative tenure values should fail validation."""

    data = create_sample_data()

    data.loc[0, "tenure"] = -1

    with pytest.raises(
        ValueError,
        match="tenure cannot contain negative values",
    ):
        build_features(data)


def test_negative_monthly_charge_is_rejected() -> None:
    """Negative monthly charges should fail validation."""

    data = create_sample_data()

    data.loc[0, "MonthlyCharges"] = -20.0

    with pytest.raises(
        ValueError,
        match="MonthlyCharges cannot contain negative values",
    ):
        build_features(data)


def test_missing_required_column_is_rejected() -> None:
    """A missing serving field should produce a clear error."""

    data = create_sample_data().drop(columns=["Contract"])

    with pytest.raises(
        ValueError,
        match="Missing columns required for feature engineering",
    ):
        build_features(data)


def test_feature_engineering_is_deterministic() -> None:
    """The same input should always produce the same features."""

    data = create_sample_data()

    first_result = build_features(data)
    second_result = build_features(data)

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )
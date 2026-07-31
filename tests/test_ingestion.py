"""Tests for the batch ingestion validation logic."""

import pandas as pd
import pytest

from src.data.ingest import validate_incoming_rows


def create_valid_batch() -> pd.DataFrame:
    """Create one valid incoming customer record."""

    return pd.DataFrame(
        [
            {
                "customerID": "TEST-0001",
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
                "Churn": "No",
            }
        ]
    )


def test_valid_row_is_accepted() -> None:
    """A correctly formatted customer should be accepted."""

    valid_rows, rejected_rows = validate_incoming_rows(
        create_valid_batch()
    )

    assert len(valid_rows) == 1
    assert rejected_rows.empty


def test_blank_total_charges_is_accepted() -> None:
    """Blank TotalCharges is valid for a zero-tenure customer."""

    data = create_valid_batch()
    data.loc[0, "TotalCharges"] = " "

    valid_rows, rejected_rows = validate_incoming_rows(data)

    assert len(valid_rows) == 1
    assert rejected_rows.empty


def test_negative_monthly_charge_is_rejected() -> None:
    """Negative charges must be rejected by ingestion."""

    data = create_valid_batch()
    data.loc[0, "MonthlyCharges"] = -10.0

    valid_rows, rejected_rows = validate_incoming_rows(data)

    assert valid_rows.empty
    assert len(rejected_rows) == 1


def test_invalid_target_is_rejected() -> None:
    """Target values other than Yes or No must be rejected."""

    data = create_valid_batch()
    data.loc[0, "Churn"] = "Unknown"

    valid_rows, rejected_rows = validate_incoming_rows(data)

    assert valid_rows.empty
    assert len(rejected_rows) == 1


def test_missing_required_column_raises_error() -> None:
    """A missing required column should fail clearly."""

    data = create_valid_batch().drop(columns=["customerID"])

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        validate_incoming_rows(data)


def test_duplicate_customer_in_batch_is_deduplicated() -> None:
    """The latest duplicate customer record should be retained."""

    data = pd.concat(
        [create_valid_batch(), create_valid_batch()],
        ignore_index=True,
    )

    data.loc[1, "MonthlyCharges"] = 25.0

    valid_rows, rejected_rows = validate_incoming_rows(data)

    assert len(valid_rows) == 1
    assert rejected_rows.empty
    assert valid_rows.iloc[0]["MonthlyCharges"] == 25.0
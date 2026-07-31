"""Pydantic request and response schemas for model serving."""

from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


YesNo = Literal["Yes", "No"]

InternetAddon = Literal[
    "Yes",
    "No",
    "No internet service",
]


class CustomerPredictionRequest(BaseModel):
    """Raw customer fields accepted by the prediction API."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    gender: Literal["Female", "Male"]
    senior_citizen: int = Field(ge=0, le=1)
    partner: YesNo
    dependents: YesNo
    tenure: int = Field(ge=0, le=100)
    phone_service: YesNo

    multiple_lines: Literal[
        "Yes",
        "No",
        "No phone service",
    ]

    internet_service: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    online_security: InternetAddon
    online_backup: InternetAddon
    device_protection: InternetAddon
    tech_support: InternetAddon
    streaming_tv: InternetAddon
    streaming_movies: InternetAddon

    contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    paperless_billing: YesNo

    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    monthly_charges: float = Field(ge=0, le=1000)

    total_charges: float | None = Field(
        default=None,
        ge=0,
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert API fields into the original dataset schema."""

        raw_record = {
            "gender": self.gender,
            "SeniorCitizen": self.senior_citizen,
            "Partner": self.partner,
            "Dependents": self.dependents,
            "tenure": self.tenure,
            "PhoneService": self.phone_service,
            "MultipleLines": self.multiple_lines,
            "InternetService": self.internet_service,
            "OnlineSecurity": self.online_security,
            "OnlineBackup": self.online_backup,
            "DeviceProtection": self.device_protection,
            "TechSupport": self.tech_support,
            "StreamingTV": self.streaming_tv,
            "StreamingMovies": self.streaming_movies,
            "Contract": self.contract,
            "PaperlessBilling": self.paperless_billing,
            "PaymentMethod": self.payment_method,
            "MonthlyCharges": self.monthly_charges,
            "TotalCharges": self.total_charges,
        }

        return pd.DataFrame([raw_record])


class PredictionResponse(BaseModel):
    """Response returned by the churn prediction endpoint."""

    request_id: str
    prediction: int
    churn_label: Literal[
        "Likely to churn",
        "Not likely to churn",
    ]
    churn_probability: float
    risk_level: Literal["low", "medium", "high"]
    model_name: str
    model_version: str
    latency_ms: float
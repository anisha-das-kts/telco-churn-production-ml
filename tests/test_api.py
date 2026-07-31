"""Tests for the FastAPI inference service."""

from fastapi.testclient import TestClient

from src.serving.app import app


client = TestClient(app)


VALID_REQUEST = {
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

def test_root_endpoint() -> None:
    """Root endpoint should provide API navigation."""

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert response.json()["documentation"] == "/docs"

def test_health_endpoint() -> None:
    """Health endpoint should confirm model availability."""

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_model_info_endpoint() -> None:
    """Model metadata should include name and version."""

    response = client.get("/model-info")

    assert response.status_code == 200
    assert response.json()["model_name"] == (
        "Logistic Regression"
    )
    assert response.json()["model_version"] == "1.0.0"
    assert response.json()["feature_count"] == 31


def test_valid_prediction() -> None:
    """A valid request should return a complete prediction."""

    response = client.post(
        "/predict",
        json=VALID_REQUEST,
    )

    assert response.status_code == 200

    result = response.json()

    assert result["prediction"] in [0, 1]
    assert result["risk_level"] in [
        "low",
        "medium",
        "high",
    ]
    assert result["model_name"] == "Logistic Regression"
    assert result["model_version"] == "1.0.0"
    assert "request_id" in result
    assert "latency_ms" in result


def test_probability_is_valid() -> None:
    """Churn probability must remain between zero and one."""

    response = client.post(
        "/predict",
        json=VALID_REQUEST,
    )

    probability = response.json()[
        "churn_probability"
    ]

    assert 0.0 <= probability <= 1.0


def test_negative_charge_is_rejected() -> None:
    """Pydantic should reject a negative charge."""

    invalid_request = VALID_REQUEST.copy()
    invalid_request["monthly_charges"] = -10

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected() -> None:
    """Pydantic should reject an incomplete request."""

    invalid_request = VALID_REQUEST.copy()
    invalid_request.pop("contract")

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422
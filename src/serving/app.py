"""FastAPI service for online churn prediction."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
from fastapi import FastAPI, HTTPException

from src.features.build_features import build_features
from src.serving.schemas import (
    CustomerPredictionRequest,
    PredictionResponse,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "model.joblib"
)

PREDICTION_LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "predictions.jsonl"
)

PREDICTION_THRESHOLD = 0.50


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_model_artifact() -> dict:
    """Load the promoted production model."""

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Production model not found: {MODEL_PATH}. "
            "Run python -m src.training.train first."
        )

    artifact = joblib.load(MODEL_PATH)

    required_keys = {
        "pipeline",
        "model_name",
        "model_version",
        "feature_columns",
        "metrics",
    }

    missing_keys = required_keys - set(artifact)

    if missing_keys:
        raise RuntimeError(
            f"Model artifact is missing keys: "
            f"{sorted(missing_keys)}"
        )

    return artifact


MODEL_ARTIFACT = load_model_artifact()
MODEL = MODEL_ARTIFACT["pipeline"]


app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description=(
        "Online inference service for predicting telecom "
        "customer churn risk."
    ),
    version=MODEL_ARTIFACT["model_version"],
)


def assign_risk_level(probability: float) -> str:
    """Convert churn probability into an operational risk band."""

    if probability < 0.40:
        return "low"

    if probability < 0.70:
        return "medium"

    return "high"


def write_prediction_log(record: dict) -> None:
    """Append non-identifying prediction metadata to a JSONL log."""

    PREDICTION_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PREDICTION_LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            json.dumps(record) + "\n"
        )

@app.get("/")
def root() -> dict:
    """Return API navigation information."""

    return {
        "service": "Telco Customer Churn Prediction API",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "model_info": "/model-info",
        "prediction_endpoint": "/predict",
    }

@app.get("/health")
def health_check() -> dict:
    """Return service and model availability."""

    return {
        "status": "healthy",
        "model_loaded": True,
        "model_name": MODEL_ARTIFACT["model_name"],
        "model_version": MODEL_ARTIFACT["model_version"],
    }


@app.get("/model-info")
def model_info() -> dict:
    """Return deployed model metadata."""

    metrics = MODEL_ARTIFACT["metrics"]

    return {
        "model_name": MODEL_ARTIFACT["model_name"],
        "model_version": MODEL_ARTIFACT["model_version"],
        "feature_count": len(
            MODEL_ARTIFACT["feature_columns"]
        ),
        "prediction_threshold": PREDICTION_THRESHOLD,
        "test_roc_auc": metrics.get("roc_auc"),
        "test_recall": metrics.get("recall"),
        "test_f1_score": metrics.get("f1_score"),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict_customer(
    request: CustomerPredictionRequest,
) -> PredictionResponse:
    """Predict churn probability for one customer."""

    start_time = perf_counter()
    request_id = str(uuid4())

    try:
        raw_features = request.to_dataframe()

        engineered_features = build_features(
            raw_features
        )

        model_input = engineered_features[
            MODEL_ARTIFACT["feature_columns"]
        ]

        churn_probability = float(
            MODEL.predict_proba(model_input)[0, 1]
        )

        prediction = int(
            churn_probability >= PREDICTION_THRESHOLD
        )

        risk_level = assign_risk_level(
            churn_probability
        )

        latency_ms = float(
            (perf_counter() - start_time) * 1000
        )

        log_record = {
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "request_id": request_id,
            "prediction": prediction,
            "churn_probability": churn_probability,
            "risk_level": risk_level,
            "model_name": MODEL_ARTIFACT["model_name"],
            "model_version": MODEL_ARTIFACT[
                "model_version"
            ],
            "latency_ms": latency_ms,
        }

        write_prediction_log(log_record)

        logger.info(
            "request_id=%s prediction=%s "
            "probability=%.4f latency_ms=%.3f",
            request_id,
            prediction,
            churn_probability,
            latency_ms,
        )

        return PredictionResponse(
            request_id=request_id,
            prediction=prediction,
            churn_label=(
                "Likely to churn"
                if prediction == 1
                else "Not likely to churn"
            ),
            churn_probability=round(
                churn_probability,
                6,
            ),
            risk_level=risk_level,
            model_name=MODEL_ARTIFACT["model_name"],
            model_version=MODEL_ARTIFACT[
                "model_version"
            ],
            latency_ms=round(latency_ms, 3),
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Prediction failed for request_id=%s",
            request_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Prediction could not be completed",
        ) from error
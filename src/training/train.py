"""Repeatable training and model-promotion pipeline."""

import logging
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.build_features import build_features
from src.training.evaluate import (
    evaluate_classifier,
    save_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_data.csv"
)

BASELINE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "baseline"
    / "model.joblib"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "candidate"
    / "model.joblib"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "model.joblib"
)

BASELINE_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "eval"
    / "baseline_metrics.json"
)

CANDIDATE_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "eval"
    / "candidate_metrics.json"
)

PRODUCTION_METRICS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "eval"
    / "production_test_metrics.json"
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "eval"
    / "evaluation_summary.json"
)

COMPARISON_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "eval"
    / "model_comparison.md"
)


NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "avg_monthly_spend",
    "service_count",
    "security_support_count",
    "streaming_service_count",
    "charges_per_service",
    "is_month_to_month",
    "has_auto_payment",
    "has_internet",
    "support_gap",
    "high_charge_short_tenure",
]


CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
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
    "tenure_group",
    "contract_tenure_interaction",
]


MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Load project configuration."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        return yaml.safe_load(config_file)


def create_preprocessor() -> ColumnTransformer:
    """Create shared numeric and categorical preprocessing."""

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def create_baseline_pipeline(
    random_state: int,
) -> Pipeline:
    """Create the Logistic Regression baseline."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def create_candidate_pipeline(
    random_state: int,
) -> Pipeline:
    """Create the Random Forest candidate."""

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=4,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def save_model_artifact(
    model,
    model_name: str,
    model_version: str,
    metrics: dict,
    output_path: Path,
) -> None:
    """Save a model together with serving metadata."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact = {
        "pipeline": model,
        "model_name": model_name,
        "model_version": model_version,
        "feature_columns": MODEL_FEATURES,
        "metrics": metrics,
        "trained_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    joblib.dump(
        artifact,
        output_path,
    )


def train_and_time(model, features, target) -> float:
    """Fit a model and return training duration."""

    start_time = perf_counter()

    model.fit(features, target)

    return float(perf_counter() - start_time)


def write_comparison_report(
    baseline_metrics: dict,
    candidate_metrics: dict,
    production_metrics: dict,
    promoted_candidate: bool,
    promotion_checks: dict,
    selected_model_name: str,
) -> None:
    """Write a human-readable model comparison report."""

    decision = (
        "Promote Random Forest candidate"
        if promoted_candidate
        else "Retain Logistic Regression baseline"
    )

    report = f"""# Offline Model Evaluation

## Validation-set comparison

| Model | Accuracy | ROC AUC | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | {baseline_metrics["accuracy"]:.4f} | {baseline_metrics["roc_auc"]:.4f} | {baseline_metrics["precision"]:.4f} | {baseline_metrics["recall"]:.4f} | {baseline_metrics["f1_score"]:.4f} |
| Random Forest | {candidate_metrics["accuracy"]:.4f} | {candidate_metrics["roc_auc"]:.4f} | {candidate_metrics["precision"]:.4f} | {candidate_metrics["recall"]:.4f} | {candidate_metrics["f1_score"]:.4f} |

## Promotion guardrail

- Candidate ROC AUC at least minimum AUC: {promotion_checks["minimum_auc_passed"]}
- Candidate ROC AUC at least matches the baseline: {promotion_checks["baseline_comparison_passed"]}
- Candidate recall at least minimum recall: {promotion_checks["minimum_recall_passed"]}

## Decision

**{decision}**

Selected production model: **{selected_model_name}**

## Final untouched test-set results

| Metric | Result |
|---|---:|
| Accuracy | {production_metrics["accuracy"]:.4f} |
| ROC AUC | {production_metrics["roc_auc"]:.4f} |
| Precision | {production_metrics["precision"]:.4f} |
| Recall | {production_metrics["recall"]:.4f} |
| F1-score | {production_metrics["f1_score"]:.4f} |

The promotion decision was made using validation data. The test
set was used only once after model selection to estimate final
generalisation performance.
"""

    COMPARISON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    COMPARISON_PATH.write_text(
        report,
        encoding="utf-8",
    )


def main() -> None:
    """Execute the complete training pipeline."""

    config = load_config()

    random_state = int(
        config["project"]["random_state"]
    )
    model_version = str(
        config["project"]["model_version"]
    )

    logger.info(
        "Loading training data from %s",
        TRAINING_DATA_PATH,
    )

    raw_data = pd.read_csv(TRAINING_DATA_PATH)

    logger.info(
        "Building shared production features"
    )

    engineered_data = build_features(raw_data)

    target = engineered_data["Churn"].map(
        {"No": 0, "Yes": 1}
    )

    if target.isna().any():
        raise ValueError(
            "Target contains values other than Yes and No"
        )

    target = target.astype(int)
    features = engineered_data[MODEL_FEATURES].copy()

    test_size = float(
        config["split"]["test_size"]
    )
    validation_size = float(
        config["split"]["validation_size"]
    )

    features_train_validation, features_test, target_train_validation, target_test = (
        train_test_split(
            features,
            target,
            test_size=test_size,
            stratify=target,
            random_state=random_state,
        )
    )

    relative_validation_size = (
        validation_size / (1.0 - test_size)
    )

    features_train, features_validation, target_train, target_validation = (
        train_test_split(
            features_train_validation,
            target_train_validation,
            test_size=relative_validation_size,
            stratify=target_train_validation,
            random_state=random_state,
        )
    )

    logger.info(
        "Split sizes - train=%s, validation=%s, test=%s",
        len(features_train),
        len(features_validation),
        len(features_test),
    )

    baseline_model = create_baseline_pipeline(
        random_state
    )
    candidate_model = create_candidate_pipeline(
        random_state
    )

    logger.info(
        "Training Logistic Regression baseline"
    )

    baseline_training_seconds = train_and_time(
        baseline_model,
        features_train,
        target_train,
    )

    baseline_metrics = evaluate_classifier(
        baseline_model,
        features_validation,
        target_validation,
    )
    baseline_metrics["model_name"] = (
        "Logistic Regression"
    )
    baseline_metrics["dataset"] = "validation"
    baseline_metrics["training_seconds"] = (
        baseline_training_seconds
    )

    logger.info(
        "Training Random Forest candidate"
    )

    candidate_training_seconds = train_and_time(
        candidate_model,
        features_train,
        target_train,
    )

    candidate_metrics = evaluate_classifier(
        candidate_model,
        features_validation,
        target_validation,
    )
    candidate_metrics["model_name"] = "Random Forest"
    candidate_metrics["dataset"] = "validation"
    candidate_metrics["training_seconds"] = (
        candidate_training_seconds
    )

    minimum_auc = float(
        config["promotion"]["minimum_auc"]
    )
    minimum_auc_improvement = float(
        config["promotion"][
            "minimum_auc_improvement"
        ]
    )
    minimum_recall = float(
        config["promotion"]["minimum_recall"]
    )

    promotion_checks = {
        "minimum_auc_passed": bool(
            candidate_metrics["roc_auc"]
            >= minimum_auc
        ),
       "baseline_comparison_passed": bool(
            candidate_metrics["roc_auc"]
            >= baseline_metrics["roc_auc"]
            + minimum_auc_improvement
        ),
        "minimum_recall_passed": bool(
            candidate_metrics["recall"]
            >= minimum_recall
        ),
    }

    promoted_candidate = all(
        promotion_checks.values()
    )

    if promoted_candidate:
        selected_template = candidate_model
        selected_model_name = "Random Forest"
    else:
        selected_template = baseline_model
        selected_model_name = "Logistic Regression"

    logger.info(
        "Baseline validation ROC AUC: %.4f",
        baseline_metrics["roc_auc"],
    )
    logger.info(
        "Candidate validation ROC AUC: %.4f",
        candidate_metrics["roc_auc"],
    )
    logger.info(
        "Candidate validation recall: %.4f",
        candidate_metrics["recall"],
    )
    logger.info(
        "Promotion checks: %s",
        promotion_checks,
    )
    logger.info(
        "Selected production model: %s",
        selected_model_name,
    )

    save_json(
        baseline_metrics,
        BASELINE_METRICS_PATH,
    )
    save_json(
        candidate_metrics,
        CANDIDATE_METRICS_PATH,
    )

    save_model_artifact(
        model=baseline_model,
        model_name="Logistic Regression",
        model_version=model_version,
        metrics=baseline_metrics,
        output_path=BASELINE_MODEL_PATH,
    )

    save_model_artifact(
        model=candidate_model,
        model_name="Random Forest",
        model_version=model_version,
        metrics=candidate_metrics,
        output_path=CANDIDATE_MODEL_PATH,
    )

    # Refit the selected model using both training and validation data.
    production_model = clone(selected_template)

    logger.info(
        "Refitting selected model on training + validation data"
    )

    production_training_seconds = train_and_time(
        production_model,
        features_train_validation,
        target_train_validation,
    )

    # The untouched test set is evaluated only after model selection.
    production_metrics = evaluate_classifier(
        production_model,
        features_test,
        target_test,
    )
    production_metrics["model_name"] = (
        selected_model_name
    )
    production_metrics["dataset"] = "test"
    production_metrics["training_seconds"] = (
        production_training_seconds
    )

    save_json(
        production_metrics,
        PRODUCTION_METRICS_PATH,
    )

    save_model_artifact(
        model=production_model,
        model_name=selected_model_name,
        model_version=model_version,
        metrics=production_metrics,
        output_path=PRODUCTION_MODEL_PATH,
    )

    evaluation_summary = {
        "model_version": model_version,
        "random_state": random_state,
        "split_sizes": {
            "train": int(len(features_train)),
            "validation": int(
                len(features_validation)
            ),
            "test": int(len(features_test)),
        },
        "baseline_validation_metrics": (
            baseline_metrics
        ),
        "candidate_validation_metrics": (
            candidate_metrics
        ),
        "promotion_checks": promotion_checks,
        "candidate_promoted": promoted_candidate,
        "selected_production_model": (
            selected_model_name
        ),
        "production_test_metrics": (
            production_metrics
        ),
    }

    save_json(
        evaluation_summary,
        SUMMARY_PATH,
    )

    write_comparison_report(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        production_metrics=production_metrics,
        promoted_candidate=promoted_candidate,
        promotion_checks=promotion_checks,
        selected_model_name=selected_model_name,
    )

    logger.info(
        "Final test ROC AUC: %.4f",
        production_metrics["roc_auc"],
    )
    logger.info(
        "Final test recall: %.4f",
        production_metrics["recall"],
    )
    logger.info(
        "Production model saved to %s",
        PRODUCTION_MODEL_PATH,
    )
    logger.info(
        "Evaluation reports saved to %s",
        BASELINE_METRICS_PATH.parent,
    )


if __name__ == "__main__":
    main()
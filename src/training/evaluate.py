"""Reusable offline evaluation utilities."""

import json
from pathlib import Path
from time import perf_counter

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classifier(model, features, target) -> dict:
    """Evaluate a fitted binary classifier."""

    start_time = perf_counter()

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    inference_seconds = perf_counter() - start_time

    evaluated_rows = int(len(target))

    return {
        "accuracy": float(
            accuracy_score(target, predictions)
        ),
        "roc_auc": float(
            roc_auc_score(target, probabilities)
        ),
        "precision": float(
            precision_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                target,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            target,
            predictions,
        ).tolist(),
        "evaluated_rows": evaluated_rows,
        "total_inference_seconds": float(
            inference_seconds
        ),
        "average_inference_ms_per_row": float(
            (inference_seconds / max(evaluated_rows, 1))
            * 1000
        ),
    }


def save_json(data: dict, output_path: Path) -> None:
    """Save a dictionary as formatted JSON."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=4,
        )
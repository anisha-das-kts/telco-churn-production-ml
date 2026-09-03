"""Tests for loading the API model from the registry."""

from src.registry.model_registry import (
    get_champion_record,
)
from src.serving.app import (
    REGISTRY_ROOT,
    load_model_artifact,
)


def test_serving_loads_registry_champion() -> None:
    """API should load the selected champion version."""

    champion = get_champion_record(
        REGISTRY_ROOT
    )

    artifact = load_model_artifact()

    assert (
        artifact["model_version"]
        == champion["model_version"]
    )

    assert (
        artifact["model_name"]
        == champion["model_name"]
    )


def test_champion_contains_model_metrics() -> None:
    """Champion artifact should retain evaluation metrics."""

    artifact = load_model_artifact()
    metrics = artifact["metrics"]

    assert "roc_auc" in metrics
    assert "recall" in metrics
    assert metrics["roc_auc"] >= 0.80
    assert metrics["recall"] >= 0.70
    
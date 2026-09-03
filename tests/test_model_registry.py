"""Tests for the local versioned model registry."""

from pathlib import Path

import joblib
import pytest
from sklearn.dummy import DummyClassifier

from src.registry.model_registry import (
    get_champion_model_path,
    get_champion_record,
    list_model_versions,
    promote_model,
    register_model,
)


def create_test_artifact(
    destination: Path,
) -> None:
    """Create a small valid model artifact."""

    model = DummyClassifier(
        strategy="prior",
    )

    model.fit(
        [[0], [1], [0], [1]],
        [0, 1, 0, 1],
    )

    artifact = {
        "pipeline": model,
        "model_name": "Test Classifier",
        "model_version": "0.0.0",
        "feature_columns": ["feature"],
        "metrics": {
            "roc_auc": 0.80,
            "recall": 0.75,
        },
    }

    joblib.dump(
        artifact,
        destination,
    )


def test_register_model_creates_versioned_artifacts(
    tmp_path: Path,
) -> None:
    """Registration should save a model and metadata."""

    source_path = tmp_path / "source.joblib"
    registry_root = tmp_path / "registry"

    create_test_artifact(source_path)

    metadata = register_model(
        source_model_path=source_path,
        registry_root=registry_root,
        version="1.0.0",
    )

    model_path = (
        registry_root
        / "1.0.0"
        / "model.joblib"
    )

    metadata_path = (
        registry_root
        / "1.0.0"
        / "metadata.json"
    )

    assert model_path.exists()
    assert metadata_path.exists()
    assert metadata["model_version"] == "1.0.0"
    assert len(metadata["model_sha256"]) == 64


def test_registration_does_not_overwrite_version(
    tmp_path: Path,
) -> None:
    """Registered versions must be immutable."""

    source_path = tmp_path / "source.joblib"
    registry_root = tmp_path / "registry"

    create_test_artifact(source_path)

    register_model(
        source_model_path=source_path,
        registry_root=registry_root,
        version="1.0.0",
    )

    with pytest.raises(FileExistsError):
        register_model(
            source_model_path=source_path,
            registry_root=registry_root,
            version="1.0.0",
        )


def test_promote_and_resolve_champion(
    tmp_path: Path,
) -> None:
    """A registered version should become champion."""

    source_path = tmp_path / "source.joblib"
    registry_root = tmp_path / "registry"

    create_test_artifact(source_path)

    register_model(
        source_model_path=source_path,
        registry_root=registry_root,
        version="1.1.0",
    )

    champion = promote_model(
        registry_root=registry_root,
        version="1.1.0",
        promotion_metadata={
            "decision": "PROMOTED",
        },
    )

    resolved_path = get_champion_model_path(
        registry_root
    )

    assert champion["model_version"] == "1.1.0"
    assert resolved_path.exists()

    loaded_artifact = joblib.load(
        resolved_path
    )

    assert (
        loaded_artifact["model_version"]
        == "1.1.0"
    )


def test_unknown_version_cannot_be_promoted(
    tmp_path: Path,
) -> None:
    """Promotion must reject an unregistered version."""

    registry_root = tmp_path / "registry"

    with pytest.raises(FileNotFoundError):
        promote_model(
            registry_root=registry_root,
            version="9.9.9",
        )


def test_versions_are_sorted_numerically(
    tmp_path: Path,
) -> None:
    """Registry should return versions in numeric order."""

    source_path = tmp_path / "source.joblib"
    registry_root = tmp_path / "registry"

    create_test_artifact(source_path)

    for version in [
        "1.10.0",
        "1.2.0",
        "2.0.0",
    ]:
        register_model(
            source_model_path=source_path,
            registry_root=registry_root,
            version=version,
        )

    assert list_model_versions(
        registry_root
    ) == [
        "1.2.0",
        "1.10.0",
        "2.0.0",
    ]


def test_champion_integrity_is_verified(
    tmp_path: Path,
) -> None:
    """Changes to a champion artifact must be detected."""

    source_path = tmp_path / "source.joblib"
    registry_root = tmp_path / "registry"

    create_test_artifact(source_path)

    register_model(
        source_model_path=source_path,
        registry_root=registry_root,
        version="1.0.0",
    )

    promote_model(
        registry_root=registry_root,
        version="1.0.0",
    )

    champion_record = get_champion_record(
        registry_root
    )

    champion_path = (
        registry_root
        / champion_record["model_path"]
    )

    champion_path.write_bytes(
        champion_path.read_bytes() + b"changed"
    )

    with pytest.raises(
        ValueError,
        match="integrity check failed",
    ):
        get_champion_record(registry_root)
        
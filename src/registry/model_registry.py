"""Local versioned model registry with champion-model management."""

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


SEMANTIC_VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+$"
)

REQUIRED_ARTIFACT_KEYS = {
    "pipeline",
    "model_name",
    "model_version",
    "feature_columns",
    "metrics",
}

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
CHAMPION_FILENAME = "champion.json"


def validate_version(version: str) -> None:
    """Validate a three-part semantic model version."""

    if not SEMANTIC_VERSION_PATTERN.fullmatch(version):
        raise ValueError(
            "Model version must use semantic versioning, "
            "for example: 1.0.0"
        )


def calculate_sha256(file_path: Path) -> str:
    """Calculate the SHA-256 digest of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as model_file:
        for block in iter(
            lambda: model_file.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def validate_model_artifact(artifact: Any) -> None:
    """Ensure a loaded artifact has the required structure."""

    if not isinstance(artifact, dict):
        raise ValueError(
            "Model artifact must be a dictionary"
        )

    missing_keys = (
        REQUIRED_ARTIFACT_KEYS - set(artifact)
    )

    if missing_keys:
        raise ValueError(
            "Model artifact is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    if not hasattr(
        artifact["pipeline"],
        "predict",
    ):
        raise ValueError(
            "Model pipeline must provide predict()"
        )

    if not hasattr(
        artifact["pipeline"],
        "predict_proba",
    ):
        raise ValueError(
            "Model pipeline must provide predict_proba()"
        )


def write_json_atomic(
    data: dict,
    destination: Path,
) -> None:
    """Write JSON through a temporary file."""

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(
            data,
            output_file,
            indent=4,
        )

    temporary_path.replace(destination)


def register_model(
    source_model_path: Path,
    registry_root: Path,
    version: str,
    additional_metadata: dict | None = None,
) -> dict:
    """Register a model artifact under an immutable version."""

    source_model_path = Path(source_model_path)
    registry_root = Path(registry_root)

    validate_version(version)

    if not source_model_path.exists():
        raise FileNotFoundError(
            f"Source model not found: "
            f"{source_model_path}"
        )

    version_directory = registry_root / version

    if version_directory.exists():
        raise FileExistsError(
            f"Model version {version} is already registered"
        )

    artifact = joblib.load(source_model_path)
    validate_model_artifact(artifact)

    registered_artifact = dict(artifact)
    registered_artifact["model_version"] = version

    model_path = (
        version_directory / MODEL_FILENAME
    )
    metadata_path = (
        version_directory / METADATA_FILENAME
    )

    version_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    try:
        temporary_model_path = (
            version_directory / "model.joblib.tmp"
        )

        joblib.dump(
            registered_artifact,
            temporary_model_path,
        )

        temporary_model_path.replace(model_path)

        metadata = {
            "model_name": registered_artifact[
                "model_name"
            ],
            "model_version": version,
            "registered_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "model_sha256": calculate_sha256(
                model_path
            ),
            "feature_count": len(
                registered_artifact[
                    "feature_columns"
                ]
            ),
            "metrics": registered_artifact[
                "metrics"
            ],
        }

        if additional_metadata:
            metadata["additional_metadata"] = (
                additional_metadata
            )

        write_json_atomic(
            metadata,
            metadata_path,
        )

    except Exception:
        shutil.rmtree(
            version_directory,
            ignore_errors=True,
        )
        raise

    return metadata


def load_registered_metadata(
    registry_root: Path,
    version: str,
) -> dict:
    """Load metadata for one registered model version."""

    registry_root = Path(registry_root)
    validate_version(version)

    metadata_path = (
        registry_root
        / version
        / METADATA_FILENAME
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Registered model version not found: "
            f"{version}"
        )

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as metadata_file:
        return json.load(metadata_file)


def list_model_versions(
    registry_root: Path,
) -> list[str]:
    """Return registered versions in numeric order."""

    registry_root = Path(registry_root)

    if not registry_root.exists():
        return []

    versions = [
        path.name
        for path in registry_root.iterdir()
        if path.is_dir()
        and SEMANTIC_VERSION_PATTERN.fullmatch(
            path.name
        )
    ]

    return sorted(
        versions,
        key=lambda value: tuple(
            int(part)
            for part in value.split(".")
        ),
    )


def promote_model(
    registry_root: Path,
    version: str,
    promotion_metadata: dict | None = None,
) -> dict:
    """Make a registered model version the champion."""

    registry_root = Path(registry_root)
    validate_version(version)

    model_path = (
        registry_root / version / MODEL_FILENAME
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot promote unregistered version: "
            f"{version}"
        )

    artifact = joblib.load(model_path)
    validate_model_artifact(artifact)

    champion_record = {
        "model_name": artifact["model_name"],
        "model_version": version,
        "model_path": (
            f"{version}/{MODEL_FILENAME}"
        ),
        "model_sha256": calculate_sha256(
            model_path
        ),
        "promoted_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if promotion_metadata:
        champion_record["promotion_metadata"] = (
            promotion_metadata
        )

    write_json_atomic(
        champion_record,
        registry_root / CHAMPION_FILENAME,
    )

    return champion_record


def get_champion_record(
    registry_root: Path,
) -> dict:
    """Load and verify the current champion record."""

    registry_root = Path(registry_root)
    champion_path = (
        registry_root / CHAMPION_FILENAME
    )

    if not champion_path.exists():
        raise FileNotFoundError(
            "No champion model has been selected"
        )

    with champion_path.open(
        "r",
        encoding="utf-8",
    ) as champion_file:
        champion_record = json.load(
            champion_file
        )

    model_path = (
        registry_root
        / champion_record["model_path"]
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Champion model artifact is missing: "
            f"{model_path}"
        )

    actual_hash = calculate_sha256(model_path)

    if actual_hash != champion_record["model_sha256"]:
        raise ValueError(
            "Champion model integrity check failed"
        )

    return champion_record


def get_champion_model_path(
    registry_root: Path,
) -> Path:
    """Return the verified champion model path."""

    registry_root = Path(registry_root)

    champion_record = get_champion_record(
        registry_root
    )

    return (
        registry_root
        / champion_record["model_path"]
    )
    
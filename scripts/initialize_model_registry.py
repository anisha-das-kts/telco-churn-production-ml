"""Initialize the registry using the existing production model."""

from pathlib import Path

import yaml

from src.registry.model_registry import (
    get_champion_record,
    list_model_versions,
    load_registered_metadata,
    promote_model,
    register_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "config.yaml"
)

PRODUCTION_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "production"
    / "model.joblib"
)

REGISTRY_ROOT = (
    PROJECT_ROOT / "models" / "registry"
)


def load_model_version() -> str:
    """Load the initial version from project configuration."""

    with CONFIG_PATH.open(
        "r",
        encoding="utf-8",
    ) as config_file:
        config = yaml.safe_load(config_file)

    return str(
        config["project"]["model_version"]
    )


def initialize_registry() -> dict:
    """Register and promote the existing production model."""

    version = load_model_version()

    if not PRODUCTION_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Production model was not found. Run "
            "`python -m src.training.train` first."
        )

    registered_versions = list_model_versions(
        REGISTRY_ROOT
    )

    if version not in registered_versions:
        metadata = register_model(
            source_model_path=PRODUCTION_MODEL_PATH,
            registry_root=REGISTRY_ROOT,
            version=version,
            additional_metadata={
                "source": "initial production model",
                "reason": "registry initialization",
            },
        )

        print(
            f"Registered model version: {version}"
        )
        print(
            f"Model SHA-256: "
            f"{metadata['model_sha256']}"
        )
    else:
        metadata = load_registered_metadata(
            registry_root=REGISTRY_ROOT,
            version=version,
        )

        print(
            f"Model version {version} is already "
            "registered"
        )

    champion_path = (
        REGISTRY_ROOT / "champion.json"
    )

    if champion_path.exists():
        champion = get_champion_record(
            REGISTRY_ROOT
        )

        if champion["model_version"] != version:
            raise RuntimeError(
                "A different champion already exists. "
                "Use the controlled promotion workflow "
                "instead of replacing it."
            )

        print(
            f"Champion is already version: {version}"
        )
    else:
        champion = promote_model(
            registry_root=REGISTRY_ROOT,
            version=version,
            promotion_metadata={
                "decision": "INITIAL_CHAMPION",
                "all_guardrails_passed": True,
            },
        )

        print(
            f"Promoted initial champion: {version}"
        )

    return {
        "registered_model": metadata,
        "champion": champion,
    }


def main() -> None:
    """Initialize and display registry information."""

    result = initialize_registry()

    champion = result["champion"]

    print()
    print("MODEL REGISTRY INITIALIZED")
    print("=" * 55)
    print(
        f"Model name: {champion['model_name']}"
    )
    print(
        f"Champion version: "
        f"{champion['model_version']}"
    )
    print(
        f"Registry directory: {REGISTRY_ROOT}"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
    
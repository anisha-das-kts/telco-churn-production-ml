"""Batch ingestion pipeline for incoming Telco customer files."""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INCOMING_DIRECTORY = PROJECT_ROOT / "data" / "incoming"

TRAINING_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "training_data.csv"
)

LOG_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "ingestion.log"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "ingestion_manifest.json"
)

REJECTED_DIRECTORY = (
    PROJECT_ROOT
    / "artifacts"
    / "logs"
    / "rejected_rows"
)


EXPECTED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
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
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def configure_logger() -> logging.Logger:
    """Configure console and file logging."""

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    ingestion_logger = logging.getLogger("telco_ingestion")
    ingestion_logger.setLevel(logging.INFO)
    ingestion_logger.propagate = False

    if not ingestion_logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(
            LOG_PATH,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)

        ingestion_logger.addHandler(console_handler)
        ingestion_logger.addHandler(file_handler)

    return ingestion_logger


logger = configure_logger()


def calculate_file_hash(file_path: Path) -> str:
    """Calculate a SHA-256 hash for idempotent file processing."""

    file_hasher = hashlib.sha256()

    with file_path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(8192), b""):
            file_hasher.update(chunk)

    return file_hasher.hexdigest()


def load_manifest() -> dict:
    """Load information about previously ingested files."""

    if not MANIFEST_PATH.exists():
        return {}

    with MANIFEST_PATH.open("r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def save_manifest(manifest: dict) -> None:
    """Save the ingestion manifest."""

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with MANIFEST_PATH.open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=4)


def validate_incoming_rows(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an incoming batch into valid and rejected rows."""

    missing_columns = sorted(
        set(EXPECTED_COLUMNS) - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Incoming file is missing required columns: "
            f"{missing_columns}"
        )

    data = dataframe[EXPECTED_COLUMNS].copy()

    customer_ids = data["customerID"].astype("string").str.strip()
    tenure = pd.to_numeric(data["tenure"], errors="coerce")
    monthly_charges = pd.to_numeric(
        data["MonthlyCharges"],
        errors="coerce",
    )

    total_charges_text = (
        data["TotalCharges"].astype("string").str.strip()
    )
    total_charges_numeric = pd.to_numeric(
        total_charges_text,
        errors="coerce",
    )

    # Blank TotalCharges is accepted because the shared feature module
    # handles this known condition for zero-tenure customers.
    valid_total_charges = (
        total_charges_text.eq("")
        | total_charges_numeric.ge(0)
    )

    valid_mask = (
        customer_ids.notna()
        & customer_ids.ne("")
        & tenure.notna()
        & tenure.ge(0)
        & monthly_charges.notna()
        & monthly_charges.ge(0)
        & valid_total_charges
        & data["Churn"].isin(["Yes", "No"])
    )

    valid_rows = data.loc[valid_mask].copy()
    rejected_rows = data.loc[~valid_mask].copy()

    # Keep only the most recent occurrence if a customer appears more
    # than once within the same batch.
    valid_rows = valid_rows.drop_duplicates(
        subset=["customerID"],
        keep="last",
    )

    return valid_rows, rejected_rows


def load_existing_training_data() -> pd.DataFrame:
    """Load the current training table if it exists."""

    if not TRAINING_DATA_PATH.exists():
        return pd.DataFrame(
            columns=EXPECTED_COLUMNS
            + ["ingestion_timestamp", "source_file"]
        )

    return pd.read_csv(TRAINING_DATA_PATH)


def merge_into_training_data(
    valid_rows: pd.DataFrame,
    source_file: str,
    ingestion_timestamp: str,
) -> dict:
    """Insert new customers and update existing customers."""

    existing_data = load_existing_training_data()

    existing_ids = set(
        existing_data["customerID"]
        .dropna()
        .astype(str)
    )

    incoming_ids = set(
        valid_rows["customerID"]
        .dropna()
        .astype(str)
    )

    updated_rows = len(existing_ids.intersection(incoming_ids))
    inserted_rows = len(incoming_ids - existing_ids)

    valid_rows = valid_rows.copy()
    valid_rows["ingestion_timestamp"] = ingestion_timestamp
    valid_rows["source_file"] = source_file

    if existing_data.empty:
        combined_data = valid_rows.copy()
    else:
        combined_data = pd.concat(
            [existing_data, valid_rows],
            ignore_index=True,
        )

    # This is an upsert operation: the latest customer record wins.
    combined_data = combined_data.drop_duplicates(
        subset=["customerID"],
        keep="last",
    )

    TRAINING_DATA_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined_data.to_csv(
        TRAINING_DATA_PATH,
        index=False,
    )

    return {
        "inserted_rows": inserted_rows,
        "updated_rows": updated_rows,
        "training_table_rows": int(len(combined_data)),
    }


def save_rejected_rows(
    rejected_rows: pd.DataFrame,
    source_path: Path,
) -> Path | None:
    """Save invalid rows for investigation."""

    if rejected_rows.empty:
        return None

    REJECTED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    rejected_path = (
        REJECTED_DIRECTORY
        / f"{source_path.stem}_rejected.csv"
    )

    rejected_rows.to_csv(
        rejected_path,
        index=False,
    )

    return rejected_path


def ingest_file(
    file_path: Path,
    manifest: dict,
) -> dict | None:
    """Validate and ingest one incoming CSV file."""

    file_hash = calculate_file_hash(file_path)

    if file_hash in manifest:
        logger.info(
            "Skipping previously ingested file: %s",
            file_path.name,
        )
        return None

    ingestion_timestamp = datetime.now(
        timezone.utc
    ).isoformat()

    logger.info("Starting ingestion: %s", file_path.name)

    incoming_data = pd.read_csv(file_path)

    rows_read = int(len(incoming_data))

    valid_rows, rejected_rows = validate_incoming_rows(
        incoming_data
    )

    rejected_path = save_rejected_rows(
        rejected_rows,
        file_path,
    )

    merge_result = merge_into_training_data(
        valid_rows=valid_rows,
        source_file=file_path.name,
        ingestion_timestamp=ingestion_timestamp,
    )

    result = {
        "source_file": file_path.name,
        "file_hash": file_hash,
        "ingestion_timestamp": ingestion_timestamp,
        "rows_read": rows_read,
        "valid_rows": int(len(valid_rows)),
        "rejected_rows": int(len(rejected_rows)),
        "inserted_rows": merge_result["inserted_rows"],
        "updated_rows": merge_result["updated_rows"],
        "training_table_rows": merge_result[
            "training_table_rows"
        ],
    }

    manifest[file_hash] = result
    save_manifest(manifest)

    logger.info("Rows read: %s", result["rows_read"])
    logger.info("Valid rows: %s", result["valid_rows"])
    logger.info("Rejected rows: %s", result["rejected_rows"])
    logger.info("Inserted rows: %s", result["inserted_rows"])
    logger.info("Updated rows: %s", result["updated_rows"])
    logger.info(
        "Training table rows: %s",
        result["training_table_rows"],
    )

    if rejected_path:
        logger.warning(
            "Rejected rows saved to %s",
            rejected_path,
        )

    logger.info(
        "Completed ingestion: %s",
        file_path.name,
    )

    return result


def run_ingestion() -> list[dict]:
    """Ingest all new CSV files from the incoming directory."""

    INCOMING_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    incoming_files = sorted(
        INCOMING_DIRECTORY.glob("*.csv")
    )

    if not incoming_files:
        logger.warning(
            "No incoming CSV files found in %s",
            INCOMING_DIRECTORY,
        )
        return []

    manifest = load_manifest()
    results = []

    for file_path in incoming_files:
        try:
            result = ingest_file(
                file_path=file_path,
                manifest=manifest,
            )

            if result is not None:
                results.append(result)

        except Exception:
            logger.exception(
                "Ingestion failed for %s",
                file_path.name,
            )

    if not results:
        logger.info("No new files required ingestion")
    else:
        logger.info(
            "Successfully ingested %s new file(s)",
            len(results),
        )

    return results


def main() -> None:
    """Run the batch ingestion pipeline."""

    run_ingestion()


if __name__ == "__main__":
    main()
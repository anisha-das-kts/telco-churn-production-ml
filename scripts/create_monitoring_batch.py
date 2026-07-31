"""Create a controlled recent batch for monitoring demonstration."""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "monitoring"
    / "recent_customer_batch.csv"
)


def main() -> None:
    """Create a reproducible batch with controlled drift."""

    source_data = pd.read_csv(SOURCE_PATH)

    recent_batch = source_data.sample(
        n=500,
        random_state=42,
    ).copy()

    # Controlled drift: recent monthly prices are 50% higher.
    recent_batch["MonthlyCharges"] = (
        recent_batch["MonthlyCharges"] * 1.50
    )

    # Controlled quality issue: 8% missing MonthlyCharges.
    missing_indices = recent_batch.sample(
        frac=0.08,
        random_state=42,
    ).index

    recent_batch.loc[
        missing_indices,
        "MonthlyCharges",
    ] = np.nan

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    recent_batch.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    missing_rate = (
        recent_batch["MonthlyCharges"].isna().mean()
        * 100
    )

    print("CONTROLLED MONITORING BATCH CREATED")
    print("=" * 45)
    print(f"Source rows: {len(source_data)}")
    print(f"Recent batch rows: {len(recent_batch)}")
    print("Controlled change: MonthlyCharges increased by 50%")
    print(
        f"MonthlyCharges missing rate: "
        f"{missing_rate:.2f}%"
    )
    print(f"Saved to: {OUTPUT_PATH}")
    print("=" * 45)
    print(
        "This is synthetic demonstration data, "
        "not genuine recent production data."
    )


if __name__ == "__main__":
    main()
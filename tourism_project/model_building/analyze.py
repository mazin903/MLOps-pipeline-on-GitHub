"""Generate business-facing summary tables for the tourism project."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tourism_project.config import PROCESSED_DIR, REPORT_DIR, TARGET
from tourism_project.model_building.prep import clean_data, load_raw_data


def conversion_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Return segment-level conversion counts and rates."""
    grouped = (
        df.groupby(column, dropna=False)[TARGET]
        .agg(customers="count", buyers="sum", conversion_rate="mean")
        .reset_index()
        .sort_values(["conversion_rate", "customers"], ascending=[False, False])
    )
    grouped["conversion_rate"] = grouped["conversion_rate"].round(4)
    return grouped


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_path = PROCESSED_DIR / "cleaned_tourism.csv"
    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)
    else:
        df = clean_data(load_raw_data())

    overall_conversion = float(df[TARGET].mean())
    total_customers = int(len(df))
    buyers = int(df[TARGET].sum())

    segment_files = {}
    for column in [
        "ProductPitched",
        "Designation",
        "Passport",
        "CityTier",
        "MaritalStatus",
        "NumberOfFollowups",
        "TypeofContact",
    ]:
        table = conversion_by(df, column)
        output_path = REPORT_DIR / f"conversion_by_{column}.csv"
        table.to_csv(output_path, index=False)
        segment_files[column] = str(output_path.relative_to(REPO_ROOT))

    summary = {
        "total_customers": total_customers,
        "buyers": buyers,
        "non_buyers": total_customers - buyers,
        "overall_conversion_rate": round(overall_conversion, 4),
        "class_balance": {
            "buyers_pct": round(overall_conversion, 4),
            "non_buyers_pct": round(1 - overall_conversion, 4),
        },
        "segment_files": segment_files,
        "business_takeaways": [
            "Passport ownership is a strong signal of near-term travel readiness.",
            "Basic package prospects convert at a materially higher rate than premium package prospects.",
            "Single customers and executive-level designations show attractive conversion patterns.",
            "The model should be used to prioritize sales outreach, not to replace human sales judgment.",
        ],
    }

    summary_path = REPORT_DIR / "business_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


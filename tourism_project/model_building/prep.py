"""Clean, split, save, and upload tourism train/test datasets."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tourism_project.config import (
    CATEGORICAL_FEATURES,
    DROP_COLUMNS,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    RAW_DATA_PATH,
    TARGET,
    get_env,
    hf_dataset_repo,
)


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset from Hugging Face when authenticated, else local CSV."""
    token = os.getenv("HF_TOKEN")
    try:
        repo_id = hf_dataset_repo()
    except RuntimeError:
        repo_id = None

    if token and repo_id:
        path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename="tourism.csv",
            token=token,
        )
        print(f"Loaded raw dataset from Hugging Face dataset repo: {repo_id}")
        return pd.read_csv(path)

    print("HF_TOKEN/HF repo not set. Loading raw dataset locally for development.")
    return pd.read_csv(RAW_DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply deterministic cleaning used by both training and analysis."""
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    generated_index_cols = [col for col in df.columns if col.lower().startswith("unnamed")]
    df = df.drop(columns=[col for col in DROP_COLUMNS + generated_index_cols if col in df.columns])
    df = df.drop_duplicates()

    text_columns = [col for col in CATEGORICAL_FEATURES if col in df.columns]
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()

    replacements = {
        "Gender": {"Fe Male": "Female"},
        "TypeofContact": {"Self Inquiry": "Self Enquiry"},
        "Occupation": {"Free Lancer": "Freelancer"},
        "MaritalStatus": {"Unmarried": "Single"},
    }
    for col, mapping in replacements.items():
        if col in df.columns:
            df[col] = df[col].replace(mapping)

    required_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    for col in NUMERIC_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[TARGET])
    df[TARGET] = df[TARGET].astype(int)

    for col in NUMERIC_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        mode = df[col].mode(dropna=True)
        fallback = mode.iloc[0] if not mode.empty else "Unknown"
        df[col] = df[col].fillna(fallback).astype(str)

    return df[required_columns]


def upload_processed_files(paths: list[Path]) -> None:
    """Upload processed train/test files back to the Hugging Face dataset repo."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set. Skipping Hugging Face upload for local development.")
        return

    repo_id = hf_dataset_repo()
    api = HfApi(token=token)
    for path in paths:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=f"processed/{path.name}",
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=f"Upload processed split {path.name}",
        )
        print(f"Uploaded processed/{path.name} to {repo_id}")


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)

    X = cleaned_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = cleaned_df[TARGET]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    outputs = {
        "cleaned_tourism.csv": cleaned_df,
        "Xtrain.csv": Xtrain,
        "Xtest.csv": Xtest,
        "ytrain.csv": ytrain.to_frame(name=TARGET),
        "ytest.csv": ytest.to_frame(name=TARGET),
    }

    saved_paths: list[Path] = []
    for filename, frame in outputs.items():
        path = PROCESSED_DIR / filename
        frame.to_csv(path, index=False)
        saved_paths.append(path)
        print(f"Saved {path} with shape {frame.shape}")

    upload_processed_files(saved_paths)


if __name__ == "__main__":
    main()


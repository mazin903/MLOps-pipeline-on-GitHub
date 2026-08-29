"""Shared configuration for the tourism package prediction pipeline."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "tourism.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

TARGET = "ProdTaken"
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "tourism_xgboost_pipeline.joblib")
STATIC_MODEL_FILENAME = os.getenv("STATIC_MODEL_FILENAME", "static_model.json")

NUMERIC_FEATURES = [
    "Age",
    "CityTier",
    "DurationOfPitch",
    "NumberOfPersonVisiting",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "NumberOfTrips",
    "Passport",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact",
    "Occupation",
    "Gender",
    "ProductPitched",
    "MaritalStatus",
    "Designation",
]

DROP_COLUMNS = ["CustomerID", "H1", "Unnamed: 0", ""]


def get_env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Read an environment variable with a clear error for required values."""
    value = os.getenv(name, default)
    if required and not value:
        raise RuntimeError(
            f"Missing required environment variable {name}. "
            "Set it locally or add it as a GitHub Actions secret."
        )
    return value


def hf_dataset_repo() -> str:
    """Return the Hugging Face dataset repository id."""
    explicit = os.getenv("HF_DATASET_REPO")
    if explicit:
        return explicit
    username = get_env("HF_USERNAME", required=True)
    return f"{username}/tourism"


def hf_model_repo() -> str:
    """Return the Hugging Face model repository id."""
    explicit = os.getenv("HF_MODEL_REPO")
    if explicit:
        return explicit
    username = get_env("HF_USERNAME", required=True)
    return f"{username}/tourism-package-model"


def hf_space_repo() -> str:
    """Return the Hugging Face Space repository id."""
    explicit = os.getenv("HF_SPACE_REPO")
    if explicit:
        return explicit
    username = get_env("HF_USERNAME", required=True)
    return f"{username}/tourism-package-prediction"

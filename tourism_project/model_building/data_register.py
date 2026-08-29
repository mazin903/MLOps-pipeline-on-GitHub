"""Register the raw tourism dataset on Hugging Face Hub."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tourism_project.config import DATA_DIR, RAW_DATA_PATH, get_env, hf_dataset_repo


def ensure_dataset_repo(api: HfApi, repo_id: str) -> None:
    """Create the dataset repository when it is not already present."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset")
        print(f"Dataset repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Dataset repository '{repo_id}' not found. Creating it now.")
        create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=False,
            token=api.token,
            exist_ok=True,
        )


def main() -> None:
    token = get_env("HF_TOKEN", required=True)
    repo_id = hf_dataset_repo()

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    api = HfApi(token=token)
    ensure_dataset_repo(api, repo_id)

    api.upload_folder(
        folder_path=str(DATA_DIR),
        repo_id=repo_id,
        repo_type="dataset",
        path_in_repo="",
        allow_patterns=["tourism.csv", "README.md"],
        commit_message="Register raw tourism dataset",
    )

    print(f"Registered {RAW_DATA_PATH.name} in https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()

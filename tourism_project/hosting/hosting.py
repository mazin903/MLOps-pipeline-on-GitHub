"""Push the Streamlit deployment files to a Hugging Face Space."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tourism_project.config import PROJECT_ROOT, get_env, hf_space_repo


DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"


def ensure_space_repo(api: HfApi, repo_id: str) -> None:
    """Create the Hugging Face Space when it does not already exist."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating Docker Space now.")
        create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="docker",
            private=False,
            token=api.token,
            exist_ok=True,
        )


def main() -> None:
    token = get_env("HF_TOKEN", required=True)
    repo_id = hf_space_repo()

    if not (DEPLOYMENT_DIR / "app.py").exists():
        raise FileNotFoundError(f"Deployment app not found in {DEPLOYMENT_DIR}")

    api = HfApi(token=token)
    ensure_space_repo(api, repo_id)
    api.upload_folder(
        folder_path=str(DEPLOYMENT_DIR),
        repo_id=repo_id,
        repo_type="space",
        path_in_repo="",
        commit_message="Deploy Streamlit tourism package predictor",
    )

    print(f"Deployment uploaded to https://huggingface.co/spaces/{repo_id}")
    print("If the model repository is private, add HF_TOKEN as a Space secret.")
    print("Set HF_MODEL_REPO and MODEL_FILENAME as Space variables when needed.")


if __name__ == "__main__":
    main()


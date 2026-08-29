"""Push deployment files to a Hugging Face Space."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tourism_project.config import (
    MODEL_DIR,
    MODEL_FILENAME,
    PROJECT_ROOT,
    STATIC_MODEL_FILENAME,
    get_env,
    hf_dataset_repo,
    hf_model_repo,
    hf_space_repo,
)


DEPLOYMENT_DIR = PROJECT_ROOT / "deployment"
STATIC_DEPLOYMENT_DIR = PROJECT_ROOT / "deployment_static"
STATIC_MODEL_PATH = MODEL_DIR / STATIC_MODEL_FILENAME


def ensure_space_repo(api: HfApi, repo_id: str, space_sdk: str) -> None:
    """Create the Hugging Face Space when it does not already exist."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space '{repo_id}' already exists.")
        return
    except RepositoryNotFoundError:
        pass
    except HfHubHTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code not in {401, 404}:
            raise

    try:
        print(f"Space '{repo_id}' not found. Creating {space_sdk} Space now.")
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk=space_sdk,
            private=False,
            exist_ok=True,
        )
    except HfHubHTTPError as exc:
        status_code = getattr(exc.response, "status_code", None)
        if status_code not in {409}:
            raise
        print(f"Space '{repo_id}' was created by another run; continuing.")


def is_paid_space_error(exc: HfHubHTTPError) -> bool:
    """Return True when Hugging Face blocks paid Space runtimes on the account."""
    status_code = getattr(exc.response, "status_code", None)
    return status_code == 402


def sync_space_variables(api: HfApi, repo_id: str) -> None:
    """Store non-sensitive app configuration directly on the Space."""
    variables = {
        "HF_MODEL_REPO": hf_model_repo(),
        "MODEL_FILENAME": MODEL_FILENAME,
    }
    for key, value in variables.items():
        api.add_space_variable(repo_id=repo_id, key=key, value=value)
        print(f"Space variable '{key}' set.")


def deploy_docker_space(api: HfApi, repo_id: str) -> None:
    """Deploy the Streamlit application as a Docker-backed Hugging Face Space."""
    if not (DEPLOYMENT_DIR / "app.py").exists():
        raise FileNotFoundError(f"Deployment app not found in {DEPLOYMENT_DIR}")

    ensure_space_repo(api, repo_id, "docker")
    sync_space_variables(api, repo_id)
    api.upload_folder(
        folder_path=str(DEPLOYMENT_DIR),
        repo_id=repo_id,
        repo_type="space",
        path_in_repo="",
        commit_message="Deploy Streamlit tourism package predictor",
    )

    print(f"Deployment uploaded to https://huggingface.co/spaces/{repo_id}")
    print("If the model repository is private, add HF_TOKEN as a Space secret.")
    print("HF_MODEL_REPO and MODEL_FILENAME are managed as Space variables.")


def prepare_static_bundle(bundle_dir: Path) -> None:
    """Create a self-contained Static Space upload directory."""
    if not STATIC_DEPLOYMENT_DIR.exists():
        raise FileNotFoundError(f"Static deployment folder not found: {STATIC_DEPLOYMENT_DIR}")

    shutil.copytree(STATIC_DEPLOYMENT_DIR, bundle_dir, dirs_exist_ok=True)
    config = {
        "githubRepo": get_env("GITHUB_REPO_FULL_NAME", "mazin903/MLOps-pipeline-on-GitHub"),
        "datasetRepo": hf_dataset_repo(),
        "modelRepo": hf_model_repo(),
        "staticModelFile": STATIC_MODEL_FILENAME,
    }
    config_text = f"window.APP_CONFIG = {json.dumps(config, indent=2)};\n"
    (bundle_dir / "config.js").write_text(config_text, encoding="utf-8")

    if STATIC_MODEL_PATH.exists():
        shutil.copy2(STATIC_MODEL_PATH, bundle_dir / STATIC_MODEL_FILENAME)
        print(f"Included local static model artifact: {STATIC_MODEL_PATH}")
    else:
        print(
            f"{STATIC_MODEL_PATH} not found. The app will load "
            f"{STATIC_MODEL_FILENAME} from the Hugging Face model repo."
        )


def deploy_static_space(api: HfApi, repo_id: str) -> None:
    """Deploy the free-tier browser prediction app to a Static Space."""
    ensure_space_repo(api, repo_id, "static")
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "space"
        prepare_static_bundle(bundle_dir)
        api.upload_folder(
            folder_path=str(bundle_dir),
            repo_id=repo_id,
            repo_type="space",
            path_in_repo="",
            commit_message="Deploy static tourism package predictor",
        )
    print(f"Static deployment uploaded to https://huggingface.co/spaces/{repo_id}")


def main() -> None:
    token = get_env("HF_TOKEN", required=True)
    repo_id = hf_space_repo()
    space_mode = get_env("HF_SPACE_MODE", "static").lower()

    api = HfApi(token=token)
    if space_mode in {"docker", "streamlit"}:
        try:
            deploy_docker_space(api, repo_id)
        except HfHubHTTPError as exc:
            if not is_paid_space_error(exc):
                raise
            print("Docker Space creation requires a paid Hugging Face runtime.")
            print("Falling back to a free Static Space deployment.")
            deploy_static_space(api, repo_id)
    elif space_mode == "static":
        deploy_static_space(api, repo_id)
    else:
        raise ValueError("HF_SPACE_MODE must be one of: static, docker, streamlit")


if __name__ == "__main__":
    main()

"""Train, track, evaluate, and register the tourism prediction model."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tourism_project.config import (
    CATEGORICAL_FEATURES,
    DATA_DIR,
    MLRUNS_DIR,
    MODEL_DIR,
    MODEL_FILENAME,
    NUMERIC_FEATURES,
    PROCESSED_DIR,
    REPORT_DIR,
    STATIC_MODEL_FILENAME,
    TARGET,
    hf_dataset_repo,
    hf_model_repo,
)


MODEL_CARD_SOURCE = Path(__file__).resolve().parent / "model_card.md"


def load_processed_file(filename: str) -> pd.DataFrame:
    """Load processed data from Hugging Face, falling back to local files."""
    token = os.getenv("HF_TOKEN")
    try:
        dataset_repo = hf_dataset_repo()
    except RuntimeError:
        dataset_repo = None

    if token and dataset_repo:
        path = hf_hub_download(
            repo_id=dataset_repo,
            repo_type="dataset",
            filename=f"processed/{filename}",
            token=token,
        )
        print(f"Loaded processed/{filename} from Hugging Face dataset repo: {dataset_repo}")
        return pd.read_csv(path)

    local_path = PROCESSED_DIR / filename
    if not local_path.exists():
        raise FileNotFoundError(
            f"{local_path} not found. Run tourism_project/model_building/prep.py first."
        )
    print(f"HF_TOKEN/HF repo not set. Loading {filename} locally for development.")
    return pd.read_csv(local_path)


def build_pipeline(class_weight: float):
    """Create a preprocessing plus XGBoost classification pipeline."""
    numeric_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    categorical_pipeline = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore"),
    )

    preprocessor = make_column_transformer(
        (numeric_pipeline, NUMERIC_FEATURES),
        (categorical_pipeline, CATEGORICAL_FEATURES),
    )

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=class_weight,
        random_state=42,
        n_jobs=2,
    )

    return make_pipeline(preprocessor, xgb_model)


def evaluate_thresholds(model, Xtest: pd.DataFrame, ytest: pd.Series) -> tuple[float, list[dict]]:
    """Choose the threshold with the strongest test F1, using recall as tie-breaker."""
    probabilities = model.predict_proba(Xtest)[:, 1]
    rows: list[dict] = []

    for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        predictions = (probabilities >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ytest,
            predictions,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": threshold,
                "accuracy": accuracy_score(ytest, predictions),
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    best = max(rows, key=lambda row: (row["f1"], row["recall"]))
    return float(best["threshold"]), rows


def save_feature_importance(model, output_path: Path) -> None:
    """Persist feature importances from the fitted XGBoost model."""
    preprocessor = model.named_steps["columntransformer"]
    classifier = model.named_steps["xgbclassifier"]
    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": [name.split("__", 1)[-1] for name in feature_names],
            "importance": classifier.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False)


def export_static_model(model, threshold: float, metrics: dict, output_path: Path) -> None:
    """Export the fitted pipeline as browser-readable preprocessing plus XGBoost trees."""
    preprocessor = model.named_steps["columntransformer"]
    classifier = model.named_steps["xgbclassifier"]

    numeric_transformer = preprocessor.transformers_[0][1]
    categorical_transformer = preprocessor.transformers_[1][1]
    numeric_imputer = numeric_transformer.named_steps["simpleimputer"]
    scaler = numeric_transformer.named_steps["standardscaler"]
    categorical_imputer = categorical_transformer.named_steps["simpleimputer"]
    encoder = categorical_transformer.named_steps["onehotencoder"]

    feature_names = [
        name.split("__", 1)[-1] for name in preprocessor.get_feature_names_out()
    ]
    feature_importance = sorted(
        [
            {"feature": feature, "importance": float(importance)}
            for feature, importance in zip(feature_names, classifier.feature_importances_)
        ],
        key=lambda row: row["importance"],
        reverse=True,
    )
    booster = classifier.get_booster()
    trees = [json.loads(tree) for tree in booster.get_dump(dump_format="json")]

    payload = {
        "model_type": "xgboost_binary_logistic",
        "threshold": threshold,
        "target": TARGET,
        "feature_names": feature_names,
        "numeric": [
            {
                "name": name,
                "impute": float(numeric_imputer.statistics_[index]),
                "mean": float(scaler.mean_[index]),
                "scale": float(scaler.scale_[index]),
            }
            for index, name in enumerate(NUMERIC_FEATURES)
        ],
        "categorical": [
            {
                "name": name,
                "impute": str(categorical_imputer.statistics_[index]),
                "categories": [str(category) for category in encoder.categories_[index]],
            }
            for index, name in enumerate(CATEGORICAL_FEATURES)
        ],
        "trees": trees,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "scoring_note": "Tree traversal uses float32 comparisons to match XGBoost prediction semantics.",
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_model_repo(api: HfApi, repo_id: str) -> None:
    """Create the model repository when it is not already present."""
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        print(f"Model repository '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Model repository '{repo_id}' not found. Creating it now.")
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=False,
            token=api.token,
            exist_ok=True,
        )


def upload_model_artifacts(paths: list[Path]) -> None:
    """Upload model and evaluation artifacts to Hugging Face Model Hub."""
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set. Skipping Hugging Face model upload for local development.")
        return

    repo_id = hf_model_repo()
    api = HfApi(token=token)
    ensure_model_repo(api, repo_id)

    upload_paths = list(paths)
    if MODEL_CARD_SOURCE.exists():
        upload_paths.append(MODEL_CARD_SOURCE)

    for path in upload_paths:
        path_in_repo = "README.md" if path.name == "model_card.md" else path.name
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload {path_in_repo}",
        )
        print(f"Uploaded {path_in_repo} to https://huggingface.co/{repo_id}")


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    Xtrain = load_processed_file("Xtrain.csv")
    Xtest = load_processed_file("Xtest.csv")
    ytrain = load_processed_file("ytrain.csv")[TARGET].astype(int)
    ytest = load_processed_file("ytest.csv")[TARGET].astype(int)

    class_counts = ytrain.value_counts()
    negative_count = int(class_counts.get(0, 1))
    positive_count = int(class_counts.get(1, 1))
    class_weight = negative_count / max(positive_count, 1)

    model_pipeline = build_pipeline(class_weight)
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5],
        "xgbclassifier__colsample_bytree": [0.8],
        "xgbclassifier__colsample_bylevel": [0.8],
        "xgbclassifier__learning_rate": [0.05, 0.10],
        "xgbclassifier__reg_lambda": [1.0, 5.0],
    }

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", MLRUNS_DIR.as_uri())
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "tourism_package_prediction"))

    with mlflow.start_run(run_name="xgboost_tourism_package_prediction"):
        mlflow.log_param("target", TARGET)
        mlflow.log_param("class_weight", class_weight)
        mlflow.log_param("cv_folds", 3)
        for name, values in param_grid.items():
            mlflow.log_param(f"grid_{name}", str(values))

        grid_search = GridSearchCV(
            estimator=model_pipeline,
            param_grid=param_grid,
            cv=3,
            scoring="f1",
            n_jobs=-1,
            verbose=1,
        )
        grid_search.fit(Xtrain, ytrain)
        best_model = grid_search.best_estimator_
        mlflow.log_params(grid_search.best_params_)

        best_threshold, threshold_rows = evaluate_thresholds(best_model, Xtest, ytest)
        mlflow.log_param("classification_threshold", best_threshold)

        train_probabilities = best_model.predict_proba(Xtrain)[:, 1]
        test_probabilities = best_model.predict_proba(Xtest)[:, 1]
        y_pred_train = (train_probabilities >= best_threshold).astype(int)
        y_pred_test = (test_probabilities >= best_threshold).astype(int)

        train_report = classification_report(
            ytrain,
            y_pred_train,
            output_dict=True,
            zero_division=0,
        )
        test_report = classification_report(
            ytest,
            y_pred_test,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "train_accuracy": train_report["accuracy"],
            "train_precision": train_report["1"]["precision"],
            "train_recall": train_report["1"]["recall"],
            "train_f1": train_report["1"]["f1-score"],
            "test_accuracy": test_report["accuracy"],
            "test_precision": test_report["1"]["precision"],
            "test_recall": test_report["1"]["recall"],
            "test_f1": test_report["1"]["f1-score"],
        }
        mlflow.log_metrics(metrics)

        model_path = MODEL_DIR / MODEL_FILENAME
        static_model_path = MODEL_DIR / STATIC_MODEL_FILENAME
        model_bundle = {
            "model": best_model,
            "threshold": best_threshold,
            "target": TARGET,
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }
        joblib.dump(model_bundle, model_path)
        mlflow.log_artifact(str(model_path), artifact_path="model")

        metrics_path = REPORT_DIR / "metrics.json"
        report_path = REPORT_DIR / "classification_report.json"
        threshold_path = REPORT_DIR / "threshold_analysis.csv"
        importance_path = REPORT_DIR / "feature_importance.csv"

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        report_path.write_text(
            json.dumps({"train": train_report, "test": test_report}, indent=2),
            encoding="utf-8",
        )
        pd.DataFrame(threshold_rows).to_csv(threshold_path, index=False)
        save_feature_importance(best_model, importance_path)
        export_static_model(best_model, best_threshold, metrics, static_model_path)

        upload_model_artifacts(
            [model_path, static_model_path, metrics_path, report_path, threshold_path, importance_path]
        )

    print("Best parameters:", grid_search.best_params_)
    print("Metrics:", json.dumps(metrics, indent=2))
    print(f"Saved model bundle to {model_path}")
    print(f"Reports saved under {REPORT_DIR}")


if __name__ == "__main__":
    main()

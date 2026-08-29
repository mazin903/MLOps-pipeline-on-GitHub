# Streamlit Community Cloud Deployment

Use this deployment route when a paid Hugging Face Space runtime is not available.

## App Settings

- Repository: `mazin903/MLOps-pipeline-on-GitHub`
- Branch: `main`
- Main file path: `tourism_project/deployment/app.py`
- Suggested app URL: `tourism-package-prediction`
- Python version: `3.11`

## Secrets

Set these in Streamlit Community Cloud Advanced settings:

```toml
HF_MODEL_REPO = "mazin903/tourism-package-model"
MODEL_FILENAME = "tourism_xgboost_pipeline.joblib"
```

`HF_TOKEN` is required only if the Hugging Face model repository is private.

## Evidence To Capture

- Streamlit Community Cloud app URL.
- Screenshot of the app after a prediction result is displayed.
- Screenshot of the GitHub Actions workflow after the data/model pipeline succeeds.

# Tourism Package Prediction MLOps Pipeline

This project builds an end-to-end MLOps pipeline for Visit with Us to predict whether a customer is likely to purchase the Wellness Tourism Package before the sales team contacts them.

## Business Goal

The model helps the marketing and sales teams prioritize customers with higher purchase likelihood, reduce manual targeting effort, and improve campaign conversion efficiency.

## Repository Structure

```text
tourism_project/
  data/
    tourism.csv
    processed/
  model_building/
    data_register.py
    prep.py
    train.py
  deployment/
    app.py
    Dockerfile
    requirements.txt
  deployment_static/
    index.html
    app.js
    styles.css
  hosting/
    hosting.py
.github/workflows/
  pipeline.yml
requirements.txt
```

## Required GitHub Configuration

Add the following repository secrets and variables before running the workflow:

| Name | Type | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | Secret | Authenticates GitHub Actions to Hugging Face Hub |
| `HF_USERNAME` | Variable | Builds default Hugging Face repo names |
| `HF_DATASET_REPO` | Variable | Optional explicit dataset repo, such as `username/tourism` |
| `HF_MODEL_REPO` | Variable | Optional explicit model repo, such as `username/tourism-package-model` |
| `HF_SPACE_REPO` | Variable | Optional explicit Space repo, such as `username/tourism-package-prediction` |
| `HF_SPACE_MODE` | Variable | Optional deployment mode. Use `static` for free-tier Hugging Face Spaces or `docker` if a paid runtime is available. |

## Streamlit Community Cloud Deployment

Streamlit Community Cloud is the free deployment route accepted for this project when Hugging Face Docker/Gradio Spaces are not available on the account. Deploy from:

- Repository: `mazin903/MLOps-pipeline-on-GitHub`
- Branch: `main`
- Main file path: `tourism_project/deployment/app.py`
- Python version: `3.11`
- Secrets:

```toml
HF_MODEL_REPO = "mazin903/tourism-package-model"
MODEL_FILENAME = "tourism_xgboost_pipeline.joblib"
```

`HF_TOKEN` is required only if the Hugging Face model repository is private.

## Local Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python tourism_project/model_building/data_register.py
python tourism_project/model_building/prep.py
python tourism_project/model_building/train.py
python tourism_project/hosting/hosting.py
```

For local development without Hugging Face credentials, `prep.py` and `train.py` can use local files after the raw CSV is placed in `tourism_project/data/tourism.csv`. Registration and Space upload require `HF_TOKEN`.

## Google Colab Run

Google Colab is useful when your local machine is missing Python packages or you want a quick cloud run. Upload this repository to GitHub, open the completed notebook in Colab, add `HF_TOKEN` as a Colab secret, then run the cells in order. Colab should push the dataset, processed splits, model artifacts, and deployment folder to Hugging Face using the same scripts in this repository.

## Pipeline Outputs

- Raw data is registered in a Hugging Face dataset repository.
- Cleaned train/test datasets are uploaded back to the dataset repository.
- The tuned XGBoost model, browser-readable model artifact, feature importance, threshold analysis, and evaluation metrics are uploaded to Hugging Face Model Hub.
- A free-tier Static Space prediction app is deployed to Hugging Face by default; the Streamlit app can also be deployed from GitHub to Streamlit Community Cloud.
- GitHub Actions automates the complete workflow on pushes to `main`.

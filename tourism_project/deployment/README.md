---
title: Tourism Package Prediction
emoji: ✈️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tourism Package Prediction

This Streamlit app downloads the registered tourism prediction model from Hugging Face Model Hub and predicts whether a customer should be prioritized for the Wellness Tourism Package sales campaign.

The Streamlit interface runs inside a Docker-backed Hugging Face Space on port 7860 for reproducible deployment evidence.

## Space Variables

Set these in the Hugging Face Space settings:

- `HF_MODEL_REPO`: your model repository, for example `username/tourism-package-model`
- `MODEL_FILENAME`: `tourism_xgboost_pipeline.joblib`
- `HF_TOKEN`: required only when the model repository is private

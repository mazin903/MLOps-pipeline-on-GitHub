"""Streamlit app for tourism package purchase prediction."""

from __future__ import annotations

import os

import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download


def secret_or_env(name: str, default: str | None = None) -> str | None:
    """Read Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name, default)


st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️", layout="centered")


@st.cache_resource(show_spinner="Loading model from Hugging Face...")
def load_model_bundle():
    repo_id = secret_or_env("HF_MODEL_REPO")
    if not repo_id and secret_or_env("HF_USERNAME"):
        repo_id = f"{secret_or_env('HF_USERNAME')}/tourism-package-model"
    if not repo_id:
        st.error("Set HF_MODEL_REPO or HF_USERNAME to load the Hugging Face model.")
        st.stop()

    filename = secret_or_env("MODEL_FILENAME", "tourism_xgboost_pipeline.joblib")
    token = secret_or_env("HF_TOKEN")
    model_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="model",
        filename=filename,
        token=token,
    )
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict) and "model" in artifact:
        return artifact
    return {"model": artifact, "threshold": 0.45}


bundle = load_model_bundle()
model = bundle["model"]
classification_threshold = float(bundle.get("threshold", 0.45))

st.title("Tourism Package Prediction")
st.write("Estimate whether a customer is likely to purchase the Wellness Tourism Package.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 70, 35)
        type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
        city_tier = st.selectbox("City tier", [1, 2, 3], index=0)
        duration_of_pitch = st.slider("Duration of pitch (minutes)", 5, 130, 15)
        occupation = st.selectbox(
            "Occupation",
            ["Salaried", "Small Business", "Large Business", "Freelancer"],
        )
        gender = st.selectbox("Gender", ["Male", "Female"])
        visitors = st.slider("Number of people visiting", 1, 5, 3)
        followups = st.slider("Number of follow-ups", 1, 6, 4)
        product_pitched = st.selectbox(
            "Product pitched",
            ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
        )

    with col2:
        preferred_star = st.selectbox("Preferred property star", [1, 2, 3, 4, 5], index=2)
        marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
        trips = st.slider("Annual number of trips", 1, 22, 3)
        passport = st.selectbox("Has passport?", ["Yes", "No"])
        pitch_score = st.slider("Pitch satisfaction score", 1, 5, 3)
        own_car = st.selectbox("Owns a car?", ["Yes", "No"])
        children = st.slider("Number of children visiting", 0, 5, 1)
        designation = st.selectbox(
            "Designation",
            ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
        )
        monthly_income = st.number_input("Monthly income", min_value=1000.0, value=23000.0, step=1000.0)

    submitted = st.form_submit_button("Predict purchase likelihood")

input_data = pd.DataFrame(
    [
        {
            "Age": age,
            "CityTier": city_tier,
            "DurationOfPitch": duration_of_pitch,
            "NumberOfPersonVisiting": visitors,
            "NumberOfFollowups": followups,
            "PreferredPropertyStar": preferred_star,
            "NumberOfTrips": trips,
            "Passport": 1 if passport == "Yes" else 0,
            "PitchSatisfactionScore": pitch_score,
            "OwnCar": 1 if own_car == "Yes" else 0,
            "NumberOfChildrenVisiting": children,
            "MonthlyIncome": monthly_income,
            "TypeofContact": type_of_contact,
            "Occupation": occupation,
            "Gender": gender,
            "ProductPitched": product_pitched,
            "MaritalStatus": marital_status,
            "Designation": designation,
        }
    ]
)

if submitted:
    probability = float(model.predict_proba(input_data)[0, 1])
    prediction = int(probability >= classification_threshold)

    st.metric("Purchase probability", f"{probability:.1%}")
    if prediction:
        st.success("Recommendation: prioritize this customer for a sales conversation.")
    else:
        st.info("Recommendation: place this customer in a lower-touch nurture segment.")

    st.caption(f"Decision threshold: {classification_threshold:.2f}")

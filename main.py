# Traffic Accident Severity Predictor (Streamlit App)

# imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# used in runtime
from xgboost import XGBClassifier

# Streamlit Page Setup
st.set_page_config(page_title="Traffic Accident Severity Predictor", layout="wide")

st.title("Traffic Accident Severity Predictor")
st.markdown(
    """
Welcome!  
Predict the **severity of a traffic accident** based on environmental and road conditions.

Dataset Source: [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
"""
)


# Cache Functions for loading
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join("data", "final_cleaned_accident_data.csv"))


@st.cache_resource
def load_model():
    model_path = os.path.join("models", "final_xgb_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# Load Resources
df = load_data()
xgb_model = load_model()

# User Input Form
st.subheader("Enter Conditions:")

with st.form("prediction_form"):
    distance = st.number_input("Distance (miles)", min_value=0.0, value=0.5, step=0.1)
    temperature = st.number_input(
        "Temperature (°F)", min_value=-50.0, max_value=130.0, value=70.0
    )
    humidity = st.slider("Humidity (%)", 0, 100, 50)
    visibility = st.number_input(
        "Visibility (miles)", min_value=0.0, max_value=50.0, value=10.0
    )

    submitted = st.form_submit_button("Predict Accident Severity")

# Prediction Logic
if submitted:
    # Base numeric features
    input_data = pd.DataFrame(
        {
            "Distance(mi)": [distance],
            "Temperature(F)": [temperature],
            "Humidity(%)": [humidity],
            "Visibility(mi)": [visibility],
        }
    )

    # Make prediction
    prediction = xgb_model.predict(input_data)[0]
    # value was offset in model training for XGBoost so need to +1
    predicted_severity = int(prediction) + 1
    proba = xgb_model.predict_proba(input_data)[0]

    # Results
    st.markdown("---")
    st.subheader("Prediction Result")
    st.success(
        f"Predicted Severity: **{predicted_severity}** (1 = least severe - 4 = most severe)"
    )

# Note
st.caption(
    "Note: Model trained on U.S. accident data (2016–2023). This demo uses the four strongest continuous features — "
    "Distance, Temperature, Humidity, and Visibility — for clarity and performance. "
    "The full model includes additional features such as weather, state, and time-based variables for improved accuracy."
)

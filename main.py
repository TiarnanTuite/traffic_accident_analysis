# Traffic Accident Severity Predictor (Streamlit App)

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# used at runtime
from xgboost import XGBClassifier

# Streamlit Page Setup
st.set_page_config(page_title="Traffic Accident Severity Predictor", layout="wide")

st.title("Traffic Accident Severity Predictor")
st.markdown(
    """
Welcome!  
Test for **severity of a traffic accident** 

Dataset Source: [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
"""
)

# Load Dataset
DATA_PATH = os.path.join("data", "final_cleaned_accident_data.csv")

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"Dataset loaded successfully - {len(df):,} records found.")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Display a sample of data
with st.expander("View Sample of Cleaned Dataset"):
    st.dataframe(df.head(10))

# Load Model
MODEL_PATH = os.path.join("models", "final_xgb_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        xgb_model = pickle.load(f)
    st.success("Machine learning model loaded successfully.")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# Model Sanity Check
st.subheader("Model Test Prediction")

# Example sample (same format as training)
sample_input = np.array(
    [[0.5, 65, 60, 10]]
)  # [Distance, Temperature, Humidity, Visibility]

try:
    test_prediction = xgb_model.predict(sample_input)[0]
    st.info(
        f"Model test prediction successful - **Predicted Severity: {int(test_prediction)}** (1 = least severe → 4 = most severe)"
    )
except Exception as e:
    st.warning(f"Model test prediction failed: {e}")

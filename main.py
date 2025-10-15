# Traffic Accident Severity Predictor (Streamlit App)

# imports
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# used at runtime
from xgboost import XGBClassifier

# Streamlit Page Setup
st.set_page_config(page_title="Traffic Accident Severity Predictor", layout="wide")


# Cache Functions for loading
@st.cache_data
def load_data():
    return pd.read_csv(os.path.join("data", "sample_accident_data.csv"))


@st.cache_resource
def load_model():
    model_path = os.path.join("models", "final_xgb_model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


# matching progress bar with actual loading
import time

progress_text = "Please wait! (Initializing model and loading dataset...)"
my_bar = st.progress(0, text=progress_text)

# Loading dataset
my_bar.progress(
    10, text=" Loading accident dataset...(First run might take a few seconds!)"
)
time.sleep(0.2)  # small delay before it starts
df = load_data()
for i in range(10, 40, 5):
    time.sleep(0.2)
    my_bar.progress(i, text="Loading accident dataset...")

# Loading model
my_bar.progress(50, text="Loading trained XGBoost model...")
time.sleep(0.2)
xgb_model = load_model()
for i in range(50, 90, 5):
    time.sleep(0.2)
    my_bar.progress(i, text="Loading trained XGBoost model...")

# Step 3: Finishing up
for i in range(90, 101, 2):
    time.sleep(0.1)
    my_bar.progress(i, text="Finalizing setup...")

my_bar.empty()
# success animation
st.balloons()


# navigation
tab1, tab2, tab3 = st.tabs(["Home", "Predictor", "Maps"])

# home page
with tab1:

    st.title("Traffic Accident Severity Predictor")
    st.markdown(
        """
    Welcome!  
    Predict the **severity of a traffic accident** based on environmental and road conditions.

    Dataset Source: [US Accidents (2016–2023)](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
    """
    )

    st.markdown("---")

    st.header("About This Project")
    st.write(
        """
    This project explores patterns in U.S. road accident data from 2016-2023, aiming to identify how
    environmental conditions influence the **severity** of accidents.

    Using machine learning — specifically an **XGBoost model** — it analyzes factors such as:
    - Distance of the accident segment  
    - Temperature  
    - Humidity  
    - Visibility  

    The trained model can then predict how severe an accident might be under given conditions,
    serving as a foundation for a real-world **road safety and risk awareness tool**.
    """
    )

    st.header("App Features")
    st.write(
        """
    - **Accident Severity Predictor:** Test different weather and road conditions to see how they affect
        predicted severity.  
    - **Interactive Maps:** Explore accident density and severity heatmaps across U.S. states.  
    - **Insights & Final Analysis:** Summarizes the model’s performance and real-world implications.
    """
    )

    st.header("Project Goals")
    st.write(
        """
    - Practice **data analysis and visualization** using Python.  
    - Apply **machine learning techniques** to real-world datasets.  
    - Build and deploy an **interactive Streamlit web app** to demonstrate predictive insights.  
    - Expand technical experience in data preprocessing, feature engineering, and deployment.
    """
    )

    st.markdown("---")
    st.info(
        "Use the **Predictor** tab to get started! test conditions and see how the model classifies accident severity."
    )


# predictor
with tab2:

    # User Input Form
    st.subheader("Enter Conditions:")

    with st.form("prediction_form"):
        distance = st.number_input(
            "Distance (miles)", min_value=0.0, value=0.5, step=0.1
        )
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
            f"Based on US data, Crashing with these journey conditions typically have Predicted Severity: **{predicted_severity}** (1 = least severe - 4 = most severe)"
        )

    # Note
    st.caption(
        "Note: Model trained on U.S. accident data (2016–2023). This demo uses the four strongest continuous features — "
        "Distance, Temperature, Humidity, and Visibility — for clarity and performance. "
        "The full model includes additional features such as weather, state, and time-based variables for improved accuracy."
    )

# maps
with tab3:
    st.header("Maps")

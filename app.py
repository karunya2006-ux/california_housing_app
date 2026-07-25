import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 California Housing Price Predictor")
st.write("Enter house and location details below to predict the median house value using your trained Machine Learning model.")

# 2. Load trained model & pipeline
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    pipeline = joblib.load("pipeline.pkl")
    return model, pipeline

try:
    model, pipeline = load_artifacts()
except Exception as e:
    st.error("Model artifacts (model.pkl / pipeline.pkl) not found. Run `python3 main.py` first to train the model!")
    st.stop()

# 3. User Input Form
with st.form("housing_input_form"):
    st.subheader("📋 Input Property Features")
    col1, col2, col3 = st.columns(3)

    with col1:
        longitude = st.number_input("Longitude", value=-122.23, format="%.4f")
        latitude = st.number_input("Latitude", value=37.88, format="%.4f")
        housing_median_age = st.slider("Housing Median Age (Years)", 1, 52, 25)

    with col2:
        total_rooms = st.number_input("Total Rooms", value=2000, step=50)
        total_bedrooms = st.number_input("Total Bedrooms", value=400, step=10)
        population = st.number_input("Population", value=1000, step=50)

    with col3:
        households = st.number_input("Households", value=350, step=10)
        median_income = st.number_input("Median Income (in $10,000s)", value=3.87, step=0.1, format="%.2f")
        ocean_proximity = st.selectbox(
            "Ocean Proximity",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
        )

    submit_button = st.form_submit_button("💰 Predict House Value")

# 4. Run Prediction when form is submitted
if submit_button:
    # Build dataframe exactly matching training feature names
    input_df = pd.DataFrame([{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": ocean_proximity
    }])

    # Preprocess inputs using saved pipeline & predict
    transformed_input = pipeline.transform(input_df)
    prediction = model.predict(transformed_input)[0]

    st.markdown("---")
    st.success(f"### Estimated Median House Value: **${prediction:,.2f}**")

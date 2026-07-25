import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide"
)

st.title("🏡 California Housing Price Predictor")
st.write("Enter house and location details below to predict the median house value.")

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])

@st.cache_resource
def train_and_load():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(PIPELINE_FILE)):
        with st.spinner("Training model for first-time startup... Please wait 10 seconds."):
            housing = pd.read_csv("housing.csv")
            housing['income_cat'] = pd.cut(
                housing['median_income'],
                bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
                labels=[1, 2, 3, 4, 5]
            )
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            for train_index, _ in split.split(housing, housing['income_cat']):
                housing = housing.loc[train_index].drop('income_cat', axis=1)

            housing_labels = housing['median_house_value'].copy()
            housing_features = housing.drop('median_house_value', axis=1)

            num_attribs = housing_features.drop("ocean_proximity", axis=1).columns
            cat_attribs = ["ocean_proximity"]

            pipeline = build_pipeline(num_attribs, cat_attribs)
            housing_prepared = pipeline.fit_transform(housing_features)

            model = RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42)
            model.fit(housing_prepared, housing_labels)

            joblib.dump(model, MODEL_FILE)
            joblib.dump(pipeline, PIPELINE_FILE)

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    return model, pipeline

model, pipeline = train_and_load()

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

if submit_button:
    input_df = pd.DataFrame([{
        "longitude": float(longitude),
        "latitude": float(latitude),
        "housing_median_age": float(housing_median_age),
        "total_rooms": float(total_rooms),
        "total_bedrooms": float(total_bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": float(median_income),
        "ocean_proximity": str(ocean_proximity)
    }])

    transformed_input = pipeline.transform(input_df)
    prediction = model.predict(transformed_input)[0]
    st.markdown("---")
    st.success(f"### Estimated Median House Value: **${prediction:,.2f}**")

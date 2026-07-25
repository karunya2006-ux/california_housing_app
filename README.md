# 🏡 California Housing Price Prediction App

An end-to-end Machine Learning web application built using **Scikit-Learn** and **Streamlit** to predict median California house values based on 1990 U.S. Census data.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://californiahousingapp-888m3y4ymzvrkv5tjw2ais.streamlit.app/)

🔗 **Live Web App:** [https://californiahousingapp-888m3y4ymzvrkv5tjw2ais.streamlit.app/](https://californiahousingapp-888m3y4ymzvrkv5tjw2ais.streamlit.app/)

---

## 📌 Project Overview & Architecture

This project implements a complete **Machine Learning Lifecycle**:

1. **Stratified Sampling:** Uses `StratifiedShuffleSplit` on `income_cat` to prevent sampling bias across income brackets.
2. **Scikit-Learn Pipeline (`ColumnTransformer`):**
   - **Numerical Pipeline:** Handles missing values via `SimpleImputer(strategy='median')` and standardizes features using `StandardScaler`.
   - **Categorical Pipeline:** One-hot encodes the `ocean_proximity` feature using `OneHotEncoder(handle_unknown='ignore')`.
3. **Model Training:** `RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42)` for optimal balance between accuracy and compact file size.
4. **Serialization (`joblib`):**
   - **Why `joblib`?** We use `joblib` instead of standard `pickle` because `joblib` is optimized for serializing large NumPy arrays and Scikit-Learn pipelines fast and efficiently without memory overhead.
5. **Web Deployment:** Integrated into an interactive UI via **Streamlit** hosted on Streamlit Cloud.

---

## 📊 Model Evaluation & Performance

Evaluated on the 20% unseen stratified test set:

- **$R^2$ Score:** **`0.8183`** (Explains **~81.8%** of price variance)
- **Root Mean Squared Error (RMSE):** **`$48,667.99`**

---

## 📁 Repository Structure
```text
california_housing_app/
├── app.py              # Streamlit Web Interface (Loads model & handles predictions)
├── main.py             # Model training & pipeline export script
├── housing.csv         # California Census Dataset
├── model.pkl           # Trained RandomForest model (16 MB)
├── pipeline.pkl        # Fitted Scikit-Learn preprocessing pipeline
├── requirements.txt    # Project dependencies
└── README.md           # Detailed documentation
```

---

## 🚀 Running Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model (Optional - Re-generates `model.pkl` & `pipeline.pkl`):**
   ```bash
   python3 main.py
   ```

3. **Run Streamlit Web App:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Cloud Deployment
Deploys seamlessly on **Streamlit Community Cloud** by linking to `app.py`.

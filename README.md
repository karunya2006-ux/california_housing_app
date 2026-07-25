# 🏡 California Housing Price Prediction App

An end-to-end Machine Learning web application built using **Scikit-Learn** and **Streamlit** to predict median California house values.

## 📌 Features
- **Data Preprocessing & Model:** Built using `StratifiedShuffleSplit`, `ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, and `RandomForestRegressor`.
- **Interactive UI:** Web application powered by `Streamlit` allowing real-time parameter tuning and immediate price estimation.

## 📁 Repository Structure
```text
california_housing_app/
├── app.py              # Streamlit Web Interface
├── main.py             # Model training & inference pipeline
├── housing.csv         # California Census Dataset
├── model.pkl           # Trained RandomForest model
├── pipeline.pkl        # Fitted Scikit-Learn preprocessing pipeline
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🚀 Running Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model (Generates `model.pkl` & `pipeline.pkl`):**
   ```bash
   python3 main.py
   ```

3. **Run Streamlit App:**
   ```bash
   streamlit run app.py
   ```

## 🌐 Free Cloud Deployment (Streamlit Community Cloud)
1. Fork or push this repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Connect your repository and set `app.py` as the main file path.

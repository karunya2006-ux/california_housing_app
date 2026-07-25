import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

/* Metric cards */
.metric-card {
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-4px); }
.metric-title { color: #a0aec0; font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
.metric-value { color: #ffffff; font-size: 28px; font-weight: 700; margin-top: 6px; }
.metric-sub   { color: #68d391; font-size: 12px; margin-top: 4px; }

/* Tab style */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #a0aec0;
    font-weight: 600;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
}

/* Predict button */
div.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 14px 40px;
    font-size: 18px;
    font-weight: 700;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.2s;
}
div.stButton > button:hover { opacity: 0.9; transform: scale(1.02); }

/* Result box */
.result-box {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    margin-top: 20px;
    box-shadow: 0 8px 32px rgba(56,239,125,0.3);
}
.result-label { color: rgba(255,255,255,0.8); font-size: 16px; font-weight: 600; letter-spacing: 1px; }
.result-value { color: white; font-size: 48px; font-weight: 800; margin-top: 8px; }

/* Form inputs */
.stNumberInput input, .stSelectbox select {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}

label, .stSlider label { color: #e2e8f0 !important; font-weight: 500 !important; }
h1, h2, h3 { color: white !important; }
p { color: #a0aec0; }
</style>
""", unsafe_allow_html=True)


# ── Data & Model Loading ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("housing.csv")

@st.cache_resource
def train_model():
    housing = load_data().copy()
    housing['income_cat'] = pd.cut(
        housing['median_income'],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(housing, housing['income_cat']))

    train_set = housing.loc[train_idx].drop('income_cat', axis=1)
    test_set  = housing.loc[test_idx].drop('income_cat', axis=1)

    y_train = train_set['median_house_value'].copy()
    X_train = train_set.drop('median_house_value', axis=1)
    y_test  = test_set['median_house_value'].copy()
    X_test  = test_set.drop('median_house_value', axis=1)

    num_attribs = X_train.drop("ocean_proximity", axis=1).columns
    cat_attribs = ["ocean_proximity"]

    pipeline = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                          ("scaler",  StandardScaler())]), num_attribs),
        ("cat", Pipeline([("onehot",  OneHotEncoder(handle_unknown="ignore"))]), cat_attribs),
    ])

    X_train_prep = pipeline.fit_transform(X_train)
    X_test_prep  = pipeline.transform(X_test)

    model = RandomForestRegressor(n_estimators=30, max_depth=15, random_state=42)
    model.fit(X_train_prep, y_train)

    r2   = r2_score(y_test, model.predict(X_test_prep))
    rmse = root_mean_squared_error(y_test, model.predict(X_test_prep))

    return model, pipeline, r2, rmse

housing = load_data()
model, pipeline, r2, rmse = train_model()


# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center; font-size:42px;'>🏡 California Housing Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:16px;'>Powered by Random Forest · Scikit-Learn · Streamlit</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊  Overview", "📈  Charts", "🏠  Prediction"])


# ══════════════════════════════ TAB 1: OVERVIEW ══════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Records</div>
            <div class="metric-value">{len(housing):,}</div>
            <div class="metric-sub">California districts</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Avg House Price</div>
            <div class="metric-value">${housing['median_house_value'].mean():,.0f}</div>
            <div class="metric-sub">Median value</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Model R² Score</div>
            <div class="metric-value">{r2:.1%}</div>
            <div class="metric-sub">Accuracy on test set</div>
        </div>""", unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">RMSE</div>
            <div class="metric-value">${rmse:,.0f}</div>
            <div class="metric-sub">Average prediction error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📋 Dataset Sample")
    st.dataframe(housing.head(10), use_container_width=True)

    st.subheader("📐 Dataset Statistics")
    st.dataframe(housing.describe().round(2), use_container_width=True)


# ══════════════════════════════ TAB 2: CHARTS ════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            housing, x="median_house_value", nbins=60,
            title="📊 House Price Distribution",
            color_discrete_sequence=["#667eea"],
            template="plotly_dark"
        )
        fig1.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='white',
            font_color='white'
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.box(
            housing, x="ocean_proximity", y="median_house_value",
            title="🌊 Price by Ocean Proximity",
            color="ocean_proximity",
            template="plotly_dark",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig2.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            title_font_color='white',
            font_color='white'
        )
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.scatter(
            housing.sample(2000, random_state=42),
            x="median_income", y="median_house_value",
            title="💰 Income vs House Value",
            opacity=0.5,
            color_discrete_sequence=["#38ef7d"],
            template="plotly_dark"
        )
        fig3.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='white',
            font_color='white'
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.scatter_mapbox(
            housing.sample(3000, random_state=42),
            lat="latitude", lon="longitude",
            color="median_house_value",
            size="population",
            color_continuous_scale="Viridis",
            zoom=4.5, height=400,
            title="🗺️ Geographic Price Map",
            mapbox_style="carto-darkmatter"
        )
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            title_font_color='white',
            font_color='white'
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════ TAB 3: PREDICTION ════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🏠 Enter Property Details")
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        longitude         = st.number_input("📍 Longitude",            value=-122.23, format="%.4f")
        latitude          = st.number_input("📍 Latitude",             value=37.88,   format="%.4f")
        housing_median_age = st.slider(     "🏗️ Housing Median Age",   1, 52, 25)

    with col2:
        total_rooms    = st.number_input("🚪 Total Rooms",    value=2000, step=50)
        total_bedrooms = st.number_input("🛏️ Total Bedrooms", value=400,  step=10)
        population     = st.number_input("👥 Population",     value=1000, step=50)

    with col3:
        households     = st.number_input("🏘️ Households",          value=350,  step=10)
        median_income  = st.number_input("💵 Median Income ($10K)", value=3.87, step=0.1, format="%.2f")
        ocean_proximity = st.selectbox( "🌊 Ocean Proximity",
            ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("💰 Predict House Value")

    if predict:
        input_df = pd.DataFrame([{
            "longitude":          float(longitude),
            "latitude":           float(latitude),
            "housing_median_age": float(housing_median_age),
            "total_rooms":        float(total_rooms),
            "total_bedrooms":     float(total_bedrooms),
            "population":         float(population),
            "households":         float(households),
            "median_income":      float(median_income),
            "ocean_proximity":    str(ocean_proximity)
        }])

        prediction = model.predict(pipeline.transform(input_df))[0]

        st.markdown(f"""
        <div class="result-box">
            <div class="result-label">🏡 ESTIMATED MEDIAN HOUSE VALUE</div>
            <div class="result-value">${prediction:,.2f}</div>
        </div>""", unsafe_allow_html=True)

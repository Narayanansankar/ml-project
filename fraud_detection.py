import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import xgboost as xgb  # Using XGBoost with CUDA
from datetime import datetime
import os
import joblib  # Fix for scaler loading issue

MODEL_FILE = "fraud_model.json"
SCALER_FILE = "scaler.pkl"
FEATURES_FILE = "feature_columns.npy"


# Load dataset
@st.cache_data
def load_data():
    if not os.path.exists(r"C:\Users\sanka\PycharmProjects\fintekathon\data\fraudTest.csv"):
        st.error("Dataset not found. Please upload fraudtest.csv")
        return None
    df = pd.read_csv(r"C:\Users\sanka\PycharmProjects\fintekathon\data\fraudTest.csv")
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    return df


def preprocess_data(df):
    df = df.copy()
    df["hour"] = df["trans_date_trans_time"].dt.hour
    df["day"] = df["trans_date_trans_time"].dt.day
    df["month"] = df["trans_date_trans_time"].dt.month
    df["dayofweek"] = df["trans_date_trans_time"].dt.dayofweek

    df["distance"] = np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
    )

    categorical_cols = ["merchant", "category", "gender", "state"]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    feature_columns = [
        "amt", "hour", "day", "month", "dayofweek", "distance",
        "merchant", "category", "state", "lat", "long", "merch_lat", "merch_long"
    ]
    return df, feature_columns


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=4,
        learning_rate=0.2,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="gpu_hist",  # Use GPU for training
        predictor="gpu_predictor",
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    model.save_model(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)  # Save entire scaler object
    np.save(FEATURES_FILE, np.array(feature_columns))

    return model, scaler


def load_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE) and os.path.exists(FEATURES_FILE):
        model = xgb.XGBClassifier()
        model.load_model(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)  # Load entire scaler object
        feature_columns = np.load(FEATURES_FILE, allow_pickle=True).tolist()
        return model, scaler, feature_columns
    return None, None, None


def predict_transaction(transaction_data, model, scaler, feature_columns):
    df = pd.DataFrame([transaction_data])
    df, _ = preprocess_data(df)
    X = df[feature_columns]
    X_scaled = scaler.transform(X)
    fraud_probability = model.predict_proba(X_scaled)[0, 1]
    return fraud_probability


df = load_data()
if df is not None and not df.empty:
    st.title("Fraud Detection System")

    model, scaler, feature_columns = load_model()
    if model is None:
        processed_df, feature_columns = preprocess_data(df)
        X = processed_df[feature_columns]
        y = processed_df["is_fraud"]
        model, scaler = train_model(X, y)

    st.session_state.model = model
    st.session_state.scaler = scaler
    st.session_state.feature_columns = feature_columns
    st.sidebar.success("Model is trained and ready (CUDA enabled).")
else:
    st.error("No data available for training. Please upload fraudtest.csv.")
    st.stop()

st.subheader("Enter New Transaction")
amount = st.number_input("Amount", min_value=0.0, value=100.0, key="amount_input")
merchant = st.text_input("Merchant", "Sample Merchant")
category = st.selectbox("Category", df["category"].unique())
state = st.selectbox("State", df["state"].unique())
gender = st.selectbox("Gender", ["M", "F"])
lat = st.number_input("Customer Latitude", value=40.7128, key="customer_lat")
long = st.number_input("Customer Longitude", value=-74.0060, key="customer_long")
merch_lat = st.number_input("Merchant Latitude", value=40.7128, key="merchant_lat")
merch_long = st.number_input("Merchant Longitude", value=-74.0060, key="merchant_long")

if st.button("Analyze Transaction"):
    if "model" not in st.session_state or "scaler" not in st.session_state:
        st.error("Model not trained. Please ensure fraudtest.csv is available.")
    else:
        transaction_data = {
            "trans_date_trans_time": datetime.now(),
            "amt": amount,
            "merchant": merchant,
            "category": category,
            "gender": gender,
            "state": state,
            "lat": lat,
            "long": long,
            "merch_lat": merch_lat,
            "merch_long": merch_long
        }
        fraud_probability = predict_transaction(transaction_data, st.session_state.model, st.session_state.scaler,
                                                st.session_state.feature_columns)
        risk_level = "High Risk" if fraud_probability > 0.7 else "Medium Risk" if fraud_probability > 0.3 else "Low Risk"
        st.metric("Fraud Probability", f"{fraud_probability:.3f}")
        st.metric("Risk Level", risk_level)
        st.success("Transaction analyzed!")

st.subheader("Fraud vs State (Map)")
fraud_by_state = df.groupby("state").agg({"is_fraud": "sum", "lat": "mean", "long": "mean"}).reset_index()
fig = px.scatter_mapbox(fraud_by_state, lat="lat", lon="long", size="is_fraud", color="is_fraud",
                        hover_name="state", title="Fraud Cases by State (Map)",
                        mapbox_style="open-street-map", zoom=3)

st.plotly_chart(fig)

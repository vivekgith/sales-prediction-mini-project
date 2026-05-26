import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load model
model = joblib.load("store_sales_model.pkl")

st.title("Daily Store Sales Prediction")

# User inputs
date = st.date_input("Select date")
promo = st.selectbox("Promo", [0, 1])
holiday = st.selectbox("Holiday", [0, 1])

# Date features
day = date.day
month = date.month
day_of_week = date.weekday()
is_weekend = 1 if day_of_week >= 5 else 0

# Lag inputs (manual for demo)
lag_1 = st.number_input("Yesterday Sales", min_value=0.0)
lag_7 = st.number_input("Last Week Same Day Sales", min_value=0.0)
rolling_mean_7 = st.number_input("Last 7 Days Average Sales", min_value=0.0)

if st.button("Predict Sales"):
    input_data = pd.DataFrame([[
        promo, holiday, day, month,
        day_of_week, is_weekend,
        lag_1, lag_7, rolling_mean_7
    ]], columns=[
        'promo', 'holiday', 'day', 'month',
        'day_of_week', 'is_weekend',
        'lag_1', 'lag_7', 'rolling_mean_7'
    ])

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Sales: {prediction:.2f}")

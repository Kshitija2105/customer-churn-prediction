import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load model files
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

st.title("📊 Customer Churn Prediction System")

st.write("Enter customer details below:")

# Create inputs dynamically
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(feature, value=0.0)

input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict"):

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)

    if prediction[0] == 1:
        st.error(f"Customer will churn ❌ (Probability: {probability[0][1]*100:.2f}%)")
    else:
        st.success(f"Customer will NOT churn ✅ (Probability: {probability[0][1]*100:.2f}%)")
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

# ==============================
# USER INPUTS (CLEAN UI)
# ==============================

credit_score = st.number_input("Credit Score", 300, 900, 650)
age = st.number_input("Age", 18, 100, 35)
tenure = st.number_input("Tenure (Years)", 0, 10, 5)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

# Categorical inputs
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Female", "Male"])

# ==============================
# CREATE INPUT DATAFRAME
# ==============================

input_dict = {
    "CreditScore": credit_score,
    "Age": age,
    "Tenure": tenure,
    "Balance": balance,
    "NumOfProducts": num_products,
    "HasCrCard": has_cr_card,
    "IsActiveMember": is_active,
    "EstimatedSalary": estimated_salary,
    "Geography_Germany": 1 if geography == "Germany" else 0,
    "Geography_Spain": 1 if geography == "Spain" else 0,
    "Gender_Male": 1 if gender == "Male" else 0
}

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df[features]

# ==============================
# PREDICTION
# ==============================

if st.button("Predict"):

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Get probability
    prob = model.predict_proba(input_scaled)[0][1]

    st.subheader("Prediction Result")

    # Adjusted threshold (IMPORTANT FIX)
    if prob > 0.3:
        st.error(f"Customer is likely to churn ❌")
    else:
        st.success(f"Customer is NOT likely to churn ✅")

    # Show probability
    st.write(f"Churn Probability: {prob*100:.2f}%")

    # Progress bar (nice UI)
    st.progress(int(prob * 100))
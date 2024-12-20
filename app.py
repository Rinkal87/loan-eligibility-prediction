import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load('loan_eligibility_model.pkl')

# Get expected feature names from the model
expected_features = model.feature_names_in_

# Title of the app
st.title("Loan Eligibility Prediction App")

# User inputs
st.header("Enter Applicant Details:")
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0, step=1)
loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=1, step=1)
credit_history = st.selectbox("Credit History", ["0.0", "1.0"])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Map categorical inputs to match one-hot encoding
input_data = {
    'ApplicantIncome': applicant_income,
    'CoapplicantIncome': coapplicant_income,
    'LoanAmount': loan_amount,
    'Loan_Amount_Term': loan_amount_term,
    'Credit_History': float(credit_history),
    'Gender_Male': 1 if gender == "Male" else 0,
    'Gender_Female': 1 if gender == "Female" else 0,
    'Married_Yes': 1 if married == "Yes" else 0,
    'Married_No': 1 if married == "No" else 0,
    'Dependents_0': 1 if dependents == "0" else 0,
    'Dependents_1': 1 if dependents == "1" else 0,
    'Dependents_2': 1 if dependents == "2" else 0,
    'Dependents_3+': 1 if dependents == "3+" else 0,
    'Education_Graduate': 1 if education == "Graduate" else 0,
    'Education_Not Graduate': 1 if education == "Not Graduate" else 0,
    'Self_Employed_Yes': 1 if self_employed == "Yes" else 0,
    'Self_Employed_No': 1 if self_employed == "No" else 0,
    'Property_Area_Urban': 1 if property_area == "Urban" else 0,
    'Property_Area_Semiurban': 1 if property_area == "Semiurban" else 0,
    'Property_Area_Rural': 1 if property_area == "Rural" else 0,
}

# Prepare input dataframe
df = pd.DataFrame([input_data])

# Add missing features with default value
for feature in expected_features:
    if feature not in df.columns:
        df[feature] = 0

# Ensure column order matches
df = df[expected_features]

# Make predictions
if st.button("Predict Loan Eligibility"):
    prediction = model.predict(df)
    if prediction[0] == 'Y':
        st.success("The applicant is eligible for a loan.")
    else:
        st.error("The applicant is not eligible for a loan.")

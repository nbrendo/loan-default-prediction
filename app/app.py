# =========================
# LOAN DEFAULT WEB APP
# =========================

import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================

model = joblib.load(
    '../models/loan_default_xgb_model.pkl'
)

# =========================
# APP TITLE
# =========================

st.title(
    'Loan Default Prediction System'
)

st.write(
    'Real World Finance Risk Prediction'
)

# =========================
# USER INPUTS
# =========================

st.header("Customer Information")

# Example inputs

customer_age = st.number_input(
    'Customer Age',
    min_value=18,
    max_value=100,
    value=30
)

loan_amount = st.number_input(
    'Loan Amount',
    min_value=0.0,
    value=1000.0
)

income = st.number_input(
    'Monthly Income',
    min_value=0.0,
    value=500.0
)

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({
    'age': [customer_age],
    'loan_amount': [loan_amount],
    'income': [income]
})

# =========================
# PREDICT BUTTON
# =========================

if st.button('Predict Loan Risk'):

    prediction = model.predict(
        input_data
    )

    probability = model.predict_proba(
        input_data
    )[:,1]

    st.subheader('Prediction Result')

    if prediction[0] == 1:

        st.error(
            'High Risk Customer'
        )

    else:

        st.success(
            'Low Risk Customer'
        )

    st.write(
        'Default Probability:',
        probability[0]
    )
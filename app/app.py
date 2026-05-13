# =========================
# LOAN DEFAULT WEB APP - FINAL VERSION
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import random
import string

# =========================
# LOAD ARTIFACTS
# =========================

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/loan_default_xgb_model.pkl')
    label_encoders = joblib.load('models/label_encoders.pkl')
    imputer = joblib.load('models/imputer.pkl')
    return model, label_encoders, imputer

model, label_encoders, imputer = load_artifacts()

# =========================
# PREPROCESSING
# =========================

def preprocess_input(input_data):
    """Apply all preprocessing steps to match training"""
    df = input_data.copy()
    
    # Identify columns
    date_cols = ['date_approved', 'date_disbursed', 'first_payment_due', 'maturity_date', 'client_dob']
    
    # Convert dates to Unix timestamps
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], format='mixed', dayfirst=True).astype('int64') // 10**9
    
    # Label encode categorical columns
    for col in label_encoders.keys():
        if col in df.columns:
            try:
                df[col] = label_encoders[col].transform(df[col].astype(str))
            except ValueError:
                df[col] = -1
    
    # Convert product_code to numeric
    if 'product_code' in df.columns:
        df['product_code'] = pd.to_numeric(df['product_code'], errors='coerce')
    
    # Create engineered features
    df['customer_age'] = (pd.Timestamp('today') - pd.to_datetime(df['client_dob'], unit='s')).dt.days // 365
    
    df['loan_duration_days'] = (pd.to_datetime(df['maturity_date'], unit='s') - pd.to_datetime(df['date_disbursed'], unit='s')).dt.days
    
    df['loan_income_ratio'] = df['amount_usd'] / (df['monthly_income_usd'] + 1)
    
    df['debt_income_ratio'] = df['existing_obligations'] / (df['monthly_income_usd'] + 1)
    
    # Fill NaN values
    df = df.fillna(0)
    
    # Apply imputer (was fitted on 27 columns including Target)
    # The model expects 22 columns, imputer needs 27
    # We need to add missing columns that imputer expects
    for col in imputer.feature_names_in_:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder to match imputer
    df = df[imputer.feature_names_in_]
    
    # Apply imputer
    df_imputed = pd.DataFrame(imputer.transform(df), columns=df.columns)
    
    # Select only the 22 features the model expects
    model_features = list(model.feature_names_in_)
    df_imputed = df_imputed[model_features]
    
    return df_imputed

# =========================
# APP TITLE
# =========================

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title('🏦 Loan Default Prediction System')

# =========================
# USER INPUTS
# =========================

st.header("📋 Loan Application Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    client_gender = st.selectbox('Gender', ['Male', 'Female'])
    client_dob = st.date_input('Date of Birth', value=datetime.date(1990, 1, 1))
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
    num_dependents = st.number_input('Number of Dependents', min_value=0, max_value=20, value=0)
    employment_sector = st.selectbox('Employment Sector', ['Private', 'Government', 'Self-Employed', 'Unemployed'])
    months_at_employer = st.number_input('Months at Employer', min_value=0, value=24)
    monthly_income_usd = st.number_input('Monthly Income (USD)', min_value=0.0, value=3000.0, step=100.0)
    province = st.selectbox('Province', ['Province1', 'Province2', 'Province3', 'Capital Region'])

with col2:
    st.subheader("Loan Details")
    product_code = st.selectbox('Product Code', ['LOAN-001', 'LOAN-002', 'LOAN-003'])
    amount_usd = st.number_input('Loan Amount (USD)', min_value=100.0, value=10000.0, step=500.0)
    annual_rate_pct = st.number_input('Annual Rate (%)', min_value=1.0, max_value=50.0, value=10.0)
    term_months = st.number_input('Term (Months)', min_value=1, max_value=360, value=36)
    payment_frequency = st.selectbox('Payment Frequency', ['Monthly', 'Bi-weekly', 'Weekly'])
    loan_purpose = st.selectbox('Loan Purpose', ['Home Purchase', 'Debt Consolidation', 'Business', 'Education'])
    collateral_type = st.selectbox('Collateral Type', ['Property', 'Vehicle', 'Savings', 'None'])
    disbursement_channel = st.selectbox('Disbursement Channel', ['Bank Transfer', 'Check', 'Cash'])
    existing_obligations = st.number_input('Existing Obligations (USD)', min_value=0.0, value=0.0, step=100.0)

# Generate auto fields
loan_id = 'LOAN-' + ''.join(random.choices(string.digits, k=8))
date_approved = datetime.date.today()
date_disbursed = date_approved + datetime.timedelta(days=7)
first_payment_due = date_disbursed + datetime.timedelta(days=30)
maturity_date = date_disbursed + datetime.timedelta(days=term_months * 30)

# Create input dataframe
input_data = pd.DataFrame({
    'ID': [loan_id],
    'product_code': [product_code],
    'date_approved': [date_approved.strftime('%Y-%m-%d')],
    'date_disbursed': [date_disbursed.strftime('%Y-%m-%d')],
    'first_payment_due': [first_payment_due.strftime('%Y-%m-%d')],
    'maturity_date': [maturity_date.strftime('%Y-%m-%d')],
    'amount_usd': [amount_usd],
    'annual_rate_pct': [annual_rate_pct],
    'term_months': [term_months],
    'payment_frequency': [payment_frequency],
    'loan_purpose': [loan_purpose],
    'client_gender': [client_gender],
    'client_dob': [client_dob.strftime('%Y-%m-%d')],
    'marital_status': [marital_status],
    'num_dependents': [num_dependents],
    'employment_sector': [employment_sector],
    'months_at_employer': [months_at_employer],
    'monthly_income_usd': [monthly_income_usd],
    'existing_obligations': [existing_obligations],
    'collateral_type': [collateral_type],
    'disbursement_channel': [disbursement_channel],
    'province': [province]
})

# =========================
# PREDICT
# =========================

st.markdown("---")

if st.button('🔍 Predict Loan Risk', type='primary', use_container_width=True):
    try:
        processed_input = preprocess_input(input_data)
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)[:, 1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction[0] == 1:
                st.error('🔴 HIGH RISK')
            else:
                st.success('🟢 LOW RISK')
        with col2:
            st.metric('Default Probability', f'{probability[0]:.1%}')
        with col3:
            st.metric('Risk Score', f'{int(probability[0] * 1000)}/1000')
            
    except Exception as e:
        st.error(f'Error: {str(e)}')
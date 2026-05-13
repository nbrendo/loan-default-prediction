# =========================
# LOAN DEFAULT WEB APP - WITH PREPROCESSING
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import random
import string

# =========================
# LOAD MODEL AND PREPROCESSORS
# =========================

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/loan_default_xgb_model.pkl')
    
    # Load your preprocessing artifacts
    try:
        label_encoders = joblib.load('models/label_encoders.pkl')
    except:
        label_encoders = None
        
    try:
        imputer = joblib.load('models/imputer.pkl')
    except:
        imputer = None
        
    return model, label_encoders, imputer

model, label_encoders, imputer = load_artifacts()

# =========================
# PREPROCESS FUNCTION
# =========================

def preprocess_input(df):
    """Apply the same preprocessing as training"""
    
    # Convert dates to numeric features (if that's how your model was trained)
    for col in ['date_approved', 'date_disbursed', 'first_payment_due', 'maturity_date', 'client_dob']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            # Extract useful numeric features from dates
            df[col + '_year'] = df[col].dt.year
            df[col + '_month'] = df[col].dt.month
            df[col + '_day'] = df[col].dt.day
    
    # Drop original date columns if your model doesn't use them
    date_cols = ['date_approved', 'date_disbursed', 'first_payment_due', 'maturity_date', 'client_dob']
    df = df.drop(columns=[c for c in date_cols if c in df.columns], errors='ignore')
    
    # Apply label encoding for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if label_encoders:
        # Use saved label encoders
        for col in categorical_cols:
            if col in label_encoders:
                try:
                    df[col] = label_encoders[col].transform(df[col].astype(str))
                except:
                    # Handle unseen categories
                    df[col] = df[col].map(lambda x: -1 if x not in label_encoders[col].classes_ 
                                         else label_encoders[col].transform([x])[0])
    else:
        # Fallback: simple label encoding
        for col in categorical_cols:
            df[col] = pd.factorize(df[col])[0]
    
    # Apply imputer if needed
    if imputer:
        df = pd.DataFrame(imputer.transform(df), columns=df.columns)
    
    # Ensure all columns are numeric
    df = df.astype(float)
    
    # Ensure correct column order matching training
    if hasattr(model, 'feature_names_in_'):
        # Drop any extra columns we created
        df = df[[c for c in model.feature_names_in_ if c in df.columns]]
        
        # Add missing columns with zeros
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
                
        # Reorder to match training
        df = df[model.feature_names_in_]
    
    return df

# =========================
# APP TITLE
# =========================

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title('🏦 Loan Default Prediction System')
st.write('Predict the likelihood of loan default using machine learning')

# =========================
# USER INPUTS
# =========================

st.header("📋 Loan Application Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    
    client_gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
    
    client_dob = st.date_input(
        'Date of Birth',
        value=datetime.date(1990, 1, 1),
        min_value=datetime.date(1940, 1, 1),
        max_value=datetime.date(2006, 1, 1)
    )
    
    marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced', 'Widowed'])
    
    num_dependents = st.number_input('Number of Dependents', min_value=0, max_value=20, value=0)
    
    employment_sector = st.selectbox(
        'Employment Sector',
        ['Private', 'Government', 'Self-Employed', 'Unemployed', 'Retired', 'Student']
    )
    
    months_at_employer = st.number_input('Months at Current Employer', min_value=0, value=24)
    
    monthly_income_usd = st.number_input('Monthly Income (USD)', min_value=0.0, value=3000.0, step=100.0)
    
    province = st.selectbox('Province', ['Province1', 'Province2', 'Province3', 'Capital Region', 'North', 'South', 'East', 'West'])

with col2:
    st.subheader("Loan Details")
    
    product_code = st.selectbox('Product Code', ['LOAN-001', 'LOAN-002', 'LOAN-003', 'LOAN-004', 'LOAN-005'])
    
    amount_usd = st.number_input('Loan Amount (USD)', min_value=100.0, value=10000.0, step=500.0)
    
    annual_rate_pct = st.number_input('Annual Interest Rate (%)', min_value=1.0, max_value=50.0, value=10.0, step=0.5)
    
    term_months = st.number_input('Term (Months)', min_value=1, max_value=360, value=36)
    
    payment_frequency = st.selectbox('Payment Frequency', ['Monthly', 'Bi-weekly', 'Weekly', 'Quarterly'])
    
    loan_purpose = st.selectbox(
        'Loan Purpose',
        ['Home Purchase', 'Home Improvement', 'Debt Consolidation', 'Business', 'Education', 'Medical', 'Vehicle', 'Other']
    )
    
    collateral_type = st.selectbox('Collateral Type', ['Property', 'Vehicle', 'Savings', 'None', 'Other'])
    
    disbursement_channel = st.selectbox('Disbursement Channel', ['Bank Transfer', 'Check', 'Cash', 'Mobile Money'])
    
    existing_obligations = st.number_input('Existing Obligations (USD)', min_value=0.0, value=0.0, step=100.0)

# =========================
# AUTO-GENERATED FIELDS
# =========================

loan_id = 'LOAN-' + ''.join(random.choices(string.digits, k=8))
date_approved = datetime.date.today()
date_disbursed = date_approved + datetime.timedelta(days=7)
first_payment_due = date_disbursed + datetime.timedelta(days=30)
maturity_date = date_disbursed + datetime.timedelta(days=term_months * 30)

# =========================
# CREATE INPUT DATAFRAME
# =========================

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
# PREDICT BUTTON
# =========================

st.markdown("---")

if st.button('🔍 Predict Loan Risk', type='primary', use_container_width=True):
    
    try:
        # Apply preprocessing
        processed_input = preprocess_input(input_data.copy())
        
        # Make prediction with processed data
        prediction = model.predict(processed_input)
        probability = model.predict_proba(processed_input)[:, 1]
        
        st.markdown("---")
        st.subheader('📊 Prediction Result')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction[0] == 1:
                st.error('🔴 HIGH RISK - Default Likely')
            else:
                st.success('🟢 LOW RISK - Default Unlikely')
        
        with col2:
            st.metric('Default Probability', f'{probability[0]:.1%}')
        
        with col3:
            risk_score = int(probability[0] * 1000)
            st.metric('Risk Score', f'{risk_score}/1000')
        
        # Show debug info in expander
        with st.expander("Debug: Processed Features"):
            st.write("Processed dataframe shape:", processed_input.shape)
            st.write("Columns:", list(processed_input.columns))
            st.dataframe(processed_input.head())
            
    except Exception as e:
        st.error(f'Prediction error: {str(e)}')
        
        # Fallback: Try with just numeric columns
        st.info("Attempting fallback prediction with numeric features only...")
        try:
            numeric_df = input_data.select_dtypes(include=[np.number])
            fallback_prediction = model.predict(numeric_df.iloc[:, :min(len(numeric_df.columns), 22)])
            st.success("Fallback prediction succeeded")
        except:
            st.error("Fallback also failed. Please check model requirements.")
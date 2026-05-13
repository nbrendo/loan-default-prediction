# =========================
# LOAN DEFAULT WEB APP
# =========================

import streamlit as st
import pandas as pd
import joblib
import datetime

# =========================
# LOAD MODEL
# =========================

@st.cache_resource
def load_model():
    return joblib.load('models/loan_default_xgb_model.pkl')

model = load_model()

# =========================
# APP TITLE
# =========================

st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title('🏦 Loan Default Prediction System')
st.write('Predict the likelihood of loan default using machine learning')

# =========================
# USER INPUTS - TWO COLUMNS
# =========================

st.header("📋 Loan Application Details")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Customer Information")
    
    client_gender = st.selectbox(
        'Gender',
        ['Male', 'Female', 'Other']
    )
    
    client_dob = st.date_input(
        'Date of Birth',
        value=datetime.date(1990, 1, 1),
        min_value=datetime.date(1940, 1, 1),
        max_value=datetime.date(2006, 1, 1)
    )
    
    marital_status = st.selectbox(
        'Marital Status',
        ['Single', 'Married', 'Divorced', 'Widowed']
    )
    
    num_dependents = st.number_input(
        'Number of Dependents',
        min_value=0,
        max_value=20,
        value=0
    )
    
    employment_sector = st.selectbox(
        'Employment Sector',
        ['Private', 'Government', 'Self-Employed', 'Unemployed', 'Retired', 'Student']
    )
    
    months_at_employer = st.number_input(
        'Months at Current Employer',
        min_value=0,
        value=24
    )
    
    monthly_income_usd = st.number_input(
        'Monthly Income (USD)',
        min_value=0.0,
        value=3000.0,
        step=100.0
    )
    
    province = st.selectbox(
        'Province',
        ['Province1', 'Province2', 'Province3', 'Capital Region', 'North', 'South', 'East', 'West']
    )

with col2:
    st.subheader("Loan Details")
    
    product_code = st.selectbox(
        'Product Code',
        ['LOAN-001', 'LOAN-002', 'LOAN-003', 'LOAN-004', 'LOAN-005']
    )
    
    amount_usd = st.number_input(
        'Loan Amount (USD)',
        min_value=100.0,
        value=10000.0,
        step=500.0
    )
    
    annual_rate_pct = st.number_input(
        'Annual Interest Rate (%)',
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=0.5
    )
    
    term_months = st.number_input(
        'Term (Months)',
        min_value=1,
        max_value=360,
        value=36
    )
    
    payment_frequency = st.selectbox(
        'Payment Frequency',
        ['Monthly', 'Bi-weekly', 'Weekly', 'Quarterly']
    )
    
    loan_purpose = st.selectbox(
        'Loan Purpose',
        ['Home Purchase', 'Home Improvement', 'Debt Consolidation', 
         'Business', 'Education', 'Medical', 'Vehicle', 'Other']
    )
    
    collateral_type = st.selectbox(
        'Collateral Type',
        ['Property', 'Vehicle', 'Savings', 'None', 'Other']
    )
    
    disbursement_channel = st.selectbox(
        'Disbursement Channel',
        ['Bank Transfer', 'Check', 'Cash', 'Mobile Money']
    )
    
    existing_obligations = st.number_input(
        'Existing Obligations (USD)',
        min_value=0.0,
        value=0.0,
        step=100.0
    )

# =========================
# AUTO-GENERATED FIELDS
# =========================

# Generate unique ID and dates automatically
import random
import string

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
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]
        
        st.markdown("---")
        st.subheader('📊 Prediction Result')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction[0] == 1:
                st.error('🔴 HIGH RISK - Default Likely')
            else:
                st.success('🟢 LOW RISK - Default Unlikely')
        
        with col2:
            st.metric(
                'Default Probability', 
                f'{probability[0]:.1%}',
                delta='High Risk' if probability[0] > 0.5 else 'Low Risk'
            )
        
        with col3:
            risk_score = int(probability[0] * 1000)
            st.metric('Risk Score', f'{risk_score}/1000')
        
        # Additional insights
        st.markdown("---")
        st.subheader('💡 Loan Summary')
        
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write(f"**Loan Amount:** ${amount_usd:,.2f}")
            st.write(f"**Monthly Payment:** ${amount_usd / term_months:,.2f}")
            st.write(f"**Interest Rate:** {annual_rate_pct}%")
            
        with summary_col2:
            st.write(f"**Loan Term:** {term_months} months")
            monthly_obligation_ratio = (existing_obligations + amount_usd/term_months) / monthly_income_usd * 100
            st.write(f"**DTI Ratio:** {monthly_obligation_ratio:.1f}%")
            
            if monthly_obligation_ratio > 43:
                st.warning('⚠️ High debt-to-income ratio (>43%)')
                
    except Exception as e:
        st.error(f'Prediction error: {str(e)}')
        st.info('Please ensure all fields are filled correctly.')

# Footer
st.markdown("---")
st.markdown("*Loan Default Prediction System v1.0 | Powered by XGBoost*")
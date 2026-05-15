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

    df = input_data.copy()

    # =========================
    # DATE COLUMNS
    # =========================

    date_cols = [
        'date_approved',
        'date_disbursed',
        'first_payment_due',
        'maturity_date',
        'client_dob'
    ]

    for col in date_cols:
        df[col] = pd.to_datetime(df[col])
        df[col] = df[col].astype('int64') // 10**9

    # =========================
    # FEATURE ENGINEERING
    # =========================

    df['customer_age'] = (
        pd.Timestamp.today() -
        pd.to_datetime(df['client_dob'], unit='s')
    ).dt.days // 365

    df['loan_duration_days'] = (
        pd.to_datetime(df['maturity_date'], unit='s') -
        pd.to_datetime(df['date_disbursed'], unit='s')
    ).dt.days

    df['loan_income_ratio'] = (
        df['amount_usd'] /
        (df['monthly_income_usd'] + 1)
    )

    df['debt_income_ratio'] = (
        df['existing_obligations'] /
        (df['monthly_income_usd'] + 1)
    )

    # =========================
    # LABEL ENCODING
    # =========================

    for col in label_encoders.keys():

        if col in df.columns:

            df[col] = df[col].astype(str)

            known_labels = set(label_encoders[col].classes_)

            df[col] = df[col].apply(
                lambda x: x if x in known_labels else 'Unknown'
            )

            if 'Unknown' not in label_encoders[col].classes_:
                label_encoders[col].classes_ = np.append(
                    label_encoders[col].classes_,
                    'Unknown'
                )

            df[col] = label_encoders[col].transform(df[col])

    # =========================
    # FILL MISSING VALUES
    # =========================

    df = df.fillna(0)

    # =========================
    # MATCH IMPUTER COLUMNS
    # =========================

    for col in imputer.feature_names_in_:

        if col not in df.columns:
            df[col] = 0

    df = df[imputer.feature_names_in_]

    df = pd.DataFrame(
        imputer.transform(df),
        columns=df.columns
    )

    # =========================
    # MATCH MODEL FEATURES
    # =========================

    model_features = list(model.feature_names_in_)

    df = df[model_features]

    return df

# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="Loan Default Prediction",
    layout="wide"
)

st.title("🏦 Loan Default Prediction System")

# =========================
# USER INPUTS
# =========================

st.header("📋 Loan Application Details")

col1, col2 = st.columns(2)

# =========================
# CUSTOMER INFORMATION
# =========================

with col1:

    st.subheader("Customer Information")

    client_gender = st.selectbox(
        'Gender',
        ['Male', 'Female']
    )

    client_dob = st.date_input(
        'Date of Birth',
        value=datetime.date(1995, 1, 1)
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
        [
            'Agriculture',
            'Business',
            'Civil Service',
            'Construction',
            'Education',
            'Health',
            'Mining',
            'Self Employed',
            'Transport',
            'Unemployed'
        ]
    )

    months_at_employer = st.number_input(
        'Months at Employer',
        min_value=0,
        value=24
    )

    monthly_income_usd = st.number_input(
        'Monthly Income (USD)',
        min_value=0.0,
        value=500.0,
        step=50.0
    )

    province = st.selectbox(
        'Province',
        [
            'Bulawayo',
            'Harare',
            'Manicaland',
            'Mashonaland Central',
            'Mashonaland East',
            'Mashonaland West',
            'Masvingo',
            'Matabeleland North',
            'Matabeleland South',
            'Midlands'
        ]
    )

# =========================
# LOAN INFORMATION
# =========================

with col2:

    st.subheader("Loan Details")

    product_code = st.selectbox(
        'Product Code',
        [
            '101',
            '102',
            '103',
            '104',
            '105'
        ]
    )

    amount_usd = st.number_input(
        'Loan Amount (USD)',
        min_value=50.0,
        value=1000.0,
        step=50.0
    )

    annual_rate_pct = st.number_input(
        'Annual Interest Rate (%)',
        min_value=1.0,
        max_value=100.0,
        value=25.0
    )

    term_months = st.number_input(
        'Loan Term (Months)',
        min_value=1,
        max_value=120,
        value=12
    )

    payment_frequency = st.selectbox(
        'Payment Frequency',
        [
            'Daily',
            'Weekly',
            'Bi-Weekly',
            'Monthly'
        ]
    )

    loan_purpose = st.selectbox(
        'Loan Purpose',
        [
            'Business',
            'Education',
            'Emergency',
            'Farming',
            'Home Improvement',
            'Medical',
            'School Fees',
            'Transport'
        ]
    )

    collateral_type = st.selectbox(
        'Collateral Type',
        [
            'House',
            'Land',
            'Motor Vehicle',
            'None',
            'Salary',
            'Savings'
        ]
    )

    disbursement_channel = st.selectbox(
        'Disbursement Channel',
        [
            'Bank Transfer',
            'Cash',
            'EcoCash',
            'OneMoney'
        ]
    )

    existing_obligations = st.number_input(
        'Existing Obligations (USD)',
        min_value=0.0,
        value=0.0,
        step=50.0
    )

# =========================
# AUTO GENERATED FIELDS
# =========================

loan_id = 'LN-' + ''.join(
    random.choices(string.digits, k=8)
)

date_approved = datetime.date.today()

date_disbursed = (
    date_approved +
    datetime.timedelta(days=1)
)

first_payment_due = (
    date_disbursed +
    datetime.timedelta(days=30)
)

maturity_date = (
    date_disbursed +
    datetime.timedelta(days=term_months * 30)
)

# =========================
# CREATE INPUT DATAFRAME
# =========================

input_data = pd.DataFrame({

    'ID': [loan_id],

    'product_code': [product_code],

    'date_approved': [
        date_approved.strftime('%Y-%m-%d')
    ],

    'date_disbursed': [
        date_disbursed.strftime('%Y-%m-%d')
    ],

    'first_payment_due': [
        first_payment_due.strftime('%Y-%m-%d')
    ],

    'maturity_date': [
        maturity_date.strftime('%Y-%m-%d')
    ],

    'amount_usd': [amount_usd],

    'annual_rate_pct': [annual_rate_pct],

    'term_months': [term_months],

    'payment_frequency': [payment_frequency],

    'loan_purpose': [loan_purpose],

    'client_gender': [client_gender],

    'client_dob': [
        client_dob.strftime('%Y-%m-%d')
    ],

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
# PREDICTION
# =========================

st.markdown("---")

if st.button(
    '🔍 Predict Loan Risk',
    type='primary',
    use_container_width=True
):

    try:

        processed_input = preprocess_input(input_data)

        prediction = model.predict(processed_input)

        probability = model.predict_proba(
            processed_input
        )[:, 1]

        col1, col2, col3 = st.columns(3)

        with col1:

            if prediction[0] == 1:
                st.error("🔴 HIGH RISK")
            else:
                st.success("🟢 LOW RISK")

        with col2:

            st.metric(
                "Default Probability",
                f"{probability[0]:.2%}"
            )

        with col3:

            st.metric(
                "Risk Score",
                f"{int(probability[0] * 1000)}/1000"
            )

    except Exception as e:

        st.error(f"Error: {str(e)}")
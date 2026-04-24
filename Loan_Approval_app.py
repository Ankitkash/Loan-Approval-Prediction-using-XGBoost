"""
============================================================================
 Streamlit Web App — Loan Approval Prediction
 Run with: streamlit run app.py
============================================================================
 This app provides an interactive web interface for the XGBoost loan
 approval model. Users can enter loan application details and get instant
 predictions with probability scores and visual feedback.
============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────
# Custom CSS for better styling
# ──────────────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box-approved {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .prediction-box-rejected {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .info-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# Load the Trained Model (cached for performance)
# ──────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the saved XGBoost model."""
    model_path = "model/xgboost_loan_approval_model.pkl"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at {model_path}")
        st.info("Please run `python xgboost_loan_approval.py` first to train and save the model.")
        st.stop()
    return joblib.load(model_path)


model = load_model()


# ──────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🏦 Loan Approval Predictor</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">AI-powered loan approval prediction using XGBoost | '
    'Enter applicant details below to get instant prediction</p>',
    unsafe_allow_html=True
)


# ──────────────────────────────────────────────────────────────────────────
# Sidebar — About & Info
# ──────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ About")
    st.info(
        "This app uses an **XGBoost** classifier trained on the "
        "Loan Prediction Dataset with 614 loan applications."
    )

    st.header("🎯 Model Details")
    st.markdown("""
    - **Algorithm:** XGBoost Classifier
    - **Features:** 16 (including engineered)
    - **Accuracy:** ~85%
    - **ROC-AUC:** ~0.82
    """)

    st.header("📊 How It Works")
    st.markdown("""
    1. Fill in applicant details
    2. Click **Predict**
    3. Get instant approval decision
    4. See probability score
    """)

    st.header("⚠️ Disclaimer")
    st.warning(
        "This is a learning project. Real loan decisions "
        "should be made by qualified professionals."
    )


# ──────────────────────────────────────────────────────────────────────────
# Input Form — Organized in columns
# ──────────────────────────────────────────────────────────────────────────
st.header("📝 Applicant Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Personal Info")
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["No", "Yes"])

with col2:
    st.subheader("Income Details")
    applicant_income = st.number_input(
        "Applicant Income (₹/month)",
        min_value=0, max_value=100000, value=5000, step=500
    )
    coapplicant_income = st.number_input(
        "Co-applicant Income (₹/month)",
        min_value=0, max_value=50000, value=2000, step=500
    )
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

with col3:
    st.subheader("Loan Details")
    loan_amount = st.number_input(
        "Loan Amount (₹ thousands)",
        min_value=10, max_value=700, value=150, step=10
    )
    loan_term = st.selectbox(
        "Loan Term (months)",
        [360, 180, 120, 240, 300, 60, 36, 12, 84, 480],
        index=0
    )
    credit_history = st.selectbox(
        "Credit History",
        [1.0, 0.0],
        format_func=lambda x: "Good (1)" if x == 1.0 else "Bad (0)"
    )


# ──────────────────────────────────────────────────────────────────────────
# Prediction Button
# ──────────────────────────────────────────────────────────────────────────
st.markdown("---")
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])

with col_btn2:
    predict_btn = st.button("🔮 Predict Loan Approval", use_container_width=True, type="primary")


# ──────────────────────────────────────────────────────────────────────────
# Make Prediction
# ──────────────────────────────────────────────────────────────────────────
if predict_btn:
    # Encode categorical inputs (same encoding used during training)
    gender_enc = 1 if gender == "Male" else 0
    married_enc = 1 if married == "Yes" else 0
    dependents_enc = 3 if dependents == "3+" else int(dependents)
    education_enc = 0 if education == "Graduate" else 1
    self_employed_enc = 1 if self_employed == "Yes" else 0
    property_area_enc = {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area]

    # Feature engineering (same as training)
    total_income = applicant_income + coapplicant_income
    emi = loan_amount / loan_term
    balance_income = total_income - (emi * 1000)
    loan_amount_log = np.log1p(loan_amount)
    total_income_log = np.log1p(total_income)

    # Create input dataframe in the EXACT column order expected by the model
    input_data = pd.DataFrame({
        "Gender": [gender_enc],
        "Married": [married_enc],
        "Dependents": [dependents_enc],
        "Education": [education_enc],
        "Self_Employed": [self_employed_enc],
        "ApplicantIncome": [applicant_income],
        "CoapplicantIncome": [coapplicant_income],
        "LoanAmount": [loan_amount],
        "Loan_Amount_Term": [loan_term],
        "Credit_History": [credit_history],
        "Property_Area": [property_area_enc],
        "TotalIncome": [total_income],
        "EMI": [emi],
        "BalanceIncome": [balance_income],
        "LoanAmount_log": [loan_amount_log],
        "TotalIncome_log": [total_income_log],
    })

    # Predict
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    # Display result
    st.markdown("---")
    st.header("🎯 Prediction Result")

    if prediction == 1:
        st.markdown(f"""
            <div class="prediction-box-approved">
                <h2 style="color: #28a745; margin: 0;">✅ LOAN APPROVED</h2>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    Congratulations! Based on the provided details, this loan application is likely to be approved.
                </p>
                <p style="font-size: 1rem; margin: 0;">
                    <b>Confidence:</b> {probability[1] * 100:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="prediction-box-rejected">
                <h2 style="color: #dc3545; margin: 0;">❌ LOAN REJECTED</h2>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    Unfortunately, based on the provided details, this loan application is likely to be rejected.
                </p>
                <p style="font-size: 1rem; margin: 0;">
                    <b>Confidence:</b> {probability[0] * 100:.2f}%
                </p>
            </div>
        """, unsafe_allow_html=True)

    # Probability breakdown
    st.subheader("📊 Probability Breakdown")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.metric(
            label="❌ Probability of Rejection",
            value=f"{probability[0] * 100:.2f}%"
        )
        st.progress(float(probability[0]))

    with col_p2:
        st.metric(
            label="✅ Probability of Approval",
            value=f"{probability[1] * 100:.2f}%"
        )
        st.progress(float(probability[1]))

    # Input summary
    with st.expander("📋 View Applicant Input Summary"):
        summary_df = pd.DataFrame({
            "Field": [
                "Gender", "Married", "Dependents", "Education", "Self Employed",
                "Applicant Income", "Co-applicant Income", "Total Income",
                "Loan Amount", "Loan Term", "Credit History", "Property Area",
                "Estimated EMI"
            ],
            "Value": [
                gender, married, dependents, education, self_employed,
                f"₹{applicant_income:,}", f"₹{coapplicant_income:,}", f"₹{total_income:,}",
                f"₹{loan_amount},000", f"{loan_term} months",
                "Good" if credit_history == 1.0 else "Bad",
                property_area, f"₹{emi * 1000:,.2f}/month"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Key insights
    st.subheader("💡 Key Factors")
    insights = []

    if credit_history == 1.0:
        insights.append("✅ **Good credit history** — strongest positive factor")
    else:
        insights.append("❌ **Poor credit history** — major concern for approval")

    if total_income > 7000:
        insights.append("✅ **Strong total household income**")
    elif total_income < 3000:
        insights.append("⚠️ **Low total household income** — may affect repayment capacity")

    if emi * 1000 > total_income * 0.5:
        insights.append("⚠️ **High EMI-to-income ratio** — over 50% of monthly income")
    else:
        insights.append("✅ **Healthy EMI-to-income ratio**")

    if education == "Graduate":
        insights.append("✅ **Graduate education** — slight positive factor")

    for insight in insights:
        st.markdown(f"- {insight}")


# ──────────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Built with ❤️ using Streamlit & XGBoost | 
        <a href='https://github.com/YOUR_USERNAME/loan-approval-xgboost' target='_blank'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)

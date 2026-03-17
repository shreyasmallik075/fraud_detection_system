import streamlit as st
import numpy as np
import joblib
import pandas as pd
import plotly.express as px


# Page Config
# --------------------

st.set_page_config(
    page_title="Fraud_Guard",
    page_icon="🛡",
    layout="wide"
)


# Load Models
# -----------------------

model = joblib.load("models/fraud_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/type_encoder.pkl")
features = joblib.load("models/feature_cols.pkl")


# Session State for History
# --------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------
# Tabs
# -----------------------

tab1, tab2, tab3 = st.tabs(["🔍 Fraud Detection", "📜 Transaction History", "📊 Analytics"])

# ------------------------------------------
# TAB 1 — Fraud Detection
# ------------------------------------------

with tab1:

    st.title("🛡 Fraud_Guard")
    st.subheader("Real-Time Payment Fraud Detection")

    st.write("Enter transaction details to check fraud risk.")

    # Inputs
    type = st.selectbox(
        "Transaction Type",
        ["PAYMENT","TRANSFER","CASH_OUT","DEBIT","CASH_IN"]
    )

    amount = st.number_input("Amount", min_value=0.0)

    oldbalanceOrg = st.number_input("Sender Balance Before")

    newbalanceOrig = st.number_input("Sender Balance After")

    oldbalanceDest = st.number_input("Receiver Balance Before")

    newbalanceDest = st.number_input("Receiver Balance After")

    # Predict Button
    if st.button("Check Transaction"):

        # Feature engineering
        amount_log = np.log1p(amount)
        bal_ratio = amount / (oldbalanceOrg + 1)

        is_risky_type = 1 if type in ["TRANSFER","CASH_OUT"] else 0

        type_encoded = encoder.transform([type])[0]

        account_drained = 1 if (newbalanceOrig == 0 and oldbalanceOrg > 0) else 0

        receiver_balance_unchanged = 1 if (
            newbalanceDest == oldbalanceDest and amount > 0
        ) else 0

        full_balance_transfer = 1 if amount == oldbalanceOrg else 0

        # Model input
        input_data = np.array([[
            amount,
            amount_log,
            oldbalanceOrg,
            newbalanceOrig,
            bal_ratio,
            is_risky_type,
            type_encoded,
            account_drained,
            receiver_balance_unchanged,
            full_balance_transfer
        ]])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        # Result
        st.write("### Result")

        if prediction == 1:
            st.error("⚠ Fraud Detected")
        else:
            st.success("✅ Transaction Looks Safe")

        # Risk meter
        st.write("### Fraud Risk Score")

        st.progress(float(probability))

        st.write(f"Risk Level: {probability:.2%}")

        # Explanation
        st.write("### Explanation")

        reasons = []

        if account_drained:
            reasons.append("Sender account drained")

        if receiver_balance_unchanged:
            reasons.append("Receiver balance unchanged")

        if full_balance_transfer:
            reasons.append("Full balance transferred")

        if is_risky_type:
            reasons.append("High risk transaction type")

        if reasons:
            for r in reasons:
                st.write("•", r)
        else:
            st.write("No strong fraud indicators")

        # Save to history
        st.session_state.history.append({
            "Type": type,
            "Amount": amount,
            "Fraud Probability": probability,
            "Prediction": "Fraud" if prediction==1 else "Legit"
        })

# ----------------------------------------------
# TAB 2 — Transaction History
# ----------------------------------------------

with tab2:

    st.header("📜 Transaction History")

    if len(st.session_state.history) == 0:

        st.info("No transactions yet.")

    else:

        df_hist = pd.DataFrame(st.session_state.history)

        st.dataframe(df_hist)

        st.write(f"Total Transactions: {len(df_hist)}")

# --------------------------------------------
# TAB 3 — Analytics Dashboard
# --------------------------------------------

with tab3:

    st.header("📊 Fraud Analytics")

    if len(st.session_state.history) == 0:

        st.info("Run some transactions to see analytics.")

    else:

        df = pd.DataFrame(st.session_state.history)

        # Fraud vs Legit chart
        fig1 = px.pie(
            df,
            names="Prediction",
            title="Fraud vs Legit Transactions"
        )

        st.plotly_chart(fig1)

        # Amount distribution
        fig2 = px.histogram(
            df,
            x="Amount",
            color="Prediction",
            title="Transaction Amount Distribution"
        )

        st.plotly_chart(fig2)

        # Risk distribution
        fig3 = px.box(
            df,
            y="Fraud Probability",
            color="Prediction",
            title="Fraud Risk Distribution"
        )

        st.plotly_chart(fig3)
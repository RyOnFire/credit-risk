import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import joblib
from model import CreditRiskNet, predict_with_threshold

st.set_page_config(page_title="Credit Risk Dashboard", layout="wide")
st.title('💳 Credit Risk Dashboard')

logistic_model = joblib.load('logistic_model.pkl')
scaler = joblib.load('scaler.pkl')

nn_model = CreditRiskNet(input_size=10)
nn_model.load_state_dict(torch.load('neural_network_simple.pt'))
nn_model.eval()

EXPECTED_COLS = ['RevolvingUtilizationOfUnsecuredLines', 'age',
                  'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio',
                  'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans',
                  'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
                  'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

st.sidebar.header("Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type="csv")
model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Neural Network"])
threshold = st.sidebar.slider("Default Probability Threshold", 0.1, 0.9, 0.75, 0.05)

st.sidebar.divider()
st.sidebar.subheader("Expected Columns")
for col in EXPECTED_COLS:
    st.sidebar.write(f"✓ {col}")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    missing = [col for col in EXPECTED_COLS if col not in df.columns]

    if missing:
        st.error(f"Missing expected columns: {missing}")
        st.info("This dashboard expects the 'Give Me Some Credit' dataset format. See expected columns in the sidebar.")
        st.stop()

    has_labels = 'SeriousDlqin2yrs' in df.columns
    if has_labels:
        true_labels = df['SeriousDlqin2yrs']

    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(df['NumberOfDependents'].median())

    features = df[EXPECTED_COLS]
    features_scaled = scaler.transform(features)

    if model_choice == "Logistic Regression":
        probabilities = logistic_model.predict_proba(features_scaled)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
    else:
        predictions, probabilities = predict_with_threshold(nn_model, features_scaled, threshold=threshold)

    df['Default_Probability'] = probabilities
    df['Prediction'] = predictions
    df['Prediction'] = df['Prediction'].map({1: 'High Risk', 0: 'Low Risk'})

    total = len(df)
    flagged = int((predictions == 1).sum())
    normal = int((predictions == 0).sum())

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Borrowers", total)
    col2.metric("Flagged High Risk", flagged)
    col3.metric("Flagged Low Risk", normal)

    if has_labels:
        st.caption(f"Actual default rate in this file: {round(true_labels.mean()*100, 2)}%")

    st.divider()

    st.subheader("Risk Distribution")
    fig = px.bar(x=['Low Risk', 'High Risk'], y=[normal, flagged],
                color=['Low Risk', 'High Risk'], title='Borrower Risk Distribution')
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Default Probability Distribution")
    fig2 = px.histogram(df, x='Default_Probability', nbins=50,
                        title='Distribution of Predicted Default Probabilities')
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Flagged High Risk Borrowers")
    flagged_df = df[df['Prediction'] == 'High Risk'].sort_values(
        'Default_Probability', ascending=False)
    st.dataframe(flagged_df)

    st.download_button(
        label="Download Flagged Borrowers",
        data=flagged_df.to_csv(index=False),
        file_name="flagged_borrowers.csv",
        mime="text/csv"
    )

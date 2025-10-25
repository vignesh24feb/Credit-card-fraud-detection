import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import load_model

# === Load model ===
model = load_model("04_creditcard.h5")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Fill in the details below to predict if a transaction is fraudulent.")

# === User Input ===
Time = st.number_input("Time", value=13.0189)
V1 = st.number_input("V1", value=0.686132504)
V2 = st.number_input("V2", value=0.686132504)
V3 = st.number_input("V3", value=0.686132504)
V4 = st.number_input("V4", value=0.686132504)
V5 = st.number_input("V5", value=0.686132504)
V6 = st.number_input("V6", value=0.686132504)
V7 = st.number_input("V7", value=0.686132504)
V8 = st.number_input("V8", value=0.686132504)
V9 = st.number_input("V9", value=0.686132504)
V10 = st.number_input("V10", value=0.686132504)
V11 = st.number_input("V11", value=0.686132504)
V12 = st.number_input("V12", value=0.686132504)
V13 = st.number_input("V13", value=0.686132504)
V14 = st.number_input("V14", value=0.686132504)
V15 = st.number_input("V15", value=0.686132504)
V16 = st.number_input("V16", value=0.686132504)
V17 = st.number_input("V17", value=0.686132504)
V18 = st.number_input("V18", value=0.686132504)
V19 = st.number_input("V19", value=0.686132504)
V20 = st.number_input("V20", value=0.686132504)
V21 = st.number_input("V21", value=0.686132504)
V22 = st.number_input("V22", value=0.686132504)
V23 = st.number_input("V23", value=0.686132504)
V24 = st.number_input("V24", value=0.686132504)
V25 = st.number_input("V25", value=0.686132504)
V26 = st.number_input("V26", value=0.686132504)
V27 = st.number_input("V27", value=0.686132504)
V28 = st.number_input("V28", value=0.686132504)
Amount = st.number_input("Amount", value=13.0189)

# === Prediction ===
if st.button("Predict"):
    # Combine all inputs into a single row (2D array)
    full_input = np.array([[Time, V1, V2, V3, V4, V5, V6, V7, V8, V9,
                            V10, V11, V12, V13, V14, V15, V16, V17, V18, V19,
                            V20, V21, V22, V23, V24, V25, V26, V27, V28, Amount]])
    
    # Optionally scale it (for demo only — ideally load pre-fitted scaler)
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(full_input)

    # Predict
    prediction = model.predict(scaled_input)

    # Interpret prediction
    fraud_prob = prediction[0][0]
    result = "⚠️ Fraudulent Transaction" if fraud_prob > 0.5 else "✅ Legitimate Transaction"

    # Display result
    st.subheader(result)
    st.write(f"Predicted Probability of Fraud: {fraud_prob:.4f}")

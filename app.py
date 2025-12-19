import streamlit as st
import numpy as np
import joblib

# Load model, label encoder, and scaler
model = joblib.load('crop_recommendation_xgb_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

# Page config
st.set_page_config(
    page_title="Smart Crop Recommender",
    page_icon="🌾",
    layout="centered",
)

# Header
st.markdown("""
    <h1 style="text-align: center; color: #2E7D32;">🌾 Smart Crop Recommender</h1>
    <p style="text-align: center; font-size: 18px;">Get crop recommendations based on soil and climate data</p>
""", unsafe_allow_html=True)

# Input form
with st.form("crop_form"):
    col1, col2 = st.columns(2)

    with col1:
        N = st.number_input("Nitrogen (N) mg/kg", min_value=0, max_value=200, value=50)
        K = st.number_input("Potassium (K) mg/kg", min_value=0, max_value=200, value=50)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=25.0)

    with col2:
        P = st.number_input("Phosphorus (P) mg/kg", min_value=0, max_value=200, value=50)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)

    rainfall = st.slider("Rainfall (mm)", 0.0, 500.0, 100.0)

    submitted = st.form_submit_button("🌱 Predict Best Crop")

# Predict
if submitted:
    input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

    # ⚠️ Apply the same scaling used during training
    input_scaled = scaler.transform(input_data)

    # Get prediction probabilities
    probs = model.predict_proba(input_scaled)[0]

    # Get top 3 crop indices
    top3_idx = np.argsort(probs)[-3:][::-1]

    # Decode class indices to actual crop names
    top3_crops = label_encoder.inverse_transform(top3_idx)
    top3_probs = probs[top3_idx]

    # Display results
    st.success("✅ Top 3 Recommended Crops:")
    for crop, prob in zip(top3_crops, top3_probs):
        st.markdown(f"- 🌱 **{crop.upper()}** — `{prob:.2%}` probability")

    st.balloons()

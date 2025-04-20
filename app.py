# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestRegressor

# --- Load trained models and assets ---

# Load the RandomForest model
rf_model = joblib.load('AqariLens_rf_model.pkl')

# Load the sentence-transformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
st_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
st_model = st_model.to(device)

# Load dropdown options
unique_cities = joblib.load('unique_cities.pkl')
unique_locations = joblib.load('unique_locations.pkl')
unique_neighborhoods = joblib.load('unique_neighborhoods.pkl')

# Validation R2 score from training
r2_score_validation = 0.62  # Update if needed

# --- Streamlit Page Config ---
st.set_page_config(page_title="AqariLens | Real Estate AI", layout="centered")

st.title("🏡 AqariLens")
st.subheader("🔎 See Real Estate Through the Power of AI")
st.markdown("---")

# --- Input Form ---
with st.form("prediction_form"):
    st.write("### 📝 Enter Property Details")
    
    city = st.selectbox("Select City", options=unique_cities)
    location = st.selectbox("Select Location", options=unique_locations)
    neighborhood = st.selectbox("Select Neighborhood", options=unique_neighborhoods)
    
    bedrooms = st.number_input("Number of Bedrooms", min_value=0, step=1)
    bathrooms = st.number_input("Number of Bathrooms", min_value=0, step=1)
    area = st.number_input("Area (m²)", min_value=1, step=1)
    
    submit = st.form_submit_button("Predict Price")

# --- Prediction Logic ---
if submit:
    with st.spinner('Predicting... Please wait.'):
        
        # Encode text features
        city_emb = st_model.encode([city], device=device)
        location_emb = st_model.encode([location], device=device)
        neighborhood_emb = st_model.encode([neighborhood], device=device)
        
        city_mean = np.mean(city_emb)
        location_mean = np.mean(location_emb)
        neighborhood_mean = np.mean(neighborhood_emb)
        
        # Preprocess numeric features
        area_log = np.log1p(area)
        bedrooms_log = np.log1p(bedrooms)
        bathrooms_log = np.log1p(bathrooms)
        math_log = np.log1p(area / (bedrooms + bathrooms + 1e-9))  # prevent division by zero
        
        # Assemble feature vector
        input_features = np.array([[bedrooms_log, bathrooms_log, area_log, math_log,
                                    neighborhood_mean, city_mean, location_mean]])
        
        # Predict log(price)
        price_log = rf_model.predict(input_features)[0]
        predicted_price = np.expm1(price_log)  # Reverse log1p
        
        # Confidence estimation (+/-10%)
        lower_bound = predicted_price * 0.9
        upper_bound = predicted_price * 1.1
        
        # --- Show Results ---
        st.success(f"💰 Estimated Property Price: {predicted_price:,.2f} EGP")
        st.info(f"📈 Confidence Range: {lower_bound:,.2f} EGP — {upper_bound:,.2f} EGP")
        st.info(f"✅ Model Confidence: {r2_score_validation * 100:.1f}% based on validation")

st.markdown("---")
st.caption("Built by Mohamed Gad | Powered by Machine Learning and Semantic Intelligence.")

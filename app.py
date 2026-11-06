import streamlit as st
import pandas as pd
import joblib

# Load the saved files
model = joblib.load("models/disease_model.pkl") 
scaler = joblib.load("models/scaler.pkl")
feature_cols = joblib.load("models/feature_columns.pkl")
le = joblib.load("models/label_encoder.pkl")
best_model_name = joblib.load("models/best_model_name.pkl")

st.set_page_config(page_title="Smart Crop AI", page_icon="🌾")

st.title("🌾 Smart Crop Recommendation System")
st.write("Enter the soil and weather conditions below to get an AI crop recommendation.")

# Create input fields with realistic default values for a better demo
inputs = {}
for col in feature_cols:
    if col in ['N', 'P', 'K']:
        val = 50.0
    elif col == 'temperature':
        val = 25.0
    elif col == 'humidity':
        val = 80.0
    elif col == 'ph':
        val = 6.5
    elif col == 'rainfall':
        val = 200.0
    else:
        val = 1.0
        
    inputs[col] = st.number_input(f"{col}", value=val, step=1.0)

if st.button("Predict Crop"):
    # Convert inputs to DataFrame
    row = pd.DataFrame([inputs])[feature_cols]
    
    # Scale the inputs
    scaled = scaler.transform(row)
    
    # Predict
    pred = model.predict(scaled)[0]
    
    # If the best model was XGBoost, decode the number back to the crop name
    if best_model_name == "XGBoost":
        pred_text = le.inverse_transform([int(pred)])[0]
    else:
        pred_text = pred  # Random Forest predicts text directly
        
    st.success(f"✅ Recommended Crop: **{pred_text.capitalize()}**")
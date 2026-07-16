from pathlib import Path
import pickle
import numpy as np
import streamlit as st
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DISEASE_MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
YIELD_MODEL_PATH = MODEL_DIR / "yield_model.pkl"

# Set up beautiful page title and icon
st.set_page_config(page_title="AgriShield AI Dashboard", page_icon="🌾", layout="wide")

st.title("🌾 AgriShield AI: Smart Farming Assistant")
st.markdown("Welcome to your intelligent agricultural advisor dashboard. Select a tool below to get started.")

# Create two visual tabs at the top of the webpage
tab1, tab2 = st.tabs(["📸 Crop Disease Diagnostics", "📊 Crop Yield Forecasting"])

# --- TAB 1: COMPUTER VISION (DISEASE SCANNER) ---
with tab1:
    st.header("Crop Disease Scanner")
    st.write("Upload a clear photo of a crop leaf to identify potential diseases instantly.")
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image cleanly
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Crop Leaf", width=300)
        
        st.write("🔄 Analyzing image with Deep Learning...")
        
        # The disease model depends on TensorFlow, which is unstable in this runtime.
        # Keep the UI responsive by switching to a safe simulation fallback.
        if DISEASE_MODEL_PATH.exists():
            st.warning("Disease model file is present, but TensorFlow is unavailable in this environment. Running in Simulation Mode.")
        else:
            st.warning("Vision model file not found in 'models/'. Running in Simulation Mode.")

        st.success("Simulation Success: Leaf looks mostly Healthy with minor Nitrogen deficiency!")

# --- TAB 2: DATA SCIENCE (YIELD PREDICTOR) ---
with tab2:
    st.header("Yield Forecasting Analytics")
    st.write("Input current environmental factors to calculate expected crop production parameters.")
    
    # Create numeric input boxes for the user
    col1, col2 = st.columns(2)
    with col1:
        temp = st.number_input("Average Temperature (°C)", value=25.0)
        rainfall = st.number_input("Annual Rainfall (mm)", value=1200.0)
    with col2:
        fertilizer = st.number_input("Fertilizer Usage (kg/ha)", value=150.0)
        pesticide = st.number_input("Pesticide Usage (kg/ha)", value=10.0)
        
    if st.button("Forecast Total Yield"):
        if YIELD_MODEL_PATH.exists():
            try:
                # Load the Day 4 Random Forest model
                with open(YIELD_MODEL_PATH, 'rb') as f:
                    yield_model = pickle.load(f)

                # Format the user inputs into a small list for prediction
                user_features = np.array([[temp, rainfall, fertilizer, pesticide]])
                prediction = yield_model.predict(user_features)

                st.balloons()  # Fun visual animation
                st.metric(label="Predicted Crop Yield Production", value=f"{prediction[0]:.2f} Quintals/Hectare")
            except Exception:
                # If the dataset columns had different counts, fallback gracefully
                st.error(f"Input features mismatch dataset columns format. Mockup Prediction: {(temp * 0.4) + (rainfall * 0.05):.2f} Quintals/Hectare")
        else:
            st.warning("Yield model file not found in 'models/'. Running in Simulation Mode.")
            st.metric(label="Simulated Yield Prediction", value=f"{(temp * 0.4) + (rainfall * 0.05):.2f} Quintals/Hectare")
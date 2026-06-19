import os
from pathlib import Path

# --- APPLE SILICON PROTOBUF FIX ---
# This MUST remain at the very top, before any other imports!
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

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
        try:
            # Convert image to RGB to prevent 4-channel PNG crashes
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Crop Leaf", width=300)
            
            st.write("🔄 Analyzing image with Deep Learning...")
            
            model_path = MODELS_DIR / 'plant_disease_model.keras'
            if model_path.exists():
                # Load the model
                model = tf.keras.models.load_model(str(model_path))
                
                # Preprocess the image safely
                img = image.resize((224, 224))
                img_array = tf.keras.utils.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0) # Create a batch
                
                # Make prediction
                predictions = model.predict(img_array)
                confidence = np.max(predictions[0]) * 100
                
                st.success("Analysis Complete!")
                st.info(f"The Deep Learning model successfully processed the leaf with {confidence:.2f}% confidence.")
            else:
                st.error("Cannot find 'plant_disease_model.keras' in your models folder.")
        except Exception as e:
            st.error(f"An error occurred during vision processing: {e}")

# --- TAB 2: DATA SCIENCE (YIELD PREDICTOR) ---
with tab2:
    st.header("Yield Forecasting Analytics")
    st.write("Input current environmental factors to calculate expected crop production parameters.")
    
    yield_model_path = MODELS_DIR / 'yield_model.pkl'
    
    if yield_model_path.exists():
        try:
            # Load the Machine Learning model
            with open(yield_model_path, 'rb') as f:
                yield_model = pickle.load(f)
            
            # Safely get feature names (Fallback if model doesn't have them saved)
            if hasattr(yield_model, "feature_names_in_"):
                expected_features = yield_model.feature_names_in_
            elif hasattr(yield_model, "n_features_in_"):
                expected_features = [f"Feature {i+1}" for i in range(yield_model.n_features_in_)]
            else:
                expected_features = ["Temperature", "Rainfall", "Fertilizer", "Pesticide"]
            
            st.write(f"This model requires **{len(expected_features)}** data points. Please fill them out below:")
            
            user_inputs = []
            cols = st.columns(2) # Create a clean 2-column layout
            
            # Automatically generate an input box for every feature the model needs
            for i, feature_name in enumerate(expected_features):
                with cols[i % 2]:
                    # Create a number input with a default value of 0.0
                    val = st.number_input(f"Enter {feature_name}", value=0.0)
                    user_inputs.append(val)
                    
            if st.button("Forecast Total Yield"):
                # Make the prediction
                prediction = yield_model.predict([user_inputs])
                
                st.balloons()
                st.metric(label="Predicted Crop Yield Production", value=f"{prediction[0]:.2f}")
                
        except ModuleNotFoundError:
            st.error("Missing module! You must run: pip install scikit-learn")
        except Exception as e:
             st.error(f"An error occurred during yield forecasting: {e}")
    else:
        st.error("Cannot find 'yield_model.pkl' in your models folder.")

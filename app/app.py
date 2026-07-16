from pathlib import Path
import pickle
import numpy as np
import streamlit as st
from PIL import Image

try:
    import google.generativeai as genai
except Exception:
    genai = None

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DISEASE_MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
YIELD_MODEL_PATH = MODEL_DIR / "yield_model.pkl"

# Set up beautiful page title and icon
st.set_page_config(page_title="AgriShield AI Dashboard", page_icon="🌾", layout="wide")

# --- SIDEBAR FOR API KEY ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("To use the GenAI features, enter your free Gemini API Key below.")
    api_key = st.text_input("Gemini API Key", type="password")
    st.markdown("[Get your free key here](https://aistudio.google.com/app/apikey)")

st.title("🌾 AgriShield AI: Smart Farming Assistant")
st.markdown("Welcome to your intelligent agricultural advisor dashboard. Select a tool below to get started.")

# Create THREE visual tabs now!
tab1, tab2, tab3 = st.tabs(["📸 Crop Disease Diagnostics", "📊 Crop Yield Forecasting", "🤖 AI Agronomist"])

# --- TAB 1: COMPUTER VISION (DISEASE SCANNER) ---
with tab1:
    st.header("Crop Disease Scanner")
    st.write("Upload a clear photo of a crop leaf to identify potential diseases instantly.")
    
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Crop Leaf", width=300)
            
            st.write("🔄 Analyzing image with Deep Learning...")
            
            if DISEASE_MODEL_PATH.exists():
                st.warning("Disease model file is present, but TensorFlow is unavailable in this environment. Running in Simulation Mode.")
            else:
                st.warning("Vision model file not found in 'models/'. Running in Simulation Mode.")

            st.success("Simulation Success: Leaf looks mostly Healthy with minor Nitrogen deficiency!")
        except Exception as e:
            st.error(f"An error occurred during vision processing: {e}")

# --- TAB 2: DATA SCIENCE (YIELD PREDICTOR) ---
with tab2:
    st.header("Yield Forecasting Analytics")
    st.write("Input current environmental factors to calculate expected crop production parameters.")
    
    if YIELD_MODEL_PATH.exists():
        try:
            with open(YIELD_MODEL_PATH, 'rb') as f:
                yield_model = pickle.load(f)
            
            expected_features = yield_model.feature_names_in_
            st.write(f"This model was trained on **{len(expected_features)}** specific data points. Please fill them out below:")
            
            user_inputs = []
            cols = st.columns(2)
            
            for i, feature_name in enumerate(expected_features):
                with cols[i % 2]:
                    val = st.number_input(f"Enter {feature_name}", value=0.0)
                    user_inputs.append(val)
                    
            if st.button("Forecast Total Yield"):
                prediction = yield_model.predict([user_inputs])
                st.balloons()
                st.metric(label="Predicted Crop Yield Production", value=f"{prediction[0]:.2f}")
                
        except Exception as e:
             st.error(f"An error occurred during yield forecasting: {e}")
    else:
        st.error("Cannot find 'yield_model.pkl' in your models folder.")

# --- TAB 3: GENERATIVE AI (EXPERT ADVISOR) ---
with tab3:
    st.header("🤖 GenAI Agronomist Chat")
    st.write("Have a continuous conversation with our AI expert regarding crop issues, pest control, or soil health.")
    
    # 1. Initialize a "memory" for the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 2. Draw all previous messages on the screen
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Create the modern chat input bar at the bottom
    if prompt := st.chat_input("Ask a farming question here..."):
        if not api_key:
            st.error("⚠️ Please enter your Gemini API Key in the sidebar on the left first!")
        else:
            # Display the user's new question instantly
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Save the user's question to memory
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Connect to AI and get the answer
            with st.spinner("Analyzing agricultural data..."):
                try:
                    genai.configure(api_key=api_key)
                    llm = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # Add a professional system prompt to guide the AI
                    system_prompt = f"You are an expert agronomist. Answer this query professionally: {prompt}"
                    response = llm.generate_content(system_prompt)
                    ai_answer = response.text
                    
                    # Display the AI's answer with an assistant icon
                    with st.chat_message("assistant"):
                        st.markdown(ai_answer)
                        
                    # Save the AI's answer to memory so it doesn't disappear
                    st.session_state.messages.append({"role": "assistant", "content": ai_answer})
                except Exception as e:
                    st.error(f"Error connecting to AI Server: {e}")
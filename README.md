# 🌾 AgriShield AI: Multimodal Farming Assistant



**Status:** ✅ Completed | 🚀 Deployed on Streamlit Cloud | 🛡️ CI/CD Tested

## Project Overview



An end-to-end multi-modal Data Science pipeline integrating predictive analytics with Deep Learning and Generative AI to assist in agricultural decision-making.

## Features
* **Crop Disease Scanner:** A Deep Learning vision model (MobileNetV2) that detects plant diseases from leaf images.
* **Yield Forecasting:** A Random Forest regression model that predicts crop yield based on environmental factors (temperature, rainfall, fertilizer use).

## Tech Stack
* Python, TensorFlow, Scikit-Learn, Pandas, Streamlit

## 🚀 How to Run Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app/app.py`

## 🏗️ System Architecture

```mermaid
graph TD;
    User[User Interface] -->|Image Upload| UI[Streamlit Frontend Tab 1];
    User -->|Sensor Data| U2[Streamlit Frontend Tab 2];
    User -->|Text Query| U3[Streamlit Frontend Tab 3];
    
    UI --> Vision[MobileNetV2 Model];
    U2 --> Tabular[Random Forest Regressor];
    U3 --> LLM[Google Gemini API];
    
    Vision --> Results[Diagnostics Output];
    Tabular --> Results;
    LLM --> Results;
    
    Results --> Cloud[Streamlit Cloud Deployment];

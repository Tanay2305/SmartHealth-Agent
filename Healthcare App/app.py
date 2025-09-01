import streamlit as st
import joblib
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load .env for Gemini key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    # print("✅ Gemini API configured successfully.")

# Paths
MODELS_DIR = "models"  # where your .joblib models are stored

# System prompt for medical agent
MEDICAL_AGENT_PROMPT = """
You are a medical assistant AI. Answer concisely in 3-5 sentences.

Rules:
- Give short, clear advice.
- Only mention specific medicine names if it is clearly appropriate for the described symptoms, and avoid repeating the same medicine in every response.
- If medicine is not essential, suggest general care methods (rest, hydration, diet, monitoring).
- Always include 1 sentence that this is not a diagnosis and to consult a doctor if symptoms persist or worsen.
- Avoid long explanations unless the user explicitly asks for details.
"""


# UI Setup
st.set_page_config(page_title="SmartHealth Agent", page_icon="🩺")
st.title("🩺 SmartHealth Agent")
st.markdown("Predict NCD risks or analyze general symptoms.")

# API mode selection
api_mode = st.sidebar.selectbox("Select Mode", ["Local ML Models", "Gemini API"])
mode = st.radio("Choose Function", ["NCD Prediction", "General Symptom Check"])





diseases = {
    "Thyroid": [
        "Thyroid Stimulating Hormone (TSH)",
        "Total Thyroxine (TT4)",
        "Free Thyroxine Index (FTI)",
        "On Thyroxine (on_thyroxine)",
        "Thyroxine Uptake (T4U)",
        "Triiodothyronine (T3)",
        "Age"
    ],
    "CKD": [
        "Specific Gravity (sg)",
        "Hemoglobin (hemo)",
        "Serum Creatinine (sc)",
        "Albumin (al)",
        "Packed Cell Volume (pcv)"
    ],
    "Diabetes": [
        "Glucose ",
        "Blood Pressure",
        "Skin Thickness ",
        "Insulin",
        "Body Mass Index (BMI)",
        "Diabetes Pedigree Function ",
        "Age (Age)",
        "Pregnancies"
    ],
    "Liver Disease": [
        "Alkaline Phosphatase",
        "Aspartate Aminotransferase ",
        "Age",
        "Alamine Aminotransferase ",
        "Total Bilirubin",
        "Albumin",
        "Total Proteins"
    ],
    "Stroke": [
        "Average Glucose Level",
        "Body Mass Index (bmi)",
        "Age",
        "Smoking Status",
        "Work Type",
        "Residence Type"
    ]

}


# NCD Mode
if mode == "NCD Prediction":
    disease = st.selectbox("Disease", list(diseases.keys()))
    inputs = {}
    for feature in diseases[disease]:
        if feature.lower() in ["yes", "no", "on_thyroxine", "smoking_status", "work_type", "residence_type"]:
            inputs[feature] = st.selectbox(feature, ["Yes", "No"]) if feature.lower() in ["yes", "no", "on_thyroxine"] else st.text_input(feature)
        else:
            inputs[feature] = st.number_input(feature)
    
    if st.button("Predict"):
        if api_mode == "Local ML Models":
            model_path = os.path.join(MODELS_DIR, f"{disease.lower().replace(' ', '_')}_model.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                feature_values = [
                    1 if str(inputs[f]) == "Yes" else 0 if str(inputs[f]) == "No" else inputs[f]
                    for f in diseases[disease]
                ]
                prob = model.predict_proba([feature_values])[0][1]
                st.success(f"Probability of {disease}: {prob*100:.2f}%")
            else:
                st.error(f"Model for {disease} not found.")
        else:
            st.warning("Gemini mode is for free-text symptom analysis.")

# General Mode
elif mode == "General Symptom Check":
    symptoms_text = st.text_area("Describe your symptoms or ask a health question:")

    if st.button("Check Symptoms"):
        if not GEMINI_API_KEY:
            st.error("Gemini API Key not set.")
        else:
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = f"{MEDICAL_AGENT_PROMPT}\nUser: {symptoms_text}"
            try:
                response = model.generate_content(prompt)
                st.markdown("### 💬 Suggested Advice:")
                st.write(response.text)
            except Exception as e:
                st.error(f"Error: {e}")




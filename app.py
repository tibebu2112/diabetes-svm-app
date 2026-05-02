import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_model():
    model  = joblib.load("svm_diabetes_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

svm_model, scaler = load_model()

smoking_options = ["No Info", "current", "ever",
                   "former", "never", "not current"]
le_smoking = LabelEncoder()
le_smoking.fit(smoking_options)

# PASTE YOUR top_features list from Colab output here
top_features = ["blood_glucose_level", "HbA1c_level", "bmi",
                "age", "hypertension", "heart_disease"]

st.title("🏥 Diabetes Prediction System")
st.markdown("### Powered by Support Vector Machine (SVM)")
st.markdown("Fill in the patient details and click **Predict**.")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Information")
    gender        = st.selectbox("Gender", ["Female", "Male"])
    age           = st.slider("Age (years)", 0, 100, 45)
    hypertension  = st.radio("Hypertension", [0, 1],
                             format_func=lambda x: "Yes" if x==1 else "No",
                             horizontal=True)
    heart_disease = st.radio("Heart Disease", [0, 1],
                             format_func=lambda x: "Yes" if x==1 else "No",
                             horizontal=True)

with col2:
    st.subheader("Medical Measurements")
    smoking_history     = st.selectbox("Smoking History", smoking_options)
    bmi                 = st.slider("BMI", 10.0, 100.0, 27.0, 0.1)
    HbA1c_level         = st.slider("HbA1c Level (%)", 3.5, 15.0, 5.5, 0.1)
    blood_glucose_level = st.slider("Blood Glucose (mg/dL)", 50, 400, 120)

st.divider()

if st.button("🔍 Predict Diabetes", use_container_width=True, type="primary"):

    gender_enc  = 0 if gender == "Female" else 1
    smoking_enc = le_smoking.transform([smoking_history])[0]

    all_features = {
        "gender":              gender_enc,
        "age":                 float(age),
        "hypertension":        int(hypertension),
        "heart_disease":       int(heart_disease),
        "smoking_history":     smoking_enc,
        "bmi":                 float(bmi),
        "HbA1c_level":         float(HbA1c_level),
        "blood_glucose_level": float(blood_glucose_level),
    }

    input_values = [[all_features[f] for f in top_features]]
    input_scaled = scaler.transform(input_values)

    prediction  = svm_model.predict(input_scaled)[0]
    probability = svm_model.predict_proba(input_scaled)[0]
    confidence  = probability[prediction] * 100

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️  DIABETES DETECTED — Confidence: {confidence:.1f}%")
        st.warning("**Recommendation:** Refer the patient for further "
                   "clinical evaluation and blood tests.")
    else:
        st.success(f"✅  NO DIABETES DETECTED — Confidence: {confidence:.1f}%")
        st.info("**Recommendation:** Maintain a healthy lifestyle. "
                "Schedule routine annual check-ups.")

    with st.expander("📋 View Patient Input Summary"):
        st.write({
            "Gender": gender,
            "Age": age,
            "Hypertension": "Yes" if hypertension==1 else "No",
            "Heart Disease": "Yes" if heart_disease==1 else "No",
            "Smoking History": smoking_history,
            "BMI": bmi,
            "HbA1c Level": HbA1c_level,
            "Blood Glucose": blood_glucose_level
        })

st.divider()
st.caption("⚕️ Disclaimer: This tool assists medical professionals only. "
           "It does not replace clinical diagnosis.")

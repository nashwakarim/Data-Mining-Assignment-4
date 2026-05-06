
# app.py — DS3002 Assignment 4, Part E
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import json
import shap

# ── Load model & config ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("xgb_heart_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("feature_config.json") as f:
        cfg = json.load(f)
    return model, scaler, cfg

model, scaler, cfg = load_artifacts()
cont_cols = cfg["cont_cols"]
cat_cols  = cfg["cat_cols"]
feat_names = cfg["feat_names"]

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(page_title="CardioAI Heart Disease Predictor",
                   page_icon="❤️", layout="wide")
st.title("❤️ CardioAI Heart Disease Risk Predictor")
st.markdown("*Decision support tool for community cardiologists — powered by XGBoost + SHAP*")
st.divider()

# ── E1: Input Form ───────────────────────────────────────────────────
st.header("Patient Input Form")
st.caption("Pre-populated with a real test patient. Edit values and click Predict.")

col1, col2, col3 = st.columns(3)

with col1:
    age      = st.number_input("Age (years) [20–80]",       min_value=20,  max_value=80,  value=63)
    sex      = st.selectbox("Sex",       [0, 1], format_func=lambda x: "Female" if x==0 else "Male", index=1)
    cp       = st.selectbox("Chest Pain Type (0=typical angina, 1=atypical, 2=non-anginal, 3=asymptomatic)", [0,1,2,3], index=0)
    trestbps = st.number_input("Resting BP (mmHg) [80–200]",  min_value=80,  max_value=200, value=145)
    chol     = st.number_input("Cholesterol (mg/dl) [100–600]",min_value=100, max_value=600, value=233)

with col2:
    fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0,1], format_func=lambda x: "No" if x==0 else "Yes", index=1)
    restecg  = st.selectbox("Resting ECG (0=normal, 1=ST abnormality, 2=LV hypertrophy)", [0,1,2], index=2)
    thalach  = st.number_input("Max Heart Rate [70–210]",      min_value=70,  max_value=210, value=150)
    exang    = st.selectbox("Exercise-Induced Angina",         [0,1], format_func=lambda x: "No" if x==0 else "Yes", index=0)
    oldpeak  = st.number_input("ST Depression (oldpeak) [0.0–6.0]", min_value=0.0, max_value=6.0, value=2.3, step=0.1)

with col3:
    slope    = st.selectbox("ST Slope (0=upsloping, 1=flat, 2=downsloping)", [0,1,2], index=2)
    ca       = st.number_input("Major Vessels (fluoroscopy) [0–3]", min_value=0, max_value=3, value=0)
    thal     = st.selectbox("Thalassemia (1=normal, 2=fixed defect, 3=reversible defect)", [1,2,3], index=2)

predict_btn = st.button("🔮 Predict Heart Disease Risk", use_container_width=True, type="primary")

# ── E2: Results Panel ─────────────────────────────────────────────────
if predict_btn:
    raw_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }
    raw_df = pd.DataFrame([raw_dict])

    # One-hot encode
    encoded = pd.get_dummies(raw_df, columns=cat_cols, drop_first=False)

    # Align columns with training feature names
    for c in feat_names:
        if c not in encoded.columns:
            encoded[c] = 0
    encoded = encoded[feat_names]

    # Standardise continuous columns
    encoded[cont_cols] = scaler.transform(encoded[cont_cols])

    X_input = encoded.values
    prob    = model.predict_proba(X_input)[0][1]
    pred    = int(prob >= 0.5)

    st.divider()
    st.header("Prediction Result")

    r1, r2 = st.columns(2)
    with r1:
        if pred == 1:
            st.error(f"🔴 **HEART DISEASE DETECTED** — Confidence: {prob*100:.1f}%")
        else:
            st.success(f"🟢 **NO DISEASE DETECTED** — Confidence: {(1-prob)*100:.1f}%")
        st.metric("Risk Score", f"{prob*100:.1f}%", delta="above 50% = disease" if pred==1 else "below 50% = healthy")

    # SHAP waterfall
    with r2:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_input)[0]   # shape: (n_features,)
        top3_idx = np.argsort(np.abs(sv))[-3:][::-1]

        fig_shap, ax_shap = plt.subplots(figsize=(6, 3))
        colors = ["#e53935" if sv[i] > 0 else "#1e88e5" for i in top3_idx]
        ax_shap.barh(
            [feat_names[i] for i in top3_idx],
            [sv[i] for i in top3_idx],
            color=colors
        )
        ax_shap.set_title("Top 3 Features Driving Prediction (SHAP)", fontsize=11)
        ax_shap.set_xlabel("SHAP Value (positive = increases disease risk)")
        ax_shap.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig_shap)

    st.divider()
    # Plain-English explanation
    top_feat = feat_names[top3_idx[0]]
    shap_dir = "increased" if sv[top3_idx[0]] > 0 else "reduced"
    st.info(
        f"**Clinical Summary for the Nurse:** "
        f"The model predicts {'elevated cardiac risk' if pred==1 else 'low cardiac risk'} "
        f"with {prob*100:.1f}% confidence. "
        f"The most influential factor is **{top_feat}**, which has {shap_dir} the risk score. "
        f"Additional significant factors include {feat_names[top3_idx[1]]} and {feat_names[top3_idx[2]]}. "
        f"{'This patient should be referred for further cardiac evaluation.' if pred==1 else 'Routine follow-up is recommended.'}"
    )

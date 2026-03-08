import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical Insurance Predictor",
    page_icon="🏥",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4f8; }
    .stApp { font-family: 'Segoe UI', sans-serif; }
    .title-block {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        padding: 2rem 2rem 1.5rem;
        border-radius: 14px;
        color: white;
        text-align: center;
        margin-bottom: 1.8rem;
    }
    .title-block h1 { font-size: 2rem; margin: 0; }
    .title-block p  { font-size: 1rem; opacity: 0.85; margin: 0.4rem 0 0; }
    .result-box {
        background: linear-gradient(135deg, #e8f5e9, #c8e6c9);
        border-left: 6px solid #2e7d32;
        border-radius: 10px;
        padding: 1.4rem 1.8rem;
        margin-top: 1.5rem;
    }
    .result-box h2 { color: #1b5e20; margin: 0 0 0.3rem; font-size: 1.1rem; }
    .result-box .charge { font-size: 2.4rem; font-weight: 700; color: #2e7d32; }
    .info-card {
        background: #fff;
        border-radius: 10px;
        padding: 1rem 1.4rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #1a73e8, #0d47a1);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2.5rem;
        font-size: 1.05rem;
        font-weight: 600;
        width: 100%;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover { opacity: 0.88; }
    .stSlider > div { padding-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🏥 Medical Insurance Charges Predictor</h1>
    <p>Enter your details below to estimate your annual insurance cost</p>
</div>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = "insurance_model.joblib"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"**Model file not found!**\n\n"
        f"Please place `insurance_model.joblib` in the same directory as this script.\n\n"
        f"Expected path: `{os.path.abspath(MODEL_PATH)}`"
    )
    st.stop()

model = load_model(MODEL_PATH)

# ── Input form ────────────────────────────────────────────────────────────────
st.subheader("👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=100, value=30, step=1,
                    help="Your current age in years")

    sex_label = st.radio("Sex", options=["Female", "Male"], horizontal=True)
    sex = 0 if sex_label == "Female" else 1

with col2:
    bmi = st.number_input(
        "BMI (Body Mass Index)",
        min_value=10.0, max_value=60.0, value=25.0, step=0.1,
        format="%.1f",
        help="Weight (kg) / Height² (m²)"
    )

    children = st.selectbox(
        "Number of Children / Dependents",
        options=[0, 1, 2, 3, 4, 5],
        index=0
    )

smoker_label = st.radio(
    "Smoker?", options=["No", "Yes"], horizontal=True,
    help="Whether you currently smoke"
)
smoker = 1 if smoker_label == "Yes" else 0

# ── BMI category helper ───────────────────────────────────────────────────────
def bmi_category(b):
    if b < 18.5:   return "Underweight", "🔵"
    elif b < 25.0: return "Normal",      "🟢"
    elif b < 30.0: return "Overweight",  "🟡"
    else:          return "Obese",       "🔴"

cat, icon = bmi_category(bmi)
st.caption(f"BMI Category: {icon} **{cat}**")

st.divider()

# ── Prediction ────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Insurance Charges"):
    features = np.array([[age, sex, bmi, children, smoker]])
    prediction = model.predict(features)[0]
    prediction = max(prediction, 0)   # floor at 0

    st.markdown(f"""
    <div class="result-box">
        <h2>Estimated Annual Insurance Charges</h2>
        <div class="charge">${prediction:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    # Summary card
    st.markdown(f"""
    <div class="info-card">
        <strong>Input Summary</strong><br>
        Age: <b>{age}</b> &nbsp;|&nbsp;
        Sex: <b>{sex_label}</b> &nbsp;|&nbsp;
        BMI: <b>{bmi:.1f}</b> ({cat}) &nbsp;|&nbsp;
        Children: <b>{children}</b> &nbsp;|&nbsp;
        Smoker: <b>{smoker_label}</b>
    </div>
    """, unsafe_allow_html=True)

    # Contextual note
    if smoker == 1:
        st.warning("⚠️ Smoking is one of the strongest predictors of higher insurance costs.")
    if bmi >= 30:
        st.info("ℹ️ A higher BMI can increase insurance charges. Consider consulting a healthcare provider.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: Linear Regression trained on medical insurance data · For educational purposes only.")

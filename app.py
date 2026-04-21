import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0b1120;
    color: #e8e4d9;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif;
}

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0f2240 0%, #1a3a5c 60%, #0b1120 100%);
    border: 1px solid #2a4a6b;
    border-radius: 12px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-size: 2.4rem;
    color: #f0e6c8;
    margin-bottom: 0.3rem;
}
.hero p {
    color: #8aa8c8;
    font-size: 1rem;
    margin: 0;
}

/* Input card */
.card {
    background: #111c2e;
    border: 1px solid #1e3550;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    color: #c8b87a;
    font-size: 1.1rem;
    margin-bottom: 1rem;
    border-bottom: 1px solid #1e3550;
    padding-bottom: 0.5rem;
}

/* Prediction box */
.pred-survived {
    background: linear-gradient(135deg, #0d3320, #145c30);
    border: 1px solid #27a84e;
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
}
.pred-died {
    background: linear-gradient(135deg, #3a0d0d, #5c1414);
    border: 1px solid #a82727;
    border-radius: 12px;
    padding: 1.8rem;
    text-align: center;
}
.pred-label {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    margin-bottom: 0.3rem;
}
.pred-prob {
    font-size: 2.6rem;
    font-weight: 700;
    margin: 0.5rem 0;
}
.pred-note {
    font-size: 0.85rem;
    opacity: 0.7;
}

/* Streamlit widget overrides */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label {
    color: #8aa8c8 !important;
    font-size: 0.9rem !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(90deg, #c8943a, #e0b85a);
    color: #0b1120;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 2.5rem;
    width: 100%;
    cursor: pointer;
    transition: opacity 0.2s;
}
div[data-testid="stButton"] button:hover {
    opacity: 0.85;
}
</style>
""", unsafe_allow_html=True)

# ── Load artefacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model       = joblib.load("Logistic.pkl")
    transformer = joblib.load("transformer.pkl")
    columns     = joblib.load("columns.pkl")
    return model, transformer, columns

try:
    model, transformer, columns = load_artifacts()
    artifacts_loaded = True
except FileNotFoundError as e:
    artifacts_loaded = False
    missing = str(e)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🚢 Titanic Survival Predictor</h1>
  <p>Logistic Regression · April 15, 1912 · North Atlantic Ocean</p>
</div>
""", unsafe_allow_html=True)

if not artifacts_loaded:
    st.error(f"Could not load model files. Make sure **Logistic.pkl**, **transformer.pkl**, and **columns.pkl** are in the same directory as this app.\n\n`{missing}`")
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="card"><div class="card-title">👤 Passenger Details</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", options=[1, 2, 3],
                          format_func=lambda x: {1:"1st — Upper", 2:"2nd — Middle", 3:"3rd — Lower"}[x])
    sex    = st.selectbox("Sex", options=["male", "female"])
    age    = st.slider("Age", min_value=1, max_value=80, value=28)

with col2:
    sibsp    = st.number_input("Siblings / Spouses aboard", min_value=0, max_value=8, value=0)
    parch    = st.number_input("Parents / Children aboard",  min_value=0, max_value=6, value=0)
    fare     = st.number_input("Fare paid (£)", min_value=0.0, max_value=520.0, value=32.2, step=0.5)
    embarked = st.selectbox("Port of Embarkation",
                            options=["S", "C", "Q"],
                            format_func=lambda x: {"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])

st.markdown('</div>', unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("Predict Survival"):
    # Build DataFrame matching training columns
    input_df = pd.DataFrame([[pclass, sex, age, sibsp, parch, fare, embarked]],
                             columns=["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"])

    # Reorder to match saved column order
    input_df = input_df[columns]

    X_transformed = transformer.transform(input_df)
    prediction    = model.predict(X_transformed)[0]
    probability   = model.predict_proba(X_transformed)[0]

    surv_prob  = round(probability[1] * 100, 1)
    death_prob = round(probability[0] * 100, 1)

    st.markdown("---")
    if prediction == 1:
        st.markdown(f"""
        <div class="pred-survived">
          <div class="pred-label">✅ Survived</div>
          <div class="pred-prob">{surv_prob}%</div>
          <div class="pred-note">Survival probability · Not-survived: {death_prob}%</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="pred-died">
          <div class="pred-label">❌ Did Not Survive</div>
          <div class="pred-prob">{death_prob}%</div>
          <div class="pred-note">Non-survival probability · Survived: {surv_prob}%</div>
        </div>""", unsafe_allow_html=True)

    # Probability bar
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Survival probability breakdown**")
    prob_df = pd.DataFrame({"Outcome": ["Survived", "Did Not Survive"],
                             "Probability": [surv_prob, death_prob]})
    st.bar_chart(prob_df.set_index("Outcome"))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><hr style='border-color:#1e3550'><p style='text-align:center;color:#3a5a7a;font-size:0.8rem'>Logistic Regression · scikit-learn · Titanic Dataset</p>", unsafe_allow_html=True)
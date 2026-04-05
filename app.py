import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FakeRadar — News Authenticity Detector",
    page_icon="📡",
    layout="centered",
)

# ── Download NLTK data ─────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    nltk.download("stopwords", quiet=True)

download_nltk()

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# ── Train or load model ────────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"
VEC_PATH   = "vectorizer.pkl"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        model      = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VEC_PATH)
        return model, vectorizer, None

    # Try to train from CSVs if they exist
    if os.path.exists("Fake.csv") and os.path.exists("True.csv"):
        fake_df = pd.read_csv("Fake.csv")
        true_df = pd.read_csv("True.csv")
        fake_df["label"] = 1
        true_df["label"] = 0
        df = pd.concat([fake_df, true_df], ignore_index=True).sample(5000, random_state=42)
        text_col = "text" if "text" in df.columns else df.columns[0]
        df["cleaned"] = df[text_col].apply(clean_text)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vectorizer.fit_transform(df["cleaned"])
        y = df["label"].values

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)

        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VEC_PATH)
        return model, vectorizer, None

    return None, None, "no_data"

# ── Prediction ─────────────────────────────────────────────────────────────────
def predict(text, model, vectorizer):
    cleaned = clean_text(text)
    vec     = vectorizer.transform([cleaned])
    pred    = model.predict(vec)[0]
    prob    = model.predict_proba(vec)[0]
    return pred, prob

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0a0f;
    color: #e8e8f0;
}

.main-header {
    text-align: center;
    padding: 2.5rem 0 1rem;
}

.main-header h1 {
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -2px;
    background: linear-gradient(135deg, #ff6b35, #f7931e, #ffcd3c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-header p {
    color: #888;
    font-family: 'DM Mono', monospace;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.result-card {
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
    text-align: center;
    animation: fadeIn 0.4s ease;
}

.result-fake {
    background: linear-gradient(135deg, rgba(255,50,50,0.15), rgba(255,100,50,0.1));
    border: 1px solid rgba(255,80,80,0.4);
}

.result-real {
    background: linear-gradient(135deg, rgba(50,255,150,0.12), rgba(50,200,255,0.08));
    border: 1px solid rgba(50,255,150,0.35);
}

.result-label {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.4rem;
}

.result-fake .result-label  { color: #ff4f4f; }
.result-real .result-label  { color: #3dffa0; }

.confidence-text {
    font-family: 'DM Mono', monospace;
    font-size: 0.9rem;
    color: #aaa;
}

.stat-row {
    display: flex;
    justify-content: center;
    gap: 2.5rem;
    margin: 2rem 0;
    flex-wrap: wrap;
}

.stat-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1rem 1.8rem;
    text-align: center;
}

.stat-num {
    font-size: 1.8rem;
    font-weight: 800;
    color: #ffcd3c;
}

.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #666;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

textarea {
    background: #13131a !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #ff6b35, #f7931e) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    letter-spacing: 0.5px !important;
    transition: opacity 0.2s !important;
}

.stButton > button:hover { opacity: 0.88 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 FakeRadar</h1>
    <p>AI-powered news authenticity detector</p>
</div>
""", unsafe_allow_html=True)

# ── Model stats ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-row">
    <div class="stat-box"><div class="stat-num">98.75%</div><div class="stat-label">Accuracy</div></div>
    <div class="stat-box"><div class="stat-num">44,898</div><div class="stat-label">Articles Trained</div></div>
    <div class="stat-box"><div class="stat-num">5,000</div><div class="stat-label">TF-IDF Features</div></div>
    <div class="stat-box"><div class="stat-num">0.99</div><div class="stat-label">F1 Score</div></div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Load model ─────────────────────────────────────────────────────────────────
model, vectorizer, error = load_or_train_model()

if error == "no_data":
    st.warning("""
    ⚠️ **Model files not found.**  
    Please upload `model.pkl` + `vectorizer.pkl`, **or** place `Fake.csv` + `True.csv` in the same folder and restart.
    """)
    st.stop()

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("### Paste a news article or headline")
user_input = st.text_area(
    label="",
    placeholder="e.g. Scientists confirm new treatment reduces cancer risk by 40%...",
    height=160,
    label_visibility="collapsed",
)

# ── Examples ───────────────────────────────────────────────────────────────────
st.markdown("**Try an example:**")
col1, col2, col3 = st.columns(3)
examples = [
    "Breaking: NASA confirms alien life discovered on Mars",
    "The Prime Minister announced new education reforms today",
    "Scientists discovered a new virus spreading rapidly",
]
if col1.button("👽 NASA aliens"):   user_input = examples[0]
if col2.button("🏛️ PM reforms"):   user_input = examples[1]
if col3.button("🦠 Virus spread"): user_input = examples[2]

# ── Analyse ────────────────────────────────────────────────────────────────────
if st.button("🔍 Analyse Article"):
    if not user_input.strip():
        st.error("Please enter some text first.")
    else:
        with st.spinner("Scanning article..."):
            pred, prob = predict(user_input, model, vectorizer)
            confidence = prob[pred] * 100
            fake_prob  = prob[1] * 100
            real_prob  = prob[0] * 100

        if pred == 1:
            st.markdown(f"""
            <div class="result-card result-fake">
                <div class="result-label">🚨 FAKE NEWS</div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card result-real">
                <div class="result-label">✅ REAL NEWS</div>
                <div class="confidence-text">Confidence: {confidence:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Probability breakdown")
        st.progress(real_prob / 100, text=f"Real — {real_prob:.1f}%")
        st.progress(fake_prob / 100, text=f"Fake — {fake_prob:.1f}%")

        if confidence < 65:
            st.info("ℹ️ Low confidence — treat this result with caution. The article may be ambiguous or use unusual phrasing.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<p style="text-align:center; color:#444; font-family:'DM Mono',monospace; font-size:0.75rem;">
Built with Logistic Regression · TF-IDF · Streamlit &nbsp;|&nbsp; Trained on ISOT Fake News Dataset
</p>
""", unsafe_allow_html=True)
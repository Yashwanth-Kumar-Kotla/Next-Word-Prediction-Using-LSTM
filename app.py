import streamlit as st
import torch
import pickle
from nltk.tokenize import word_tokenize
from model import LSTMNN

import nltk
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="🔍",
    layout="centered",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

  html, body, [class*="css"] {
    background-color: #0e0c0a;
    color: #e8d9b5;
  }

  .stApp {
    background: radial-gradient(ellipse at top, #1a1510 0%, #0e0c0a 70%);
    min-height: 100vh;
  }

  /* Header */
  .hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
  }
  .hero-icon {
    font-size: 3rem;
    line-height: 1;
    margin-bottom: 0.4rem;
  }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #f5c842;
    letter-spacing: 0.02em;
    margin: 0;
  }
  .hero p {
    font-family: 'Lora', serif;
    font-style: italic;
    color: #a89070;
    font-size: 0.95rem;
    margin-top: 0.35rem;
  }

  /* Divider */
  .divider {
    border: none;
    border-top: 1px solid #3a2e1e;
    margin: 1.2rem 0;
  }

  /* Input card */
  .card {
    background: #18140f;
    border: 1px solid #3a2e1e;
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.2rem;
  }
  .card-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.78rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #f5c842;
    margin-bottom: 0.4rem;
  }

  /* Override Streamlit input */
  .stTextInput > div > div > input {
    background-color: #0e0c0a !important;
    color: #e8d9b5 !important;
    border: 1px solid #5a4a2e !important;
    border-radius: 8px !important;
    font-family: 'Lora', serif !important;
    font-size: 1.05rem !important;
    padding: 0.65rem 0.9rem !important;
    caret-color: #f5c842;
  }
  .stTextInput > div > div > input:focus {
    border-color: #f5c842 !important;
    box-shadow: 0 0 0 2px rgba(245,200,66,0.15) !important;
  }
  .stTextInput > label { display: none !important; }

  /* Slider */
  .stSlider > label { display: none !important; }
  .stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #f5c842 !important;
    border-color: #f5c842 !important;
  }
  .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] {
    color: #f5c842 !important;
  }

  /* Result box */
  .result-box {
    background: #12100d;
    border: 1px solid #f5c842;
    border-radius: 12px;
    padding: 1.4rem 1.8rem;
    margin-top: 0.6rem;
    font-family: 'Lora', serif;
    font-size: 1.12rem;
    line-height: 1.8;
  }
  .result-original {
    color: #c8b89a;
  }
  .result-predicted {
    color: #f5c842;
    font-weight: 600;
  }
  .result-label {
    font-family: 'Playfair Display', serif;
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #7a6a50;
    margin-bottom: 0.6rem;
  }

  /* Badges */
  .badge-row {
    display: flex;
    gap: 0.5rem;
    flex-wrap: wrap;
    margin-top: 0.9rem;
  }
  .badge {
    background: #1e190f;
    border: 1px solid #3a2e1e;
    border-radius: 20px;
    padding: 0.22rem 0.75rem;
    font-size: 0.75rem;
    color: #a89070;
    font-family: 'Lora', serif;
  }
  .badge span {
    color: #f5c842;
    font-weight: 600;
  }

  /* Footer */
  .footer {
    text-align: center;
    color: #4a3e2a;
    font-size: 0.72rem;
    font-family: 'Lora', serif;
    margin-top: 2.5rem;
    padding-bottom: 1.5rem;
    letter-spacing: 0.05em;
  }

  /* Hide Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)
    m = LSTMNN(len(vocab))
    m.load_state_dict(torch.load("model.pth", map_location="cpu"))
    m.eval()
    return vocab, max_len, m

vocab, max_len, model = load_model()


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_next_words(text, n_words=5):
    predicted = []
    current = text
    for _ in range(n_words):
        tokens = word_tokenize(current.lower())
        numerical = [vocab.get(w, vocab["unk"]) for w in tokens]
        padded = torch.tensor(
            [0] * (max_len - 1 - len(numerical)) + numerical,
            dtype=torch.long
        ).unsqueeze(0)
        with torch.no_grad():
            output = model(padded)
        _, index = torch.max(output, dim=1)
        next_word = list(vocab.keys())[index.item()]
        predicted.append(next_word)
        current = current + " " + next_word
    return predicted


# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-icon">🔍</div>
  <h1>Next Word Predictor</h1>
  <p>An LSTM model trained on <em>The Adventures of Sherlock Holmes</em></p>
</div>
<hr class="divider">
""", unsafe_allow_html=True)


# ── Input card ──────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">Your text</div>', unsafe_allow_html=True)
text = st.text_input("input", placeholder='e.g. "It is a capital mistake to theorise"', label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">Words to predict</div>', unsafe_allow_html=True)
n_words = st.slider("n_words", min_value=1, max_value=20, value=5, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)


# ── Result ──────────────────────────────────────────────────────────────────────
if text and text.strip():
    with st.spinner("Deducing..."):
        predicted_words = predict_next_words(text.strip(), n_words)

    predicted_str = " ".join(predicted_words)

    st.markdown(f"""
    <div class="result-box">
      <div class="result-label">Prediction</div>
      <span class="result-original">{text.strip()} </span><span class="result-predicted">{predicted_str}</span>
    </div>
    <div class="badge-row">
      <div class="badge">Words predicted: <span>{n_words}</span></div>
      <div class="badge">Vocabulary size: <span>{len(vocab):,}</span></div>
      <div class="badge">Model: <span>LSTM</span></div>
    </div>
    """, unsafe_allow_html=True)

elif text == "":
    st.markdown("""
    <div style="text-align:center; color:#4a3e2a; font-family:'Lora',serif; font-style:italic; padding: 1.5rem 0; font-size:0.9rem;">
      Enter a phrase above and the model will continue the story…
    </div>
    """, unsafe_allow_html=True)


# ── Footer ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
 Next Word Prediction Using LSTM · PyTorch · Streamlit
</div>
""", unsafe_allow_html=True)

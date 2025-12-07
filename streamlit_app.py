# streamlit_app.py
# Glassmorphism theme — Paste this file into your repo root alongside model.joblib
import streamlit as st
import joblib
import re
import time
import numpy as np
from pathlib import Path

# ---------------------------
# Page config + CSS
# ---------------------------
st.set_page_config(
    page_title="SMS / WhatsApp Spam Detector — Glass UI",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = r"""
/* Google font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root{
  --bg1: #0b0c0f;
  --glass: rgba(255,255,255,0.06);
  --glass-2: rgba(255,255,255,0.04);
  --accent: rgba(255,133,97,0.95);
  --muted: #bfc7c9;
  --card-radius: 16px;
}

/* Global */
html, body, [class*="css"] {
  background: linear-gradient(180deg,#070709 0%, #0b0c0f 60%);
  font-family: "Inter", sans-serif;
  color: #e9e9e9;
}

/* main container spacing */
.block {
  padding: 18px;
}

/* header */
.header {
  display:flex;
  align-items:center;
  gap:18px;
}
.logo {
  width:84px; height:84px;
  border-radius:20px;
  display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg, rgba(255,160,130,0.18), rgba(255,110,85,0.12));
  box-shadow: 0 10px 40px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.02);
}
.logo-emoji { font-size:36px; transform:translateY(-2px) }

/* frosted glass card */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  backdrop-filter: blur(8px) saturate(1.1);
  -webkit-backdrop-filter: blur(8px) saturate(1.1);
  border-radius: var(--card-radius);
  border: 1px solid rgba(255,255,255,0.04);
  padding: 20px;
  box-shadow: 0 8px 30px rgba(2,6,12,0.7);
}

/* textarea styling */
textarea[aria-label="Message"] {
  background: rgba(255,255,255,0.02) !important;
  color: #efecec !important;
  border-radius: 12px !important;
  padding: 18px !important;
  min-height: 160px !important;
  border: 1px solid rgba(255,255,255,0.035) !important;
  font-size: 15px !important;
}

/* buttons */
.stButton>button {
  border-radius: 12px;
  padding: 10px 18px;
  font-weight: 700;
  letter-spacing: 0.2px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.6);
  border: none;
}

/* primary style */
.btn-primary {
  background: linear-gradient(90deg, rgba(255,133,97,0.95), rgba(255,93,120,0.95));
  color: white !important;
  border: 1px solid rgba(255,255,255,0.04);
}

/* secondary */
.btn-ghost {
  background: rgba(255,255,255,0.02);
  color: var(--muted) !important;
  border: 1px solid rgba(255,255,255,0.03);
}

/* result banners */
.result-success {
  background: linear-gradient(90deg, rgba(18,87,64,0.9), rgba(12,60,44,0.92));
  padding: 14px; border-radius: 10px; color: #ddf6e8; font-weight:700;
  box-shadow: 0 8px 30px rgba(6,40,30,0.45);
}
.result-danger {
  background: linear-gradient(90deg, rgba(120,20,20,0.95), rgba(80,10,10,0.93));
  padding: 14px; border-radius: 10px; color: #ffe3e3; font-weight:700;
  box-shadow: 0 8px 30px rgba(60,10,10,0.45);
}

/* confidence bar container */
.conf-wrap {
  margin-top:10px;
  background: rgba(255,255,255,0.015);
  border-radius: 8px;
  padding:8px;
  border: 1px solid rgba(255,255,255,0.02);
}

/* examples */
.example {
  padding:10px; border-radius:8px; margin-bottom:8px;
  background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.005));
  border: 1px solid rgba(255,255,255,0.02);
  color:#dfe8ea;
}

/* titles */
.h1 {
  font-size:36px; font-weight:800; margin:0;
}
.h2 { font-size:20px; font-weight:700; color:#dfe8ea; }

/* small muted */
.small { color: #aeb7b9; font-size:13px; }

/* subtle animated glow around main card */
@keyframes floatglow {
  0% { box-shadow: 0 6px 30px rgba(0,0,0,0.6); transform: translateY(0px); }
  50% { box-shadow: 0 18px 60px rgba(0,0,0,0.65); transform: translateY(-6px); }
  100% { box-shadow: 0 6px 30px rgba(0,0,0,0.6); transform: translateY(0px); }
}
.card-animated { animation: floatglow 6s ease-in-out infinite; }

/* responsive tweaks */
@media (max-width:900px) {
  .h1 { font-size:26px; }
  .logo { width:64px; height:64px; }
}
"""

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


# ---------------------------
# Utilities
# ---------------------------
def clean_text(txt: str) -> str:
    if not isinstance(txt, str):
        txt = str(txt)
    txt = txt.lower()
    txt = re.sub(r"http\S+", " ", txt)
    txt = re.sub(r"[^a-z0-9\s]", " ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def load_model(p="model.joblib"):
    path = Path(p)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path.resolve()}")
    return joblib.load(path)

# ---------------------------
# Layout: Header
# ---------------------------
left_col, right_col = st.columns([0.12, 0.88])
with left_col:
    st.markdown('<div class="logo"><div class="logo-emoji">✉️</div></div>', unsafe_allow_html=True)
with right_col:
    st.markdown('<div><div class="h1">SMS / WhatsApp Spam Detector</div><div class="small">Real-time classification — presentation-ready UI</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------
# Main two-column layout
# ---------------------------
main, sidebar = st.columns([2.2, 0.9])

with main:
    st.markdown('<div class="card card-animated">', unsafe_allow_html=True)
    st.markdown('<div class="h2">Enter a message and press Predict</div>', unsafe_allow_html=True)
    st.markdown('<div class="small">Paste a message (with or without links), then click Predict.</div><br>', unsafe_allow_html=True)

    # Message input
    msg = st.text_area("Message", value="Your appointment is confirmed for Monday at 4pm.", key="msg_input")

    # Buttons row
    b1, b2, b3 = st.columns([1,1,1])
    with b1:
        if st.button("Predict", key="btn_predict"):
            st.session_state._predict = True
    with b2:
        if st.button("Quick: Spam sample", key="btn_spam"):
            st.session_state.msg_input = "Congratulations! You have won a FREE iPhone. Click here to claim: http://bit.ly/win-phone"
            st.session_state._predict = True
    with b3:
        if st.button("Quick: Ham sample", key="btn_ham"):
            st.session_state.msg_input = "See you at the library at 6pm. Bring the notes."
            st.session_state._predict = True

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # Advanced expander
    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: `TF-IDF` + `MultinomialNB` (scikit-learn pipeline).")
        st.markdown("- The pipeline expects raw text; it includes TF-IDF inside the pipeline.")
        st.markdown("- Use the Quick buttons for fast demo in your viva.")

    # Prediction area
    if st.session_state.get("_predict", False):
        with st.spinner("Analyzing..."):
            time.sleep(0.5)
            try:
                model = load_model("model.joblib")
                current_msg = st.session_state.get("msg_input", msg)
                cleaned = clean_text(current_msg)
                pred = model.predict([cleaned])[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        prob = float(np.max(model.predict_proba([cleaned])[0]))
                    except Exception:
                        prob = None

                if str(pred).lower() in ("spam", "1", "true"):
                    st.markdown('<div class="result-danger">🚨 This message is <strong>SPAM</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success">✅ This message is <strong>NOT spam</strong></div>', unsafe_allow_html=True)

                if prob is not None:
                    pct = int(round(prob*100))
                    st.markdown(f'<div class="conf-wrap"><div class="small">Model confidence</div></div>', unsafe_allow_html=True)
                    st.progress(pct)
                    st.write(f"**Confidence:** {pct}%")
                else:
                    st.write("**Confidence:** N/A")

            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as e:
                st.error("Prediction error: " + str(e))
        st.session_state._predict = False

    st.markdown("</div>", unsafe_allow_html=True)

    # Examples card
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div class='h2'>Example messages</div>", unsafe_allow_html=True)
    examples = [
        ("Congratulations! You have won a FREE iPhone. Click here to claim: http://bit.ly/win-phone", "Spam"),
        ("Your parcel delivery failed. Track here: http://track.example", "Spam"),
        ("Are you coming to class today?", "Ham"),
        ("Don't forget the meeting at 3pm.", "Ham"),
    ]
    for txt, lbl in examples:
        badge = "🔴 SPAM" if lbl=="Spam" else "🟢 HAM"
        st.markdown(f'<div class="example"><strong>{badge}</strong> &nbsp; {txt}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="h2">Project Info</div>', unsafe_allow_html=True)
    st.markdown('- Model: **TF-IDF + MultinomialNB**')
    st.markdown('- Demo by: **Your Name**')
    st.markdown('- Repo: `sms-spam-detection`')
    st.markdown('<br>', unsafe_allow_html=True)
    st.metric("Test Accuracy (example)", "96%")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick tests", unsafe_allow_html=True)
    if st.button("Try spam sample (sidebar)"):
        st.session_state.msg_input = "Free entry! Win cash now. Click http://claim.example"
        st.session_state._predict = True
    if st.button("Try ham sample (sidebar)"):
        st.session_state.msg_input = "I'll be late by 10 minutes, stuck in traffic."
        st.session_state._predict = True
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Notes", unsafe_allow_html=True)
    st.markdown("- Use public URL in slides.")
    st.markdown("- Keep a recording as fallback.")

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><div style='text-align:center;color:#9da7a9;font-size:13px'>Pro tip: Use Quick buttons to demonstrate cases quickly during your viva.</div>", unsafe_allow_html=True)

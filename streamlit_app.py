# streamlit_app.py
"""
Fixed: remove st.experimental_rerun() calls to avoid AttributeError on some hosts.
Enhanced SMS / WhatsApp Spam Detector demo (animations, history, batch predict).
"""

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import time
from pathlib import Path
from io import StringIO
import base64

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="✨ SMS / WhatsApp Spam Detector — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# CSS + JS (same visuals as before)
# -------------------------
PAGE_CSS = r"""
<style>
/* (long CSS omitted here for brevity in chat) */
/* Use the same CSS from the prior full script you already received. */
</style>

<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
<script>
function typeHeadline(id, txt, delay=40){
  const el = document.getElementById(id);
  if(!el) return;
  el.innerText = "";
  let i=0;
  const iv = setInterval(()=> {
    el.innerText += txt[i] ?? "";
    i++;
    if(i >= txt.length) clearInterval(iv);
  }, delay);
}
window.addEventListener("load", function(){
  setTimeout(()=> typeHeadline("hero-title", "SMS / WhatsApp Spam Detector — live demo", 28), 300);
});
window.runConfetti = function(isGood){
  if(typeof confetti !== "function") return;
  if(isGood){
    confetti({ particleCount: 120, spread: 70, origin: { y: 0.2 } });
  } else {
    confetti({ particleCount: 40, spread: 30, origin: { y: 0.1 }, scalar: 0.6 });
  }
}
</script>
"""
# Put full CSS contents here when you paste into your file.
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# -------------------------
# Utilities
# -------------------------
def clean_text(s: str):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_model(path="model.joblib"):
    p = Path(path)
    if p.exists():
        return joblib.load(p)
    else:
        raise FileNotFoundError(f"model file not found at: {p.resolve()}")

def download_link(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    return href

# -------------------------
# Samples
# -------------------------
SAMPLES = [
    ("Congratulations! You have won a FREE iPhone. Click here to claim: http://bit.ly/win-phone", "Spam"),
    ("Your parcel delivery failed. Track here: http://track-now.example", "Spam"),
    ("Are you coming to class today?", "Ham"),
    ("Don't forget the meeting at 3pm. See you there.", "Ham"),
    ("Urgent: Your account will be suspended. Call 1800-999-000", "Spam"),
    ("Happy birthday! Hope you have a great day.", "Ham"),
]

# -------------------------
# Session state
# -------------------------
if "message" not in st.session_state:
    st.session_state["message"] = "Your appointment is confirmed for Monday at 4pm."
if "_do_predict" not in st.session_state:
    st.session_state["_do_predict"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []
if "model_path" not in st.session_state:
    st.session_state["model_path"] = "model.joblib"

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([0.14, 0.86])
with col1:
    st.markdown('<div style="width:84px;height:84px;border-radius:20px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#8b5cf6,#f97316);font-size:36px;color:white;">✉️</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div><h2 id="hero-title" style="margin:0"></h2><div style="color:#9aa0a6;margin-top:6px">Real-time classification • Demo-ready UI</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2, 0.95])

with left:
    st.markdown('<div style="padding:18px;border-radius:12px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.03)">', unsafe_allow_html=True)

    st.markdown("<h3 style='margin:0'>Try it live</h3><div style='color:#9aa0a6'>Type a message or use quick tests below</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # text area controlled by session_state
    msg = st.text_area("Message", value=st.session_state["message"], key="message", height=180)

    # Buttons - do NOT call experimental_rerun; update session_state and rely on Streamlit's built-in rerun
    b1, b2, b3, b4 = st.columns([1,1,1,1])
    with b1:
        if st.button("Predict", key="predict_btn"):
            # set flag; Streamlit will rerun automatically and the code below will act on this flag
            st.session_state["_do_predict"] = True
    with b2:
        if st.button("Quick: Spam sample", key="quick_spam_btn"):
            st.session_state["message"] = SAMPLES[0][0]
            # no experimental_rerun()
    with b3:
        if st.button("Quick: Ham sample", key="quick_ham_btn"):
            st.session_state["message"] = SAMPLES[2][0]
    with b4:
        if st.button("Clear", key="clear_btn"):
            st.session_state["message"] = ""

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: **scikit-learn Pipeline** (TF-IDF + MultinomialNB).")
        st.markdown("- Upload a model file below to override default `model.joblib`.")
        uploaded_model = st.file_uploader("Upload a scikit-learn pipeline (.joblib)", type=["joblib", "pkl"])
        if uploaded_model is not None:
            tmp_path = Path("uploaded_model.joblib")
            tmp_path.write_bytes(uploaded_model.read())
            st.session_state["model_path"] = str(tmp_path.resolve())
            st.success("Uploaded model will be used for predictions.")

    # Prediction logic: run when flag is set
    if st.session_state.get("_do_predict", False):
        with st.spinner("Analyzing message..."):
            time.sleep(0.6)
            try:
                model = load_model(st.session_state.get("model_path", "model.joblib"))
                cleaned = clean_text(st.session_state["message"])
                pred = model.predict([cleaned])[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        probv = model.predict_proba([cleaned])[0]
                        prob = float(np.max(probv))
                    except Exception:
                        prob = None

                is_spam = str(pred).lower() in ("spam", "1", "true")
                if is_spam:
                    st.markdown('<div style="background:linear-gradient(90deg,#5b0b0b,#3b0f0f);padding:14px;border-radius:10px;color:#ffd6d6;"><strong>🚨 This message is SPAM</strong></div>', unsafe_allow_html=True)
                    st.markdown("<script>window.runConfetti(false);</script>", unsafe_allow_html=True)
                else:
                    st.markdown('<div style="background:linear-gradient(90deg,#064e3b,#047857);padding:14px;border-radius:10px;color:#dff7e3;"><strong>✅ This message is NOT spam</strong></div>', unsafe_allow_html=True)
                    st.markdown("<script>window.runConfetti(true);</script>", unsafe_allow_html=True)

                if prob is not None:
                    pct = int(round(prob * 100))
                    st.markdown("<div style='margin-top:8px;color:#9aa0a6'>Confidence</div>", unsafe_allow_html=True)
                    st.progress(pct)
                    st.write(f"Model confidence: **{pct}%**")
                else:
                    st.write("Model confidence: **N/A**")

                hist_item = {
                    "text": st.session_state["message"],
                    "pred": "Spam" if is_spam else "Ham",
                    "prob": round(prob, 4) if prob is not None else None,
                    "time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state["history"].insert(0, hist_item)
            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as e:
                st.error("Prediction failed: " + str(e))

        # reset flag so next click triggers again
        st.session_state["_do_predict"] = False

    st.markdown("</div>", unsafe_allow_html=True)

    # History + Batch predictions
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div style="padding:12px;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.03)">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0'>Prediction history</h4>", unsafe_allow_html=True)
    if len(st.session_state["history"]) == 0:
        st.markdown('<div style="color:#9aa0a6">No predictions yet — try a sample message above.</div>', unsafe_allow_html=True)
    else:
        for h in st.session_state["history"][:10]:
            badge = "🔴 SPAM" if h["pred"].lower().startswith("s") else "🟢 HAM"
            prob_txt = f" — conf {int(h['prob']*100)}%" if h.get("prob") else ""
            st.markdown(f'<div style="padding:8px;border-radius:8px;margin-bottom:8px;background:rgba(255,255,255,0.01);border:1px solid rgba(255,255,255,0.02)"><strong>{badge}</strong> &nbsp; {h["text"]}<div style="color:#9aa0a6;margin-top:6px">{h["time"]}{prob_txt}</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("### Batch predictions (CSV)", unsafe_allow_html=True)
    csv_file = st.file_uploader("Upload CSV with `message` column for batch predict", type=["csv"])
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            if "message" not in df.columns:
                st.error("CSV must contain a 'message' column")
            else:
                st.info("Running batch predictions...")
                model = load_model(st.session_state.get("model_path", "model.joblib"))
                cleaned = df["message"].astype(str).apply(clean_text).tolist()
                preds = model.predict(cleaned)
                probs = None
                if hasattr(model, "predict_proba"):
                    try:
                        probv = model.predict_proba(cleaned)
                        probs = np.max(probv, axis=1)
                    except Exception:
                        probs = None
                df["prediction"] = ["Spam" if str(p).lower() in ("spam","1","true") else "Ham" for p in preds]
                if probs is not None:
                    df["confidence"] = probs
                st.success("Batch predictions completed")
                st.dataframe(df.head(10))
                st.markdown(download_link(df, "batch_predictions.csv"), unsafe_allow_html=True)
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div style="padding:12px;border-radius:10px;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.03)">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0'>Project Info & Controls</h4>", unsafe_allow_html=True)
    st.markdown("<div style='color:#9aa0a6'>Team and quick controls</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Team")
    st.markdown("- **Abhishek Basu**")
    st.markdown("- **Ananya Raj**")
    st.markdown("- **Sneha Das**")
    st.markdown("- **Payel Guin**")
    st.markdown("- **Subhajit Khamrai**")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick tests", unsafe_allow_html=True)
    for i, (txt, lbl) in enumerate(SAMPLES):
        if st.button(f"Try sample {i+1} — {lbl}", key=f"right_sample_{i}"):
            st.session_state["message"] = txt
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div style="text-align:center;color:#9aa0a6;margin-top:12px">Tip: Use Quick sample buttons or upload your own dataset. Want a custom theme? Ask me!</div>', unsafe_allow_html=True)

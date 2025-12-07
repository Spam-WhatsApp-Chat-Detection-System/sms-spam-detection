# streamlit_app.py
"""
Streamlit demo: SMS / WhatsApp Spam Detector
- Uses TF-IDF + MultinomialNB pipeline saved as model_tokenized.joblib or model.joblib
- Token-preserving cleaning (keeps __URL__ and __PHONE__ tokens)
- Single and batch predictions, history, animated UI bits
"""

import streamlit as st
import joblib
import re
import numpy as np
import pandas as pd
import time
from pathlib import Path
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
# Small CSS + JS (compact)
# -------------------------
PAGE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
:root{--muted:#9aa0a6;--accent1:#8b5cf6;--accent2:#06b6d4;--accent3:#f97316;}
html, body, [class*="css"]{font-family:Inter, sans-serif;color:#e6e6e6;background:linear-gradient(180deg,#03040a 0%,#071018 60%);}
.brand{width:84px;height:84px;border-radius:20px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,var(--accent1),var(--accent3));font-size:36px;color:white;}
.card{background:rgba(255,255,255,0.02);padding:18px;border-radius:12px;border:1px solid rgba(255,255,255,0.03);box-shadow:0 8px 26px rgba(2,6,23,0.5);}
textarea[aria-label="Message"]{background:rgba(0,0,0,0.45)!important;color:#f0f0f0!important;border-radius:10px!important;padding:16px!important;height:160px!important;border:1px solid rgba(255,255,255,0.04)!important;}
.stButton>button{border-radius:12px;padding:10px 18px;font-weight:700;box-shadow:0 10px 30px rgba(2,6,23,0.5);}
.result-success{background:linear-gradient(90deg,#064e3b,#047857);padding:14px;border-radius:10px;color:#dff7e3;}
.result-danger{background:linear-gradient(90deg,#5b0b0b,#3b0f0f);padding:14px;border-radius:10px;color:#ffd6d6;}
.small-muted{color:var(--muted);font-size:13px;}
.history-item{padding:8px;border-radius:8px;margin-bottom:8px;background:rgba(255,255,255,0.01);border:1px solid rgba(255,255,255,0.02);}
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
window.addEventListener("load", function(){ setTimeout(()=> typeHeadline("hero-title", "SMS / WhatsApp Spam Detector — live demo", 28), 300); });
window.runConfetti = function(isGood){
  if(typeof confetti !== "function") return;
  if(isGood){ confetti({ particleCount: 120, spread: 70, origin: { y: 0.2 } }); }
  else { confetti({ particleCount: 40, spread: 30, origin: { y: 0.1 }, scalar: 0.6 }); }
}
</script>
"""
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# -------------------------
# Utilities: cleaner and model loader
# -------------------------
def clean_text_keep_tokens(s: str):
    """Token-preserving cleaner: replace URLs and long numbers with tokens the model knows."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' __URL__ ', s)
    s = re.sub(r'\b\d{6,}\b', ' __PHONE__ ', s)
    s = re.sub(r'[^a-z0-9_\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_model(path_preferred="model_tokenized.joblib", fallback="model.joblib"):
    """Try preferred model path, then fallback. Raise FileNotFoundError if none present."""
    p1 = Path(path_preferred)
    p2 = Path(fallback)
    if p1.exists():
        return joblib.load(p1)
    if p2.exists():
        return joblib.load(p2)
    raise FileNotFoundError(f"model file not found (tried: {p1.resolve()}, {p2.resolve()})")

def download_link(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    return href

# -------------------------
# Samples & session state
# -------------------------
SAMPLES = [
    ("Congratulations! You have won a FREE iPhone. Click here to claim: http://bit.ly/win-phone", "Spam"),
    ("Your parcel delivery failed. Track here: http://track-now.example", "Spam"),
    ("Are you coming to class today?", "Ham"),
    ("Don't forget the meeting at 3pm. See you there.", "Ham"),
    ("Urgent: Your account will be suspended. Call 1800-999-000", "Spam"),
    ("Happy birthday! Hope you have a great day.", "Ham"),
]

if "message" not in st.session_state:
    st.session_state["message"] = "Your appointment is confirmed for Monday at 4pm."
if "_do_predict" not in st.session_state:
    st.session_state["_do_predict"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []
if "model_path" not in st.session_state:
    st.session_state["model_path"] = "model_tokenized.joblib"

# -------------------------
# Layout / Header
# -------------------------
col1, col2 = st.columns([0.14, 0.86])
with col1:
    st.markdown('<div class="brand">✉️</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div><h2 id="hero-title" style="margin:0"></h2><div class="small-muted">Real-time classification • Demo-ready UI</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2, 1])

# -------------------------
# Left: main app
# -------------------------
with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h3 style='margin:0'>Try it live</h3><div class='small-muted'>Type a message or use quick tests below</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # controlled text area
    msg = st.text_area("Message", value=st.session_state["message"], key="message", height=180)

    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("Predict", key="predict_btn"):
            st.session_state["_do_predict"] = True
    with c2:
        if st.button("Quick: Spam sample", key="quick_spam_btn"):
            st.session_state["message"] = SAMPLES[0][0]
    with c3:
        if st.button("Quick: Ham sample", key="quick_ham_btn"):
            st.session_state["message"] = SAMPLES[2][0]
    with c4:
        if st.button("Clear", key="clear_btn"):
            st.session_state["message"] = ""

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: **TF-IDF + MultinomialNB**")
        st.markdown("- Upload a model file (.joblib) to override the built-in model.")
        uploaded_model = st.file_uploader("Upload a scikit-learn pipeline (.joblib)", type=["joblib","pkl"])
        if uploaded_model is not None:
            tmp = Path("uploaded_model.joblib")
            tmp.write_bytes(uploaded_model.read())
            st.session_state["model_path"] = str(tmp.resolve())
            st.success("Uploaded model will be used for predictions (saved as uploaded_model.joblib).")

    # Prediction block
    if st.session_state.get("_do_predict", False):
        with st.spinner("Analyzing message..."):
            time.sleep(0.45)
            try:
                # load model (preferred path or fallback)
                model = load_model(st.session_state.get("model_path", "model_tokenized.joblib"))
                # use token-preserving cleaning
                cleaned = clean_text_keep_tokens(st.session_state.get("message", ""))
                # optional debug: st.write("DEBUG cleaned:", cleaned)
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
                    st.markdown('<div class="result-danger"><strong>🚨 This message is SPAM</strong></div>', unsafe_allow_html=True)
                    st.markdown("<script>window.runConfetti(false);</script>", unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success"><strong>✅ This message is NOT spam</strong></div>', unsafe_allow_html=True)
                    st.markdown("<script>window.runConfetti(true);</script>", unsafe_allow_html=True)

                if prob is not None:
                    pct = int(round(prob * 100))
                    st.markdown("<div class='small-muted' style='margin-top:8px'>Confidence</div>", unsafe_allow_html=True)
                    st.progress(pct)
                    st.write(f"Model confidence: **{pct}%**")
                else:
                    st.write("Model confidence: **N/A**")

                # store history
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
        # reset flag
        st.session_state["_do_predict"] = False

    st.markdown("</div>", unsafe_allow_html=True)

    # History + Batch
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0'>Prediction history</h4>", unsafe_allow_html=True)
    if not st.session_state["history"]:
        st.markdown('<div class="small-muted">No predictions yet — try a sample message above.</div>', unsafe_allow_html=True)
    else:
        for h in st.session_state["history"][:10]:
            badge = "🔴 SPAM" if h["pred"].lower().startswith("s") else "🟢 HAM"
            prob_txt = f" — conf {int(h['prob']*100)}%" if h.get("prob") else ""
            st.markdown(f'<div class="history-item"><strong>{badge}</strong> &nbsp; {h["text"]}<div class="small-muted" style="margin-top:6px">{h["time"]}{prob_txt}</div></div>', unsafe_allow_html=True)

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
                model = load_model(st.session_state.get("model_path", "model_tokenized.joblib"))
                cleaned_series = df["message"].astype(str).apply(clean_text_keep_tokens)
                preds = model.predict(cleaned_series.tolist())
                probs = None
                if hasattr(model, "predict_proba"):
                    try:
                        probv = model.predict_proba(cleaned_series.tolist())
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

# -------------------------
# Right: info & quick samples
# -------------------------
with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0'>Project Info & Controls</h4>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Team and quick controls</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Team")
    st.markdown("- **Abhishek Basu**")
    st.markdown("- **Ananya Raj**")
    st.markdown("- **Sneha Das**")
    st.markdown("- **Payel Guin**")
    st.markdown("- **Subhajit Khamkai**")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Quick tests", unsafe_allow_html=True)
    for i, (txt, lbl) in enumerate(SAMPLES):
        if st.button(f"Try sample {i+1} — {lbl}", key=f"right_sample_{i}"):
            st.session_state["message"] = txt

    st.markdown("<br>", unsafe_allow_html=True)
    # show model diagnostics (safe)
    try:
        mdl = load_model(st.session_state.get("model_path", "model_tokenized.joblib"))
        if hasattr(mdl, "classes_"):
            st.markdown(f"**Model classes:** {list(mdl.classes_)}")
        if hasattr(mdl, "named_steps") and "tfidf" in mdl.named_steps:
            tf = mdl.named_steps["tfidf"]
            try:
                st.markdown(f"**TF-IDF vocab size:** {len(tf.vocabulary_)}")
            except Exception:
                pass
    except Exception:
        st.markdown("<div class='small-muted'>Model diagnostics not available (model may be missing).</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown('<div style="text-align:center;color:#9aa0a6;margin-top:12px">Tip: Use Quick sample buttons or upload your own dataset. Want a custom theme? Ask me!</div>', unsafe_allow_html=True)

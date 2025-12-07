# streamlit_app.py
"""
Enhanced SMS / WhatsApp Spam Detector demo
- Long, attractive UI with animations
- Reliable quick-sample buttons (use session_state + experimental_rerun)
- Prediction history + batch CSV prediction + download
- Confetti (for ham) or shake animation (for spam)
- Keep model logic: expects 'model.joblib' in same folder (or upload alternative)
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
import json

# -------------------------
# Page config
# -------------------------
st.set_page_config(
    page_title="✨ SMS / WhatsApp Spam Detector — Demo",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "mailto:your-email@example.com",
        "Report a bug": "mailto:your-email@example.com",
        "About": "This demo app uses TF-IDF + MultinomialNB. Designed by Abhishek & team."
    }
)

# -------------------------
# Fancy CSS + small JS
# -------------------------
PAGE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root{
  --glass-bg: rgba(255,255,255,0.03);
  --muted: #9aa0a6;
  --accent1: #8b5cf6;
  --accent2: #06b6d4;
  --accent3: #f97316;
}

/* global */
html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
    color: #e6e6e6;
    background: radial-gradient(1200px 400px at 10% 10%, rgba(139,92,246,0.06), transparent 8%),
                radial-gradient(1000px 300px at 90% 80%, rgba(6,182,212,0.05), transparent 8%),
                linear-gradient(180deg, #03040a 0%, #071018 60%);
}

/* animated gradient overlay */
.gradient-hero {
  position:relative;
  overflow:hidden;
  border-radius: 16px;
  padding: 22px;
  background: linear-gradient(90deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: 0 10px 30px rgba(2,6,23,0.6);
}
.gradient-hero:before {
  content: "";
  position:absolute;
  top:-20%;
  left:-40%;
  width:180%;
  height:180%;
  background: linear-gradient(60deg, rgba(139,92,246,0.06), rgba(6,182,212,0.06), rgba(249,115,22,0.04));
  transform: rotate(15deg);
  animation: float 12s linear infinite;
  z-index:0;
}
@keyframes float {
  0% { transform: translateX(-3%) translateY(0) rotate(15deg); }
  50% { transform: translateX(3%) translateY(4%) rotate(15deg); }
  100% { transform: translateX(-3%) translateY(0) rotate(15deg); }
}

/* header */
.header-row { display:flex; gap:18px; align-items:center; z-index:1; position:relative; }
.brand {
  width:84px; height:84px; border-radius:20px; display:flex; align-items:center; justify-content:center;
  background: linear-gradient(135deg,var(--accent1), var(--accent3));
  font-size:36px; box-shadow: 0 16px 40px rgba(139,92,246,0.12); color:white;
}
.h-title { font-size:26px; font-weight:800; margin:0; color:#fff; }
.h-sub { color:var(--muted); margin-top:4px; font-size:13px; }

/* cards */
.card {
  background: var(--glass-bg);
  padding:18px;
  border-radius:12px;
  border: 1px solid rgba(255,255,255,0.03);
  box-shadow: 0 8px 26px rgba(2,6,23,0.5);
  position:relative;
}

/* input area */
textarea[aria-label="Message"] {
    background: rgba(0,0,0,0.45) !important;
    color: #f0f0f0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    height: 160px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
}

/* buttons */
.stButton>button {
    border-radius: 12px;
    padding: 10px 18px;
    font-weight:700;
    letter-spacing:0.2px;
    box-shadow: 0 10px 30px rgba(2,6,23,0.5);
    transition: transform .12s ease;
}
.stButton>button:active { transform: translateY(1px); }

/* colorful primary */
.btn-primary {
  background: linear-gradient(90deg,var(--accent1), var(--accent2)) !important;
  color: #fff !important;
  border: none !important;
}

/* sample row */
.sample-pill { display:inline-block; padding:8px 12px; border-radius:999px; background: rgba(255,255,255,0.02); margin-right:8px; border:1px solid rgba(255,255,255,0.02); }

/* result */
.result-success {
    background: linear-gradient(90deg,#064e3b,#047857);
    padding:14px;border-radius:10px;color:#dff7e3;
    display:flex; align-items:center; gap:12px;
}
.result-danger {
    background: linear-gradient(90deg,#5b0b0b,#3b0f0f);
    padding:14px;border-radius:10px;color:#ffd6d6;
    display:flex; align-items:center; gap:12px;
}

/* tiny helper */
.small-muted { color: var(--muted); font-size:13px; }

/* history list */
.history-item { padding:10px 12px; border-radius:10px; margin-bottom:8px; background: rgba(255,255,255,0.01); border:1px solid rgba(255,255,255,0.02); }

/* shake animation for spam */
@keyframes shake {
  0% { transform: translateX(0); }
  20% { transform: translateX(-6px); }
  40% { transform: translateX(6px); }
  60% { transform: translateX(-4px); }
  80% { transform: translateX(4px); }
  100% { transform: translateX(0); }
}
.shake { animation: shake 0.6s linear; }

/* fade-in */
.fade-in { animation: fadeIn .6s ease; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(6px);} to { opacity:1; transform: translateY(0);} }

/* footer */
.footer-note { color:var(--muted); text-align:center; font-size:13px; margin-top:12px; }

</style>

<!-- small JS helpers: typing headline & confetti library -->
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

// run when DOM ready
window.addEventListener("load", function(){
  setTimeout(()=> typeHeadline("hero-title", "SMS / WhatsApp Spam Detector — live demo", 28), 300);
});

// trigger confetti from python by calling window.runConfetti(true/false)
window.runConfetti = function(isGood){
  if(typeof confetti !== "function") return;
  if(isGood){
    confetti({ particleCount: 120, spread: 70, origin: { y: 0.2 } });
  } else {
    // negative confetti (small)
    confetti({ particleCount: 40, spread: 30, origin: { y: 0.1 }, scalar: 0.6 });
  }
}
</script>
"""

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
# Pre-made samples
# -------------------------
SAMPLES = [
    ("Congratulations! You have won a FREE iPhone. Click here to claim: http://bit.ly/win-phone", "Spam"),
    ("Your parcel delivery failed. Track here: http://track-now.example", "Spam"),
    ("Are you coming to class today?", "Ham"),
    ("Don't forget the meeting at 3pm. See you there.", "Ham"),
    ("Urgent: Your account will be suspended. Call 1800-999-000", "Spam"),
    ("Happy birthday! Hope you have a great day.", "Ham"),
    ("Limited time offer: Get 70% OFF, visit http://cheap-deals.example", "Spam"),
    ("Congrats on your new job! Let's celebrate this weekend 🎉", "Ham"),
]

# -------------------------
# Initialize session state
# -------------------------
if "message" not in st.session_state:
    st.session_state["message"] = "Your appointment is confirmed for Monday at 4pm."
if "_do_predict" not in st.session_state:
    st.session_state["_do_predict"] = False
if "history" not in st.session_state:
    st.session_state["history"] = []  # list of dicts: {"text":..,"pred":..,"prob":..,"time":..}
if "model_path" not in st.session_state:
    st.session_state["model_path"] = "model.joblib"

# -------------------------
# Header / Hero
# -------------------------
col1, col2 = st.columns([0.14, 0.86])
with col1:
    st.markdown('<div class="brand">✉️</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header-row"><div style="z-index:1"><h2 id="hero-title" class="h-title"></h2><div class="h-sub">Real-time classification • Demo-ready UI • Batch predict & download</div></div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -------------------------
# Main layout: left (app) and right (info)
# -------------------------
left, right = st.columns([2, 0.95])

with left:
    st.markdown('<div class="gradient-hero card">', unsafe_allow_html=True)

    st.markdown('<div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">'
                '<div><h3 style="margin:0">Try it live</h3><div class="small-muted">Type a message or use quick tests below</div></div>'
                '<div style="display:flex; gap:8px; align-items:center;">'
                '<div class="sample-pill small-muted">Model: TF-IDF + MultinomialNB</div>'
                '<div class="sample-pill small-muted">Demo mode</div>'
                '</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # message text area controlled by session_state
    msg = st.text_area("Message", value=st.session_state["message"], key="message", height=180)

    # action buttons row
    b1, b2, b3, b4 = st.columns([1,1,1,1])
    with b1:
        if st.button("Predict", key="predict_btn", help="Run model prediction on current message"):
            st.session_state._do_predict = True
            # force rerun to show result immediately
            st.experimental_rerun()
    with b2:
        if st.button("Quick: Spam sample", key="quick_spam_btn"):
            st.session_state["message"] = SAMPLES[0][0]
            st.experimental_rerun()
    with b3:
        if st.button("Quick: Ham sample", key="quick_ham_btn"):
            st.session_state["message"] = SAMPLES[2][0]
            st.experimental_rerun()
    with b4:
        if st.button("Clear", key="clear_btn"):
            st.session_state["message"] = ""
            st.experimental_rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # advanced / diagnostics expander
    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: **scikit-learn Pipeline** (TF-IDF + MultinomialNB).")
        st.markdown("- Tip: paste messages containing links or phone numbers to test edge cases.")
        st.markdown("- Upload a model file below to override default `model.joblib`.")
        uploaded_model = st.file_uploader("Upload a scikit-learn pipeline (.joblib)", type=["joblib", "pkl"])
        if uploaded_model is not None:
            # save uploaded model to temporary path and set state
            model_bytes = uploaded_model.read()
            tmp_path = Path("uploaded_model.joblib")
            tmp_path.write_bytes(model_bytes)
            st.session_state["model_path"] = str(tmp_path.resolve())
            st.success("Uploaded model will be used for predictions.")

    # Prediction area
    if st.session_state.get("_do_predict", False):
        with st.spinner("Analyzing message..."):
            time.sleep(0.6)
            try:
                # attempt to load model from chosen path
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

                # display animated result
                is_spam = str(pred).lower() in ("spam", "1", "true")
                if is_spam:
                    # spam -> show danger banner & add shake class
                    st.markdown('<div class="shake result-danger"><strong>🚨 This message is SPAM</strong><div style="flex:1"></div></div>', unsafe_allow_html=True)
                    # call small confetti for spam (neg)
                    st.markdown(
                        "<script>setTimeout(()=>window.runConfetti(false), 120);</script>", unsafe_allow_html=True
                    )
                else:
                    st.markdown('<div class="result-success"><strong>✅ This message is NOT spam</strong><div style="flex:1"></div></div>', unsafe_allow_html=True)
                    st.markdown("<script>setTimeout(()=>window.runConfetti(true), 120);</script>", unsafe_allow_html=True)

                # confidence
                if prob is not None:
                    pct = int(round(prob * 100))
                    st.markdown(f"<div style='margin-top:10px'><div class='small-muted'>Confidence</div></div>", unsafe_allow_html=True)
                    st.progress(pct)
                    st.write(f"Model confidence: **{pct}%**")
                else:
                    st.write("Model confidence: **N/A**")

                # save to history
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
        st.session_state._do_predict = False

    st.markdown("</div>", unsafe_allow_html=True)

    # History + Batch predict card
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
                "<div><h4 style='margin:0'>Prediction history</h4><div class='small-muted'>Recent messages you predicted</div></div>"
                "<div class='small-muted'>You can also upload a CSV with a `message` column for batch predictions</div>"
                "</div>", unsafe_allow_html=True)

    # show history
    if len(st.session_state["history"]) == 0:
        st.markdown('<div class="small-muted">No predictions yet — try a sample message above.</div>', unsafe_allow_html=True)
    else:
        for h in st.session_state["history"][:10]:
            badge = "🔴 SPAM" if h["pred"].lower().startswith("s") else "🟢 HAM"
            prob_txt = f" — conf {int(h['prob']*100)}%" if h.get("prob") else ""
            st.markdown(f'<div class="history-item fade-in"><strong>{badge}</strong> &nbsp; {h["text"]}<div class="small-muted" style="margin-top:6px;">{h["time"]}{prob_txt}</div></div>', unsafe_allow_html=True)

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
                # load model
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
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("<h4 style='margin:0'>Project Info & Controls</h4>", unsafe_allow_html=True)
    st.markdown("<div class='small-muted'>Quick links and demo settings</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Team")
    st.markdown("- **Abhishek Basu** (Lead)")
    st.markdown("- Ananya Raj")
    st.markdown("- Sneha Das")
    st.markdown("- Payal Guin")
    st.markdown("- Subhojit Khamrai")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Performance (demo)")
    st.markdown("- Test Accuracy on testing data: **95.87%**")
    st.markdown("- Precision / Recall: placeholder values (replace with real metrics)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick tests", unsafe_allow_html=True)

    # quick sample buttons reliable (use unique keys)
    for i, (txt, lbl) in enumerate(SAMPLES):
        col = st.columns([1])[0]
        if col.button(f"Try sample {i+1} — {lbl}", key=f"right_sample_{i}"):
            st.session_state["message"] = txt
            st.experimental_rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Notes & Tips", unsafe_allow_html=True)
    st.markdown("- Use the public URL in your viva slides.")
    st.markdown("- If model predictions look off, try uploading your own `.joblib` pipeline that contains preprocessing.")
    st.markdown("- Add demo video as fallback in case of connectivity issues.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer-note'>Tip: Use Quick sample buttons or upload your own dataset. Want a custom color/theme? Ask and I'll make one.</div>", unsafe_allow_html=True)

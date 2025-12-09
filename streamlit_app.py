# streamlit_app_ui_redesign.py
# UI-redesigned version of the user's original streamlit_app.py
# All original logic, functions and behavior are preserved. Only the UI layer (layout, styling, HTML/CSS) has been replaced.

import streamlit as st
import joblib
import re
import numpy as np
import time
from pathlib import Path
import pandas as pd
import base64

# -------------------------
# Styling (custom CSS + animations)
# -------------------------
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

:root{
  --bg1: #0f1115;
  --bg2: #0b0c0f;
  --card-bg: rgba(255,255,255,0.02);
  --muted: #9aa0a6;
  --accent1: #ff7a59;
  --accent2: #7bd389;
}

html, body, [class*="css"]  {
    font-family: "Inter", sans-serif;
    color: #e6e6e6;
    background: linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 60%);
}

/* Header */
.header {
    display:flex;
    align-items:center;
    gap:18px;
}
.brand-icon {
    width:72px; height:72px;
    border-radius:18px;
    display:flex;
    align-items:center;
    justify-content:center;
    background: linear-gradient(135deg,var(--accent1),#ffb86b);
    box-shadow: 0 10px 40px rgba(255,120,90,0.12);
    font-size:34px;
}
.title {
    font-weight:800;
    font-size:34px;
    color: #fff;
    margin:0;
}
.subtitle { color: #c7c7c7; margin-top:4px; }

/* Typewriter */
.typewriter { font-weight:700; font-size:20px; color:#dfe6f0; }
.typewriter .cursor{ color: var(--accent1); margin-left:6px; animation: blink 1s infinite; }
@keyframes blink{ 50% { opacity: 0 } }

/* Card */
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.03);
    transition: transform .18s ease, box-shadow .18s ease;
}
.card:hover{ transform: translateY(-6px); box-shadow: 0 18px 60px rgba(0,0,0,0.6); }

/* Text area */
textarea[aria-label="Message"] {
    background: rgba(0,0,0,0.45) !important;
    color: #f0f0f0 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    height: 160px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    padding: 10px 18px;
    font-weight:600;
    box-shadow: 0 8px 24px rgba(0,0,0,0.45);
}

/* Result banners */
.result-success {
    background: linear-gradient(90deg,#164e37,#0d3928);
    padding:14px;border-radius:10px;color:#dff7e3;
}
.result-danger {
    background: linear-gradient(90deg,#611b1b,#3b0f0f);
    padding:14px;border-radius:10px;color:#ffd6d6;
}

/* animated probability bar */
.prob-wrap{ width:100%; background:rgba(255,255,255,0.03); border-radius:10px; padding:6px; }
.prob-bar{ height:18px; border-radius:8px; background:linear-gradient(90deg,var(--accent1), #ffb86b); width:0%; transition: width 1s ease; }
.prob-label{ font-weight:700; margin-top:8px; }

/* small helper */
.small-muted { color: var(--muted); font-size:13px; }

/* examples table */
.example-row {
    padding:10px 14px; border-radius:10px;
    background: rgba(255,255,255,0.01);
    margin-bottom:10px;
    border: 1px solid rgba(255,255,255,0.02);
}

/* sidebar tweak */
[data-testid="stSidebar"]{ background: linear-gradient(180deg,#071018, #08121a); }

/* responsive tweaks */
@media (max-width: 768px){ .title{ font-size:22px } }

</style>
"""

# -------------------------
# Utilities (cleaners + model loader)
# (Kept exactly the same as original; only comments/formatting adjusted)
# -------------------------

def clean_text(s: str):
    """Original simpler cleaner (kept for reference)."""
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_text_keep_tokens(s: str):
    """Token-preserving cleaner: replace URLs and long numbers with tokens.
    This matches the preprocessing used for the tokenized model.
    """
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' __URL__ ', s)
    s = re.sub(r'\b\d{6,}\b', ' __PHONE__ ', s)
    s = re.sub(r'[^a-z0-9_\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def load_model(path=None):
    """Load model. If path is provided, try it; otherwise try model_tokenized.joblib then model.joblib."""
    if path:
        p = Path(path)
        if p.exists():
            return joblib.load(p)
        else:
            raise FileNotFoundError(f"model file not found at: {p.resolve()}")

    p1 = Path("model_tokenized.joblib")
    p2 = Path("model.joblib")
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
# Samples (same)
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
# App layout (UI redesigned)
# -------------------------
st.set_page_config(page_title="SMS / WhatsApp Spam Detector", layout="wide", initial_sidebar_state="expanded")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Header: brand + typewriter subtitle
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.markdown('<div class="brand-icon">✉️</div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        '<div class="header"><div><h1 class="title">SMS / WhatsApp Spam Detector</h1>'
        '<div class="subtitle small-muted">Real-time spam classification — demo & presentation ready</div>'
        '<div style="height:8px"></div>'
        '<div class="typewriter" id="typewriter">Detecting inbox threats <span class="cursor">|</span></div>'
        '</div></div>', unsafe_allow_html=True)

# Small script for typewriter rotating words (pure front-end)
TYPEWRITER_JS = """
<script>
const phrases = ["Detecting inbox threats", "Catch scams automatically", "Keep your chats clean", "Demo-ready & fast"];
let i = 0;
let j = 0;
let current = '';
let isDeleting = false;
const speed = 80;
function tick(){
  const el = document.getElementById('typewriter');
  if(!el) return;
  const full = phrases[i];
  if(isDeleting){
    current = full.substring(0, j--);
  } else {
    current = full.substring(0, j++);
  }
  el.childNodes[0].textContent = current;
  if(!isDeleting && j===full.length+1){ isDeleting = true; setTimeout(tick, 900); }
  else if(isDeleting && j===0){ isDeleting=false; i=(i+1)%phrases.length; setTimeout(tick, 300); }
  else { setTimeout(tick, speed); }
}
setTimeout(tick, 800);
</script>
"""
st.components.v1.html(TYPEWRITER_JS, height=0)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar: theme toggle, model info, small avatar
with st.sidebar:
    st.markdown('<div style="text-align:center"><h3 style="margin-bottom:4px">Demo Controls</h3></div>', unsafe_allow_html=True)
    theme = st.radio("Theme", ["Dark (default)", "Light"], index=0)
    st.markdown("---")
    st.markdown("**Model loader**")
    uploaded_model = st.file_uploader("Upload a scikit-learn pipeline (.joblib)", type=["joblib", "pkl"], key="upload_sidebar")
    if uploaded_model is not None:
        tmp_path = Path("uploaded_model.joblib")
        tmp_path.write_bytes(uploaded_model.read())
        st.session_state["model_path"] = str(tmp_path.resolve())
        st.success("Uploaded model will be used for predictions.")
    st.markdown("---")
    st.checkbox("Enable debug output (show model & cleaned text)", key="debug_enabled_checkbox")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Presented by: **Abhishek Basu & team**")
    st.markdown("Repo: `sms-spam-detection`")
    st.markdown("Purpose: Final year project")

# Main content columns (improved layout)
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Enter a message")
    msg = st.text_area("Message", value="Your appointment is confirmed for Monday at 4pm.", key="message")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    # buttons row
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("Predict", key="predict"):
            st.session_state._do_predict = True
    with c2:
        if st.button("Quick: Spam sample", key="samp_spam"):
            st.session_state.message = SAMPLES[0][0]
            msg = st.session_state.message
    with c3:
        if st.button("Quick: Ham sample", key="samp_ham"):
            st.session_state.message = SAMPLES[2][0]
            msg = st.session_state.message
    with c4:
        if st.button("Random sample", key="random_sample"):
            import random
            txt, _ = random.choice(SAMPLES)
            st.session_state.message = txt
            msg = txt

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

    # Advanced options (kept content same but with UX polish)
    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: **scikit-learn Pipeline** (TF-IDF + MultinomialNB).")
        st.markdown("- Tip: paste messages containing links or phone numbers to test edge cases.")
        st.markdown("- If your model was trained with different preprocessing, results may vary.")
        st.checkbox("Show debug output (legacy checkbox)", key="debug_enabled_legacy")

    # Predict & show result (original logic preserved)
    if st.session_state.get("_do_predict", False):
        # run prediction with spinner
        with st.spinner("Analyzing message..."):
            time.sleep(0.6)
            try:
                model_path = st.session_state.get("model_path", None)
                model_path_in_use = None
                if model_path:
                    model = load_model(model_path)
                    model_path_in_use = model_path
                else:
                    model = load_model()
                    if Path("model_tokenized.joblib").exists():
                        model_path_in_use = "model_tokenized.joblib"
                    else:
                        model_path_in_use = "model.joblib"

                cleaned = clean_text_keep_tokens(msg)
                raw = clean_text(msg)

                def run_predict(m, text):
                    try:
                        pred = m.predict([text])[0]
                    except Exception as e:
                        raise RuntimeError(f"model.predict failed: {e}")
                    probv = None
                    if hasattr(m, "predict_proba"):
                        try:
                            probv = m.predict_proba([text])[0]
                        except Exception:
                            probv = None
                    return pred, probv

                pred, probv = run_predict(model, cleaned)
                prob_max = float(np.max(probv)) if probv is not None else None

                # debug outputs preserved
                if st.session_state.get("debug_enabled", False) or st.session_state.get("debug_enabled_checkbox", False) or st.session_state.get("debug_enabled_legacy", False):
                    st.write("DEBUG: model_path_in_use =", model_path_in_use)
                    st.write("DEBUG: model.classes_:", getattr(model, "classes_", None))
                    steps = getattr(model, "named_steps", None)
                    st.write("DEBUG: pipeline steps:", list(steps.keys()) if steps else None)
                    try:
                        if steps and "tfidf" in steps:
                            st.write("DEBUG: TF-IDF vocab size:", len(steps["tfidf"].vocabulary_))
                    except Exception as e:
                        st.write("DEBUG: TF-IDF read error:", e)
                    st.write("DEBUG: cleaned (token-preserving):", cleaned)
                    st.write("DEBUG: raw cleaned (no tokens):", raw)
                    st.write("DEBUG: model.predict_proba(cleaned):", probv)
                    try:
                        pred_raw, probv_raw = run_predict(model, raw)
                        st.write("DEBUG: prediction using raw cleaned text:", pred_raw, "prob:", probv_raw)
                    except Exception as e:
                        st.write("DEBUG: raw predict error:", e)

                # -------------------------
                # Interpret prediction using spam probability (preferred)
                # -------------------------
                SPAM_THRESHOLD = 0.50
                is_spam = False
                spam_prob = None

                if probv is not None:
                    try:
                        classes = getattr(model, "classes_", None)
                        if classes is not None:
                            classes_l = [str(c).lower() for c in classes]
                            try:
                                spam_idx = classes_l.index("spam")
                            except ValueError:
                                if "1" in classes_l:
                                    spam_idx = classes_l.index("1")
                                else:
                                    spam_idx = 1 if len(classes_l) > 1 else 0
                            spam_prob = float(probv[spam_idx])
                            is_spam = (spam_prob >= SPAM_THRESHOLD)
                        else:
                            is_spam = str(pred).lower() in ("spam", "1", "true", "yes")
                    except Exception:
                        try:
                            is_spam = str(pred).lower() in ("spam", "1", "true", "yes")
                        except Exception:
                            is_spam = False
                else:
                    try:
                        pred_str = str(pred).lower()
                        if pred_str in ("spam", "1", "true", "yes"):
                            is_spam = True
                        else:
                            is_spam = False
                    except Exception:
                        is_spam = False

                # show result banner (enhanced display)
                if is_spam:
                    st.markdown('<div class="result-danger"> <strong>🚨 Be Careful — this message looks SPAM</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success"> <strong>✅ Looks Safe</strong></div>', unsafe_allow_html=True)

                # show probability visualization
                if spam_prob is not None:
                    pct = int(round(spam_prob * 100))
                    st.markdown(f"<div style='margin-top:12px'><div class='small-muted'>Spam probability</div></div>", unsafe_allow_html=True)
                    # animated progress bar via CSS width change
                    bar_html = f"<div class='prob-wrap'><div class='prob-bar' id='probbar' style='width:{pct}%;'></div></div>"
                    st.markdown(bar_html, unsafe_allow_html=True)
                    st.markdown(f"<div class='prob-label'>{pct}%</div>", unsafe_allow_html=True)
                    st.write(f"Model spam probability: **{pct}%**")
                else:
                    if prob_max is not None:
                        pct = int(round(prob_max * 100))
                        st.markdown(f"<div style='margin-top:12px'><div class='small-muted'>Confidence</div></div>", unsafe_allow_html=True)
                        st.progress(pct)
                        st.write(f"Model confidence: **{pct}%**")
                    else:
                        st.write("Model confidence: **N/A**")

            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as e:
                st.error("Prediction failed: " + str(e))
        st.session_state._do_predict = False

    st.markdown('</div>', unsafe_allow_html=True)

    # Examples / demo table (card)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Example messages")
    for text, label in SAMPLES:
        label_badge = "🔴 SPAM" if label.lower()=="spam" else "🟢 HAM"
        st.markdown(f'<div class="example-row"><strong>{label_badge}</strong> &nbsp; {text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Project Info")
    st.markdown("- Model: TF-IDF + MultinomialNB")
    st.markdown("- Presented by:  **Abhishek Basu, Ananya Raj, Sneha Das, Payel Guin, Subhajit Khamrai**")
    st.markdown("- Repo: `sms-spam-detection`")
    st.markdown("- Purpose: Final year project")
    st.markdown("<br>", unsafe_allow_html=True)

    st.metric("Test Accuracy on Testing Data", "97.55%")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Quick tests")
    for i, (txt, lbl) in enumerate(SAMPLES):
        if st.button(f"Try sample {i+1}"):
            st.session_state.message = txt

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Share / Notes", unsafe_allow_html=True)
    st.markdown("- Use the public URL in your viva slides.")
    st.markdown("- Add a short demo video as fallback in case of connectivity issues.")
    st.markdown('</div>', unsafe_allow_html=True)

# Footer small note
st.markdown("<br><div class='small-muted' style='text-align:center'>Tip: Use the Quick: Spam / Ham buttons to demo fast. Want a different color scheme? Ask and I'll create it.</div>", unsafe_allow_html=True)

# End of file

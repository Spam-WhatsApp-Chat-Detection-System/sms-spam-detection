# streamlit_app.py
import streamlit as st
import joblib
import re
import numpy as np
import time
from pathlib import Path
import pandas as pd
import base64

# -------------------------
# Styling (custom CSS)
# -------------------------
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: "Inter", sans-serif;
    color: #e6e6e6;
    background: linear-gradient(180deg, #0f1115 0%, #0b0c0f 60%);
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
    background: linear-gradient(135deg,#ff7a59,#ffb86b);
    box-shadow: 0 8px 30px rgba(255,120,90,0.14);
    font-size:34px;
}
.title {
    font-weight:800;
    font-size:34px;
    color: #fff;
    margin:0;
}
.subtitle { color: #c7c7c7; margin-top:4px; }

/* Card */
.card {
    background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01));
    border-radius: 14px;
    padding: 22px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.6);
    border: 1px solid rgba(255,255,255,0.03);
}

/* Text area */
textarea[aria-label="Message"] {
    background: rgba(0,0,0,0.4) !important;
    color: #f0f0f0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    height: 160px !important;
    border: 1px solid rgba(255,255,255,0.04) !important;
}

/* Buttons */
.stButton>button {
    border-radius: 12px;
    padding: 10px 18px;
    font-weight:600;
    box-shadow: 0 6px 18px rgba(0,0,0,0.45);
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

/* small helper */
.small-muted { color: #9aa0a6; font-size:13px; }

/* examples table */
.example-row {
    padding:8px 12px; border-radius:8px;
    background: rgba(255,255,255,0.01);
    margin-bottom:8px;
    border: 1px solid rgba(255,255,255,0.02);
}

</style>
"""

# -------------------------
# Utilities (cleaners + model loader)
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
    # replace URLs with a clear token
    s = re.sub(r'(https?://\S+|www\.\S+)', ' __URL__ ', s)
    # replace phone-like long numbers (6+ digits) with a token
    s = re.sub(r'\b\d{6,}\b', ' __PHONE__ ', s)
    # remove other punctuation but keep underscores from tokens
    s = re.sub(r'[^a-z0-9_\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_model(path=None):
    """Load model. If path is provided, try it; otherwise try model_tokenized.joblib then model.joblib."""
    # if explicit path given
    if path:
        p = Path(path)
        if p.exists():
            return joblib.load(p)
        else:
            raise FileNotFoundError(f"model file not found at: {p.resolve()}")

    # try tokenized model first (preferred)
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
# App layout
# -------------------------
st.set_page_config(page_title="SMS / WhatsApp Spam Detector", layout="wide", initial_sidebar_state="auto")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Header
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.markdown('<div class="brand-icon">✉️</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header"><div><h1 class="title">SMS / WhatsApp Spam Detector</h1><div class="subtitle small-muted">Real-time spam classification — demo & presentation ready</div></div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Main content columns
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.markdown("#### Enter a message and press **Predict**", unsafe_allow_html=True)
    msg = st.text_area("Message", value="Your appointment is confirmed for Monday at 4pm.", key="message")
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,1])
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

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    # Advanced options row
    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: **scikit-learn Pipeline** (TF-IDF + MultinomialNB).")
        st.markdown("- Tip: paste messages containing links or phone numbers to test edge cases.")
        st.markdown("- If your model was trained with different preprocessing, results may vary.")

        # allow user to upload a model; also a debug toggle
        uploaded_model = st.file_uploader("Upload a scikit-learn pipeline (.joblib)", type=["joblib", "pkl"])
        if uploaded_model is not None:
            tmp_path = Path("uploaded_model.joblib")
            tmp_path.write_bytes(uploaded_model.read())
            st.session_state["model_path"] = str(tmp_path.resolve())
            st.success("Uploaded model will be used for predictions.")

        debug_enabled = st.checkbox("Enable debug output (show model & cleaned text)", value=False)
        st.session_state["debug_enabled"] = bool(debug_enabled)

    # Predict & show result
    if st.session_state.get("_do_predict", False):
        # run prediction with spinner
        with st.spinner("Analyzing message..."):
            time.sleep(0.6)  # minor delay for UX polish
            try:
                # prefer model_path set by uploaded model, else try default loader
                model_path = st.session_state.get("model_path", None)
                if model_path:
                    model = load_model(model_path)
                    model_path_in_use = model_path
                else:
                    model = load_model()  # will attempt tokenized then model.joblib
                    # find which file we used
                    if Path("model_tokenized.joblib").exists():
                        model_path_in_use = "model_tokenized.joblib"
                    else:
                        model_path_in_use = "model.joblib"

                # use token-preserving cleaner (keeps __URL__ & __PHONE__)
                cleaned = clean_text_keep_tokens(msg)

                # If debug mode is on, show debugging info first
                if st.session_state.get("debug_enabled", False):
                    st.write("DEBUG: model_path_in_use =", model_path_in_use)
                    st.write("DEBUG: model.classes_:", getattr(model, "classes_", None))
                    steps = getattr(model, "named_steps", None)
                    st.write("DEBUG: pipeline steps:", list(steps.keys()) if steps else None)
                    try:
                        if steps and "tfidf" in steps:
                            st.write("DEBUG: TF-IDF vocab size:", len(steps["tfidf"].vocabulary_))
                    except Exception as e:
                        st.write("DEBUG: TF-IDF read error:", e)
                    st.write("DEBUG: cleaned text:", cleaned)

                # if pipeline expects raw text then passing cleaned may be okay; we trained tokenized model so use cleaned
                pred = model.predict([cleaned])[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        probv = model.predict_proba([cleaned])[0]
                        prob = float(np.max(probv))
                        if st.session_state.get("debug_enabled", False):
                            st.write("DEBUG: model.predict_proba(cleaned):", probv)
                    except Exception as e:
                        if st.session_state.get("debug_enabled", False):
                            st.write("DEBUG: predict_proba error:", e)
                        prob = None

                # show result banner
                if str(pred).lower() in ("spam", "1", "true"):
                    st.markdown('<div class="result-danger"> <strong>🚨 This message is SPAM</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success"> <strong>✅ This message is NOT spam</strong></div>', unsafe_allow_html=True)

                # confidence bar
                if prob is not None:
                    pct = int(round(prob*100))
                    st.markdown(f"<div style='margin-top:12px'><div class='small-muted'>Confidence</div></div>", unsafe_allow_html=True)
                    st.progress(pct)
                    st.write(f"Model confidence: **{pct}%**")
                else:
                    st.write("Model confidence: **N/A**")

            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as e:
                st.error("Prediction failed: " + str(e))
        # reset flag so Predict button can be used again
        st.session_state._do_predict = False

    st.markdown("</div>", unsafe_allow_html=True)

    # Examples / demo table
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Example messages")
    for text, label in SAMPLES:
        label_badge = "🔴 SPAM" if label.lower()=="spam" else "🟢 HAM"
        st.markdown(f'<div class="example-row"><strong>{label_badge}</strong> &nbsp; {text}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Project Info")
    st.markdown("- Model: TF-IDF + MultinomialNB")
    st.markdown("- Presented by:  **Abhishek Basu, Ananya Raj, Sneha Das, Payel Guin, Subhajit Khamkai**")
    st.markdown("- Repo: `sms-spam-detection`")
    st.markdown("- Purpose: Final year project")
    st.markdown("<br>", unsafe_allow_html=True)

    # Show simple metrics (placeholders — replace with real values if you want)
    st.metric("Test Accuracy on Testing Data", "95.87%")
    st.markdown("<br>", unsafe_allow_html=True)

    # quick sample buttons that update text area
    st.markdown("### Quick tests")
    for i, (txt, lbl) in enumerate(SAMPLES):
        if st.button(f"Try sample {i+1}"):
            st.session_state.message = txt

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Share / Notes", unsafe_allow_html=True)
    st.markdown("- Use the public URL in your viva slides.")
    st.markdown("- Add a short demo video as fallback in case of connectivity issues.")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer small note
st.markdown("<br><div class='small-muted' style='text-align:center'>Tip: Use the Quick: Spam / Ham buttons to demo fast. Want a different color scheme? Ask and I'll create it.</div>", unsafe_allow_html=True)

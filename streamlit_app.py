# streamlit_app.py
import streamlit as st
import joblib
import re
import numpy as np
import time
from pathlib import Path

# -------------------------
# Styling (custom CSS)
# -------------------------
PAGE_CSS = """  ... (same CSS as before) ... """
# (paste your original CSS here; omitted in this snippet for brevity)

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
    if not p.exists():
        raise FileNotFoundError(f"model file not found at: {p.resolve()}")
    return joblib.load(p)

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
    st.markdown(
        '<div class="header"><div><h1 class="title">SMS / WhatsApp Spam Detector</h1>'
        '<div class="subtitle small-muted">Real-time spam classification — demo & presentation ready</div>'
        '</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('#### Enter a message and press **Predict**', unsafe_allow_html=True)

    # Let the text area be controlled by session_state directly.
    # If session_state['message'] doesn't exist yet, initialize it to a helpful default.
    if "message" not in st.session_state:
        st.session_state["message"] = "Your appointment is confirmed for Monday at 4pm."

    # The text_area uses key="message" so its value is stored in st.session_state["message"]
    msg = st.text_area("Message", key="message", height=160)

    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        if st.button("Predict", key="predict_btn"):
            st.session_state._do_predict = True
            # rerun so UI reflects any changes immediately
            st.experimental_rerun()

    with c2:
        # set sample 1 (spam)
        if st.button("Quick: Spam sample", key="quick_spam_btn"):
            st.session_state["message"] = SAMPLES[0][0]
            st.experimental_rerun()

    with c3:
        # set sample 3 (ham)
        if st.button("Quick: Ham sample", key="quick_ham_btn"):
            st.session_state["message"] = SAMPLES[2][0]
            st.experimental_rerun()

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)

    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: **scikit-learn Pipeline** (TF-IDF + MultinomialNB).")
        st.markdown("- Tip: paste messages containing links or phone numbers to test edge cases.")
        st.markdown("- If your model was trained with different preprocessing, results may vary.")

    # Predict & show result
    if st.session_state.get("_do_predict", False):
        with st.spinner("Analyzing message..."):
            time.sleep(0.6)
            try:
                model = load_model("model.joblib")
                cleaned = clean_text(st.session_state["message"])
                pred = model.predict([cleaned])[0]
                prob = None
                if hasattr(model, "predict_proba"):
                    try:
                        probv = model.predict_proba([cleaned])[0]
                        prob = float(np.max(probv))
                    except Exception:
                        prob = None

                if str(pred).lower() in ("spam", "1", "true"):
                    st.markdown('<div class="result-danger"> <strong>🚨 This message is SPAM</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success"> <strong>✅ This message is NOT spam</strong></div>', unsafe_allow_html=True)

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
    st.markdown("- Presented by:  **Abhishek Basu, Ananya Raj, Sneha Das, Payal Guin, Subhojit Khamrai**")
    st.markdown("- Repo: `sms-spam-detection`")
    st.markdown("- Purpose: Final year project")
    st.markdown("<br>", unsafe_allow_html=True)

    st.metric("Test Accuracy", "95.87%")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Quick tests")
    # Use unique keys for each button and update session_state then rerun to reflect in text area immediately
    for i, (txt, lbl) in enumerate(SAMPLES):
        if st.button(f"Try sample {i+1}", key=f"sample_btn_{i}"):
            st.session_state["message"] = txt
            st.experimental_rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Share / Notes", unsafe_allow_html=True)
    st.markdown("- Use the public URL in your viva slides.")
    st.markdown("- Add a short demo video as fallback in case of connectivity issues.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br><div class='small-muted' style='text-align:center'>Tip: Use the Quick: Spam / Ham buttons to demo fast. Want a different color scheme? Ask and I'll create it.</div>", unsafe_allow_html=True)

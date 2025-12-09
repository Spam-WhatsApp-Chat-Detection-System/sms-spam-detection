# streamlit_app.py
# Clean, robust UI version. All model logic unchanged.
import streamlit as st
import joblib
import re
import numpy as np
import time
from pathlib import Path
import pandas as pd
import base64
import random

# -------------------------
# Lightweight stable CSS (no fragile hover lifts)
# -------------------------
PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: "Inter", sans-serif; background: #0b0c0f; color: #e6e6e6; }
.container { width:100%; padding: 8px 12px; }
.header-wrap { display:flex; gap:16px; align-items:center; margin-bottom:8px; }
.brand { width:64px; height:64px; border-radius:12px; display:flex; align-items:center; justify-content:center; background: linear-gradient(135deg,#ff7a59,#ffb86b); box-shadow: 0 12px 30px rgba(0,0,0,0.6);}
.h1 { font-size:30px; margin:0; font-weight:800; }
.subtitle { color:#bfc7cd; margin-top:6px; margin-bottom:6px; font-size:13px; }
.card { background: rgba(255,255,255,0.02); border-radius:12px; padding:16px; border:1px solid rgba(255,255,255,0.03); }
.small-muted { color:#9aa0a6; font-size:13px; }
.controls-row > div { padding-right:8px; }
.textarea { border-radius:10px !important; }
.button-row .stButton>button { border-radius:10px; padding:8px 14px; font-weight:600; }
.result-success { background: linear-gradient(90deg,#164e37,#0d3928); padding:12px;border-radius:8px;color:#dff7e3; }
.result-danger { background: linear-gradient(90deg,#611b1b,#3b0f0f); padding:12px;border-radius:8px;color:#ffd6d6; }
.prob-wrap{ width:100%; background:rgba(255,255,255,0.03); border-radius:8px; padding:6px; margin-top:8px; }
.prob-bar{ height:16px; border-radius:6px; background:linear-gradient(90deg,#ff7a59,#ffb86b); width:0%; transition: width 700ms ease; }
.example-row{ padding:10px; border-radius:8px; background: rgba(255,255,255,0.01); margin-bottom:8px; border:1px solid rgba(255,255,255,0.02); }
</style>
"""

# -------------------------
# Utilities (exact same as your original logic)
# -------------------------
def clean_text(s: str):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def clean_text_keep_tokens(s: str):
    if not isinstance(s, str):
        s = str(s)
    s = s.lower()
    s = re.sub(r'(https?://\S+|www\.\S+)', ' __URL__ ', s)
    s = re.sub(r'\b\d{6,}\b', ' __PHONE__ ', s)
    s = re.sub(r'[^a-z0-9_\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_model(path=None):
    if path:
        p = Path(path)
        if p.exists():
            return joblib.load(p)
        else:
            raise FileNotFoundError(f"model file not found at: {p.resolve()}")
    p1 = Path("model_tokenized.joblib"); p2 = Path("model.joblib")
    if p1.exists(): return joblib.load(p1)
    if p2.exists(): return joblib.load(p2)
    raise FileNotFoundError(f"model file not found (tried: {p1.resolve()}, {p2.resolve()})")

def download_link(df: pd.DataFrame, filename: str):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">⬇️ Download CSV</a>'
    return href

SAMPLES = [
    ("Congratulations! You have won a FREE iPhone. Click here to claim: http://bit.ly/win-phone", "Spam"),
    ("Your parcel delivery failed. Track here: http://track-now.example", "Spam"),
    ("Are you coming to class today?", "Ham"),
    ("Don't forget the meeting at 3pm. See you there.", "Ham"),
    ("Urgent: Your account will be suspended. Call 1800-999-000", "Spam"),
    ("Happy birthday! Hope you have a great day.", "Ham"),
]

# -------------------------
# App UI
# -------------------------
st.set_page_config(page_title="SMS / WhatsApp Spam Detector", layout="wide")
st.markdown(PAGE_CSS, unsafe_allow_html=True)

# Header + typewriter (robust injection)
col1, col2 = st.columns([0.12, 0.88])
with col1:
    st.markdown('<div class="brand">✉️</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="header-wrap"><div><h1 class="h1">SMS / WhatsApp Spam Detector</h1>'
                '<div class="subtitle small-muted">Real-time spam classification — demo & presentation ready</div>'
                '<div id="typewriter" class="small-muted">Detecting inbox threats</div>'
                '</div></div>', unsafe_allow_html=True)

# Inject a small, robust typewriter script: put it inside an HTML element with height so Streamlit executes it
typewriter_html = """
<div style="height:1px;">
<script>
(function(){
  const phrases = ["Detecting inbox threats","Catch scams automatically","Keep your chats clean","Demo-ready & fast"];
  let pi=0, ci=0, deleting=false, cur='';
  function step(){
    const el = document.getElementById('typewriter');
    if(!el) return;
    const full = phrases[pi];
    if(deleting){ cur = full.substring(0, ci--); }
    else { cur = full.substring(0, ci++); }
    el.innerText = cur;
    if(!deleting && ci===full.length+1){ deleting=true; setTimeout(step,900); }
    else if(deleting && ci===0){ deleting=false; pi=(pi+1)%phrases.length; setTimeout(step,300); }
    else setTimeout(step, 80);
  }
  setTimeout(step, 400);
})();
</script>
</div>
"""
st.components.v1.html(typewriter_html, height=1)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar: uploader and toggles
with st.sidebar:
    st.markdown("## Demo Controls")
    uploaded = st.file_uploader("Upload model (.joblib)", type=["joblib","pkl"])
    if uploaded:
        p = Path("uploaded_model.joblib")
        p.write_bytes(uploaded.read())
        st.session_state["model_path"] = str(p.resolve())
        st.success("Uploaded model saved and will be used.")
    st.checkbox("Enable debug output", key="debug_enabled")

# Ensure message exists in session state so buttons can update it reliably
if "message" not in st.session_state:
    st.session_state["message"] = "Your appointment is confirmed for Monday at 4pm."

left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Enter a message")
    # text_area uses session_state key so updates to it reflect in the UI
    msg = st.text_area("Message", value=st.session_state["message"], key="message")
    st.markdown('<div class="small-muted" style="margin-top:6px">Type or paste a message, then press Predict.</div>', unsafe_allow_html=True)

    # Buttons row
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        if st.button("Predict"):
            st.session_state._do_predict = True
            # force rerun so UI updates immediately with session_state changes
            st.experimental_rerun()
    with c2:
        if st.button("Quick: Spam sample"):
            st.session_state["message"] = SAMPLES[0][0]
            st.experimental_rerun()
    with c3:
        if st.button("Quick: Ham sample"):
            st.session_state["message"] = SAMPLES[2][0]
            st.experimental_rerun()
    with c4:
        if st.button("Random sample"):
            txt, _ = random.choice(SAMPLES)
            st.session_state["message"] = txt
            st.experimental_rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Advanced options (kept)
    with st.expander("Advanced options & diagnostics", expanded=False):
        st.markdown("- Model: scikit-learn Pipeline (TF-IDF + MultinomialNB).")
        st.markdown("- Tip: paste messages containing links or phone numbers to test edge cases.")
        st.checkbox("Show full debug info (legacy)", key="debug_enabled_legacy")

    # Prediction (exact same logic; robustly triggered by session flag)
    if st.session_state.get("_do_predict", False):
        with st.spinner("Analyzing message..."):
            time.sleep(0.5)
            try:
                # load model (uploaded path preferred)
                model_path = st.session_state.get("model_path", None)
                if model_path:
                    model = load_model(model_path)
                    model_path_in_use = model_path
                else:
                    model = load_model()
                    model_path_in_use = "model_tokenized.joblib" if Path("model_tokenized.joblib").exists() else "model.joblib"

                cleaned = clean_text_keep_tokens(st.session_state["message"])
                raw = clean_text(st.session_state["message"])

                def run_predict(m, text):
                    pred = m.predict([text])[0]
                    probv = None
                    if hasattr(m, "predict_proba"):
                        try:
                            probv = m.predict_proba([text])[0]
                        except Exception:
                            probv = None
                    return pred, probv

                pred, probv = run_predict(model, cleaned)
                prob_max = float(np.max(probv)) if probv is not None else None

                if st.session_state.get("debug_enabled", False) or st.session_state.get("debug_enabled_legacy", False):
                    st.write("DEBUG: model_path_in_use =", model_path_in_use)
                    st.write("DEBUG: model.classes_:", getattr(model, "classes_", None))
                    steps = getattr(model, "named_steps", None)
                    st.write("DEBUG: pipeline steps:", list(steps.keys()) if steps else None)
                    st.write("DEBUG: cleaned (token-preserving):", cleaned)
                    st.write("DEBUG: raw cleaned (no tokens):", raw)
                    st.write("DEBUG: model.predict_proba(cleaned):", probv)

                SPAM_THRESHOLD = 0.50
                is_spam = False
                spam_prob = None

                if probv is not None:
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
                        is_spam = spam_prob >= SPAM_THRESHOLD
                    else:
                        is_spam = str(pred).lower() in ("spam", "1", "true", "yes")
                else:
                    is_spam = str(pred).lower() in ("spam", "1", "true", "yes")

                # result banner
                if is_spam:
                    st.markdown('<div class="result-danger"><strong>🚨 Be Careful — this message looks SPAM</strong></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="result-success"><strong>✅ Looks Safe</strong></div>', unsafe_allow_html=True)

                # probability visualization (always shows even if using prob_max fallback)
                if spam_prob is None and prob_max is not None:
                    spam_prob = prob_max
                if spam_prob is not None:
                    pct = int(round(spam_prob * 100))
                    st.markdown('<div class="small-muted" style="margin-top:8px">Spam probability</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="prob-wrap"><div class="prob-bar" style="width:{pct}%;"></div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div style="font-weight:700;margin-top:6px">{pct}%</div>', unsafe_allow_html=True)
                else:
                    st.write("Model confidence: **N/A**")

            except FileNotFoundError as fe:
                st.error(str(fe))
            except Exception as e:
                st.error("Prediction failed: " + str(e))

        # reset flag so user can predict again
        st.session_state._do_predict = False

    st.markdown('</div>', unsafe_allow_html=True)

    # Examples card
    st.markdown('<div class="card" style="margin-top:12px">', unsafe_allow_html=True)
    st.markdown("### Example messages")
    for text, label in SAMPLES:
        badge = "🔴 SPAM" if label.lower()=="spam" else "🟢 HAM"
        st.markdown(f'<div class="example-row"><strong>{badge}</strong> &nbsp; {text}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Project Info")
    st.markdown("- Model: TF-IDF + MultinomialNB")
    st.markdown("- Presented by: **Abhishek Basu, Ananya Raj, Sneha Das, Payel Guin, Subhajit Khamrai**")
    st.markdown("- Repo: `sms-spam-detection`")
    st.markdown("- Purpose: Final year project")
    st.markdown("<br>", unsafe_allow_html=True)
    st.metric("Test Accuracy on Testing Data", "97.55%")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Quick tests")
    for i, (txt, lbl) in enumerate(SAMPLES):
        if st.button(f"Try sample {i+1}"):
            st.session_state["message"] = txt
            st.experimental_rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# small footer hint
st.markdown("<div class='small-muted' style='text-align:center;margin-top:12px'>Tip: Use the Quick samples to demo fast. Want a brighter theme or Lottie animations? I can add them.</div>", unsafe_allow_html=True)

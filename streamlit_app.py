import streamlit as st
import joblib
import re

st.set_page_config(page_title="SMS / WhatsApp Spam Detection", layout="centered")

# Cleaning function
def clean_text(s):
    s = s.lower()
    s = re.sub(r"[^a-zA-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# Load trained pipeline
model = joblib.load("model.joblib")

st.title("📩 SMS / WhatsApp Spam Detector")
st.write("Enter a message and click Predict!")

msg = st.text_area("Message", height=150)

if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        cleaned = clean_text(msg)
        pred = model.predict([cleaned])[0]

        if pred.lower() == "spam":
            st.error("🚨 This message is **SPAM**")
        else:
            st.success("✅ This message is **NOT spam**")

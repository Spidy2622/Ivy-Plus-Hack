"""HemoSense — Help & FAQ"""
import streamlit as st
from components import inject_css, render_header, render_nav, render_footer, section_title, divider

inject_css()
render_header()
render_nav(active_key="help")

section_title("❓","Help & FAQ")
st.markdown('<div class="info-box"><h4>🚀 Getting Started</h4><p>Follow these steps for your first risk assessment:</p></div>',unsafe_allow_html=True)
st.markdown("")

for icon,title,desc in [
    ("1️⃣","Set Up AI (Optional)","Click ☰ top-right → enter Gemini API Key. Or skip and fill forms manually."),
    ("2️⃣","Describe or Fill","Use 🔬 AI Parser for NLP extraction, or go to 🎯 Risk Assessment directly."),
    ("3️⃣","Run Assessment","Fill symptoms, exposure, labs → click Predict Risk Level."),
    ("4️⃣","Review Explanation","AI generates a WHO-grounded explanation of the prediction."),
    ("5️⃣","Ask Chatbot","Visit 🤖 HemoBot to ask any CCHF question."),
]:
    st.markdown(f'<div class="info-box" style="margin-bottom:10px;"><h4>{icon} {title}</h4><p>{desc}</p></div>',unsafe_allow_html=True)

divider(); section_title("💬","Frequently Asked Questions")

for q,a in [
    ("What is CCHF?","A viral tick-borne disease with 10–40% mortality, found in Africa, Balkans, Middle East, Asia."),
    ("Is this a medical diagnosis?","**No.** Educational/research only. Always consult a healthcare professional."),
    ("How accurate is the model?","Random Forest on synthetic WHO data. Provides probability scores, not lab-confirmed diagnosis."),
    ("Do I need a Gemini API key?","Optional. Without it: manual form + prediction. With it: AI Parser + Chatbot."),
    ("Is my data saved?","No. Runs entirely in your browser session. No server storage."),
    ("What is the ☰ menu?","Hamburger menu for Display Mode (Public/Doctor) and AI configuration."),
]:
    with st.expander(q): st.markdown(a)

divider(); section_title("📧","Need More Help?")
st.markdown('<div class="info-box"><h4>Get in Touch</h4><p>Reach out via the project GitHub or contact the Ivy Plus Hackathon team.</p></div>',unsafe_allow_html=True)

render_footer()

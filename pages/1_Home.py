"""HemoSense — Home Page"""
import streamlit as st
from components import inject_css, render_header, render_nav, render_footer, divider, PAGE_RISK, PAGE_CHATBOT, PAGE_AI_PARSER

inject_css()
render_header()
render_nav(active_key="app")

st.markdown("""
<div class="hero">
    <h1>HemoSense</h1>
    <p class="tagline">Intelligent CCHF Risk Assessment — Powered by AI & WHO Guidelines</p>
    <p class="sub-tagline">Crimean-Congo Hemorrhagic Fever clinical decision support with machine learning,
    natural language processing, and real-time WHO knowledge retrieval.</p>
</div>
""", unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    c1, c2 = st.columns(2)
    with c1:
        st.page_link(PAGE_RISK, label="🎯  Start Risk Assessment", use_container_width=True)
    with c2:
        st.page_link(PAGE_CHATBOT, label="🤖  Ask WHO Chatbot", use_container_width=True)

divider()

st.markdown("""
<div class="feature-grid">
    <div class="feature-card">
        <span class="card-icon">🔬</span>
        <h3>AI Symptom Parser</h3>
        <p>Describe your symptoms in plain language and let our Gemini-powered NLP engine
        automatically extract clinical features — fever duration, tick exposure, bleeding,
        and 15+ WHO-aligned variables — in seconds.</p>
    </div>
    <div class="feature-card">
        <span class="card-icon">🎯</span>
        <h3>Risk Assessment</h3>
        <p>Enter 26 WHO-aligned clinical, exposure, and laboratory features into our
        Random Forest model to receive an instant risk level (Low / Medium / High)
        and predicted disease stage with confidence scores and gauge visualisation.</p>
    </div>
    <div class="feature-card">
        <span class="card-icon">🤖</span>
        <h3>WHO Knowledge Chatbot</h3>
        <p>Ask any question about CCHF and get answers grounded in official WHO fact sheets.
        Our RAG system ensures responses are accurate, referenced, and non-diagnostic.</p>
    </div>
    <div class="feature-card">
        <span class="card-icon">👨‍⚕️</span>
        <h3>Doctor & Public Modes</h3>
        <p>Toggle between a simplified patient-friendly interface and a detailed
        clinician view showing complete exposure history, temporal context,
        lab breakdowns, and confidence intervals via the ☰ menu.</p>
    </div>
    <div class="feature-card">
        <span class="card-icon">📄</span>
        <h3>PDF Clinical Reports</h3>
        <p>Generate downloadable PDF reports featuring patient information,
        risk assessment results with colour-coded severity, clinical recommendations,
        and timestamped disclaimers — ready for medical records.</p>
    </div>
    <div class="feature-card">
        <span class="card-icon">🗺️</span>
        <h3>Regional Risk Mapping</h3>
        <p>Visualise endemic risk across 10 geographic regions — from low-risk
        North America to high-endemic Turkey and the Balkans. Region selection
        automatically adjusts risk and endemic level features.</p>
    </div>
</div>
""", unsafe_allow_html=True)

fc1, fc2, fc3 = st.columns(3)
with fc1:
    st.page_link(PAGE_AI_PARSER, label="🔬 Open AI Parser →", use_container_width=True)
with fc2:
    st.page_link(PAGE_RISK, label="🎯 Assess Risk →", use_container_width=True)
with fc3:
    st.page_link(PAGE_CHATBOT, label="🤖 Start Chat →", use_container_width=True)

divider()

st.markdown("""
<div style="display:flex; justify-content:center; gap:60px; padding:16px 0 24px 0; flex-wrap:wrap;">
    <div style="text-align:center;">
        <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:900; background:linear-gradient(135deg,#38bdf8,#a78bfa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">26</div>
        <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">WHO Features</div>
    </div>
    <div style="text-align:center;">
        <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:900; background:linear-gradient(135deg,#38bdf8,#a78bfa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">10</div>
        <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">Regions Covered</div>
    </div>
    <div style="text-align:center;">
        <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:900; background:linear-gradient(135deg,#38bdf8,#a78bfa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">2</div>
        <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">ML Models</div>
    </div>
    <div style="text-align:center;">
        <div style="font-family:'Inter',sans-serif; font-size:2rem; font-weight:900; background:linear-gradient(135deg,#38bdf8,#a78bfa);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;">AI</div>
        <div style="font-family:'Inter',sans-serif; font-size:0.75rem; color:#64748b; text-transform:uppercase; letter-spacing:1.5px;">Gemini Powered</div>
    </div>
</div>
""", unsafe_allow_html=True)

render_footer()

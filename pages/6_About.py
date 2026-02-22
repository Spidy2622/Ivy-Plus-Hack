"""HemoSense — About"""
import streamlit as st
from components import inject_css, render_header, render_nav, render_footer, section_title, divider

inject_css()
render_header()
render_nav(active_key="about")

section_title("ℹ️","About HemoSense")
st.markdown('<div class="info-box"><h4>🩸 What is HemoSense?</h4><p>HemoSense is a <strong style="color:#38bdf8">WHO-aligned clinical decision support system</strong> for rapid risk assessment of Crimean-Congo Hemorrhagic Fever (CCHF). It combines ML models trained on 26 WHO-recommended features with AI-powered NLP and a knowledge retrieval chatbot grounded in official WHO fact sheets.</p></div>',unsafe_allow_html=True)
st.markdown("")
st.markdown('<div class="info-box"><h4>⚠️ Why CCHF Matters</h4><p>CCHF is a tick-borne viral disease with a <strong style="color:#fb7185">mortality rate of 10–40%</strong>, endemic in parts of Africa, Balkans, Middle East, and Asia. Early detection is critical but often under-diagnosed due to non-specific early symptoms.</p></div>',unsafe_allow_html=True)

divider(); section_title("🛠️","Technology Stack")
st.markdown("""
<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">
    <div class="info-box"><h4>🤖 Machine Learning</h4><p>Two Random Forest classifiers for risk level and disease stage prediction.</p></div>
    <div class="info-box"><h4>✨ NLP Engine</h4><p>Google Gemini AI extracts 18 clinical variables from free-text descriptions.</p></div>
    <div class="info-box"><h4>📚 RAG Chatbot</h4><p>Retrieval-Augmented Generation using WHO CCHF fact sheets.</p></div>
    <div class="info-box"><h4>📊 Visualisation</h4><p>Interactive Plotly gauge charts and confidence metrics.</p></div>
    <div class="info-box"><h4>🐍 Framework</h4><p>Built with Streamlit featuring multi-page routing and responsive design.</p></div>
    <div class="info-box"><h4>📄 Reporting</h4><p>ReportLab PDF generation for downloadable clinical reports.</p></div>
</div>
""",unsafe_allow_html=True)

divider(); section_title("🏥","WHO Alignment")
st.markdown('<div class="info-box"><h4>26 WHO-Aligned Features</h4><p><strong style="color:#34d399">Symptoms:</strong> Fever, Bleeding, Headache, Muscle Pain, Vomiting, Dizziness, Neck Pain, Photophobia, Abdominal Pain, Diarrhea<br><strong style="color:#fbbf24">Exposure:</strong> Tick Bite, Livestock Contact, Slaughterhouse, Healthcare Setting, CCHF Patient Contact<br><strong style="color:#fb7185">Clinical:</strong> Thrombocytopenia, Leukopenia, Elevated AST/ALT, Liver Impairment, Shock<br><strong style="color:#a78bfa">Context:</strong> Occupation Risk, Region Risk, Endemic Level, Temporal Factors</p></div>',unsafe_allow_html=True)

divider(); section_title("🏆","Credits")
st.markdown('<div class="info-box"><h4>Ivy Plus Hackathon 2026</h4><p>HemoSense was developed for the <strong style="color:#38bdf8">Ivy Plus Hackathon 2026</strong> to create an accessible AI-powered clinical decision support tool.</p></div>',unsafe_allow_html=True)

render_footer()

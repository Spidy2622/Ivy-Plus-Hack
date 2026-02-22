"""
HemoSense — Main Application Router
Uses st.navigation + st.Page for multi-page routing.
"""
import streamlit as st

st.set_page_config(
    page_title="HemoSense",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── Define all pages using st.Page ──
home_page = st.Page("pages/1_Home.py", title="Home", icon="🏠", default=True)
ai_parser_page = st.Page("pages/2_AI_Symptom_Parser.py", title="AI Parser", icon="🔬")
risk_page = st.Page("pages/3_Risk_Assessment.py", title="Risk Assessment", icon="🎯")
chatbot_page = st.Page("pages/4_HemoBot.py", title="HemoBot", icon="🤖")
account_page = st.Page("pages/5_Account.py", title="Account", icon="👤")
about_page = st.Page("pages/6_About.py", title="About", icon="ℹ️")
help_page = st.Page("pages/7_Help.py", title="Help", icon="❓")
outbreak_page = st.Page("pages/8_Outbreak_Simulation.py", title="Outbreak Sim", icon="📊")

# ── Register with st.navigation ──
pg = st.navigation([
    home_page,
    ai_parser_page,
    risk_page,
    chatbot_page,
    outbreak_page,
    account_page,
    about_page,
    help_page,
], position="hidden")

# ── Run selected page ──
pg.run()

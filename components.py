"""
HemoSense — Shared UI Components
Header, navigation, footer, CSS, and AI setup for all pages.
"""
import streamlit as st
import os

# Page file paths (must match registration in app.py)
PAGE_HOME = "pages/1_Home.py"
PAGE_AI_PARSER = "pages/2_AI_Symptom_Parser.py"
PAGE_RISK = "pages/3_Risk_Assessment.py"
PAGE_CHATBOT = "pages/4_WHO_Chatbot.py"
PAGE_ACCOUNT = "pages/5_Account.py"
PAGE_ABOUT = "pages/6_About.py"
PAGE_HELP = "pages/7_Help.py"

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    header[data-testid="stHeader"] {display:none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}
    .block-container {padding-top: 0 !important; max-width: 1200px;}

    :root {
        --bg-card: #111827;
        --accent: #38bdf8;
        --accent-glow: rgba(56, 189, 248, 0.20);
        --border-subtle: rgba(56, 189, 248, 0.12);
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-muted: #64748b;
        --gradient-header: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
        --gradient-accent: linear-gradient(135deg, #38bdf8, #a78bfa);
    }
    .hs-header {
        background: var(--gradient-header);
        padding: 16px 36px;
        display: flex; align-items: center; justify-content: space-between;
        border-bottom: 1px solid var(--border-subtle);
        margin: -1rem -1rem 0 -1rem;
        z-index: 999; box-shadow: 0 4px 30px rgba(0,0,0,0.4);
    }
    .hs-header .brand { display: flex; align-items: center; gap: 14px; text-decoration: none; }
    .hs-header .brand-icon { font-size: 2rem; filter: drop-shadow(0 0 10px var(--accent-glow)); }
    .hs-header .brand h1 {
        margin: 0; font-family: 'Inter', sans-serif; font-weight: 900; font-size: 1.5rem;
        background: var(--gradient-accent); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; letter-spacing: -0.5px;
    }
    .hs-header .brand p {
        margin: 0; font-family: 'Inter', sans-serif; font-size: 0.7rem;
        color: var(--text-muted); letter-spacing: 2px; text-transform: uppercase;
    }
    .hs-footer {
        background: var(--gradient-header); margin: 50px -1rem -1rem -1rem;
        padding: 40px 36px 20px 36px; border-top: 1px solid var(--border-subtle);
        font-family: 'Inter', sans-serif;
    }
    .footer-grid { display: grid; grid-template-columns: 1.8fr 1fr 1.5fr; gap: 40px; margin-bottom: 28px; }
    .footer-col h4 { font-size: 0.8rem; font-weight: 700; color: var(--accent); margin: 0 0 14px 0; text-transform: uppercase; letter-spacing: 2px; }
    .footer-col p { font-size: 0.78rem; color: var(--text-muted); line-height: 1.8; margin: 0; }
    .footer-bottom { border-top: 1px solid rgba(100,116,139,0.15); padding-top: 18px; text-align: center; font-size: 0.72rem; color: var(--text-muted); }
    .hs-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--accent), transparent); border: none; margin: 28px 0; opacity: 0.5; }
    .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 30px 0; }
    .feature-card {
        background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 16px;
        padding: 28px 24px; transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden;
    }
    .feature-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--gradient-accent); opacity: 0; transition: opacity 0.3s; }
    .feature-card:hover { border-color: rgba(56,189,248,0.3); transform: translateY(-4px); box-shadow: 0 12px 40px rgba(56,189,248,0.1); }
    .feature-card:hover::before { opacity: 1; }
    .feature-card .card-icon { font-size: 2rem; margin-bottom: 14px; display: block; }
    .feature-card h3 { font-family: 'Inter', sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--text-primary); margin: 0 0 8px 0; }
    .feature-card p { font-family: 'Inter', sans-serif; font-size: 0.82rem; color: var(--text-muted); line-height: 1.65; margin: 0; }
    .hero { text-align: center; padding: 60px 20px 40px 20px; }
    .hero h1 { font-family: 'Inter', sans-serif; font-weight: 900; font-size: 3.2rem; background: var(--gradient-accent); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 12px 0; letter-spacing: -1px; }
    .hero .tagline { font-family: 'Inter', sans-serif; font-size: 1.15rem; color: var(--text-secondary); margin: 0 0 8px 0; }
    .hero .sub-tagline { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: var(--text-muted); margin: 0 0 32px 0; }
    .section-title { font-family: 'Inter', sans-serif; font-size: 1.35rem; font-weight: 800; color: var(--text-primary); margin: 24px 0 6px 0; display: flex; align-items: center; gap: 10px; }
    .section-title .icon { font-size: 1.4rem; }
    .section-underline { height: 3px; width: 60px; background: var(--gradient-accent); border: none; border-radius: 2px; margin: 0 0 20px 0; }
    .info-box { background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 20px 24px; margin: 12px 0; }
    .info-box h4 { font-family: 'Inter', sans-serif; font-weight: 700; color: var(--text-primary); margin: 0 0 8px 0; }
    .info-box p { font-family: 'Inter', sans-serif; font-size: 0.85rem; color: var(--text-secondary); line-height: 1.7; margin: 0; }
    [data-testid="stPopover"] button { font-size: 1.6rem !important; background: transparent !important; border: none !important; color: var(--text-muted) !important; padding: 0 !important; cursor: pointer; }
    [data-testid="stPopover"] button:hover { color: var(--accent) !important; }
    html { scroll-behavior: smooth; }
    @media (max-width: 768px) { .feature-grid { grid-template-columns: 1fr; } .footer-grid { grid-template-columns: 1fr; } .hero h1 { font-size: 2.2rem; } }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
    <div class="hs-header">
        <div class="brand">
            <span class="brand-icon">🩸</span>
            <div><h1>HemoSense</h1><p>WHO-Aligned CCHF Decision Support</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    hdr_spacer, hdr_menu = st.columns([11, 1])
    with hdr_menu:
        with st.popover("☰"):
            st.markdown("#### ⚙️ Settings")
            mode = st.radio("Display Mode", ["Public", "Doctor"], index=0, key="hs_display_mode")
            st.markdown("---")
            st.markdown("#### 🤖 AI Configuration")
            api_key = st.text_input("Gemini API Key", type="password", help="Enter your Gemini API key.", key="hs_api_key_input")
            if not api_key and "GEMINI_API_KEY" in os.environ:
                api_key = os.environ["GEMINI_API_KEY"]
            elif not api_key:
                try: api_key = st.secrets["GEMINI_API_KEY"]
                except Exception: pass
            st.session_state["hs_api_key"] = api_key
            st.session_state["hs_mode"] = mode


def render_nav(active_key="app"):
    nav_items = [
        ("🏠 Home", PAGE_HOME, "app"),
        ("🔬 AI Parser", PAGE_AI_PARSER, "ai_parser"),
        ("🎯 Risk Assessment", PAGE_RISK, "risk"),
        ("🤖 WHO Chatbot", PAGE_CHATBOT, "chatbot"),
        ("👤 Account", PAGE_ACCOUNT, "account"),
        ("ℹ️ About", PAGE_ABOUT, "about"),
        ("❓ Help", PAGE_HELP, "help"),
    ]
    cols = st.columns(len(nav_items))
    for i, (label, page_path, key) in enumerate(nav_items):
        with cols[i]:
            st.page_link(page_path, label=label, use_container_width=True)


def render_footer():
    st.markdown("""
    <div class="hs-footer">
        <div class="footer-grid">
            <div class="footer-col">
                <h4>🩸 HemoSense</h4>
                <p>A WHO-aligned clinical decision support system for CCHF risk assessment,
                powered by machine learning and official WHO guidelines.</p>
            </div>
            <div class="footer-col"><h4>Navigation</h4><p>Use the nav bar above to access all features.</p></div>
            <div class="footer-col">
                <h4>Disclaimer</h4>
                <p>This tool is for <strong style="color:#cbd5e1">educational and research purposes only</strong>.
                Always consult a qualified healthcare professional.</p>
            </div>
        </div>
        <div class="footer-bottom">🩸 &copy; 2026 HemoSense &mdash; Ivy Plus Hackathon &middot; All rights reserved</div>
    </div>
    """, unsafe_allow_html=True)


def setup_ai():
    try:
        from google import genai
        HAS_GENAI = True
    except ImportError:
        HAS_GENAI = False
    api_key = st.session_state.get("hs_api_key", "")
    if HAS_GENAI and api_key:
        return True, genai.Client(api_key=api_key)
    return False, None


def section_title(icon, title):
    st.markdown(f'<div class="section-title"><span class="icon">{icon}</span>{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="hs-divider"></div>', unsafe_allow_html=True)

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
PAGE_CHATBOT = "pages/4_HemoBot.py"
PAGE_ACCOUNT = "pages/5_Account.py"
PAGE_ABOUT = "pages/6_About.py"
PAGE_HELP = "pages/7_Help.py"
PAGE_OUTBREAK = "pages/8_Outbreak_Simulation.py"

def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    header[data-testid="stHeader"] {display:none !important;}
    [data-testid="stSidebar"] {display: none !important;}
    [data-testid="collapsedControl"] {display: none !important;}

    /* Make the block container take full width with nice margins */
    .block-container {
        padding-top: 0 !important;
        max-width: 100% !important;
        padding-left: 5% !important;
        padding-right: 5% !important;
    }

    /* Core Theme Variables */
    :root {
        --bg-card: rgba(17, 24, 39, 0.6);
        --accent: #38bdf8;
        --accent-glow: rgba(56, 189, 248, 0.25);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --text-primary: #f8fafc;
        --text-secondary: #cbd5e1;
        --text-muted: #94a3b8;
        --gradient-header: linear-gradient(135deg, #020617 0%, #0f172a 100%);
        --gradient-accent: linear-gradient(135deg, #38bdf8, #818cf8);
    }
    
    /* Global Background overlay for a premium glass feel */
    .stApp {
        background-color: #0f172a;
        background-image: radial-gradient(circle at 15% 50%, rgba(56, 189, 248, 0.04), transparent 30%),
                          radial-gradient(circle at 85% 30%, rgba(129, 140, 248, 0.04), transparent 30%);
    }

    /* Header styling */
    .hs-header {
        background: var(--gradient-header);
        padding: 20px 5%;
        display: flex; align-items: center; justify-content: flex-start;
        border-bottom: 1px solid var(--border-subtle);
        margin: 0 -5vw 1.5rem -5vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        width: 100vw;
        z-index: 999;
        box-shadow: 0 10px 40px rgba(0,0,0,0.4);
    }
    .hs-header .brand { display: flex; align-items: center; gap: 18px; text-decoration: none; margin: 0 auto; max-width: 1400px; width: 100%; }
    .hs-header .brand-icon { font-size: 2.2rem; filter: drop-shadow(0 0 12px var(--accent-glow)); animation: pulse 3s infinite alternate; }
    
    @keyframes pulse {
        0% { transform: scale(1); filter: drop-shadow(0 0 10px rgba(56, 189, 248, 0.3)); }
        100% { transform: scale(1.05); filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.6)); }
    }

    .hs-header .brand .title-container { display: flex; flex-direction: column; }
    .hs-header .brand h1 {
        margin: 0; font-family: 'Inter', sans-serif; font-weight: 900; font-size: 1.8rem;
        background: var(--gradient-accent); -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; letter-spacing: -0.5px; line-height: 1.1;
    }
    .hs-header .brand p {
        margin: 0; font-family: 'Inter', sans-serif; font-size: 0.75rem;
        color: var(--text-muted); letter-spacing: 2.5px; text-transform: uppercase; font-weight: 600;
        margin-top: 2px;
    }

    /* Navigation styling wrapper */
    .nav-wrapper {
        margin-bottom: 2rem;
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border: 1px solid var(--border-subtle);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Make the page links look like premium buttons */
    a[data-testid="stPageLink-NavLink"] {
        background: transparent;
        border: 1px solid transparent;
        border-radius: 12px;
        padding: 10px 8px;
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: var(--text-secondary);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none !important;
    }
    a[data-testid="stPageLink-NavLink"] p {
        color: var(--text-secondary) !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    a[data-testid="stPageLink-NavLink"]:hover {
        background: var(--accent-glow);
        border-color: rgba(56, 189, 248, 0.3);
        transform: translateY(-2px);
    }
    a[data-testid="stPageLink-NavLink"]:hover p {
        color: var(--text-primary) !important;
    }

    /* Active state styling so the current page link looks "selected" */
    a[data-testid="stPageLink-NavLink"][aria-current="page"] {
        background: var(--accent-glow);
        border: 1px solid var(--accent);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        pointer-events: none; /* Already disabled by Streamlit, but explicit */
    }
    a[data-testid="stPageLink-NavLink"][aria-current="page"] p {
        color: var(--text-primary) !important;
    }

    /* Feature Cards - Ensure anchor wrappers remove text formatting natively */
    a.feature-card-link {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
    }

    /* Footer styling */
    .hs-footer {
        background: var(--gradient-header); 
        padding: 60px 5% 30px 5%; 
        border-top: 1px solid var(--border-subtle);
        font-family: 'Inter', sans-serif;
        margin: 60px -5vw 0 -5vw;
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -50vw;
        margin-right: -50vw;
        width: 100vw;
        box-shadow: 0 -10px 40px rgba(0,0,0,0.3);
    }
    .footer-grid { display: grid; grid-template-columns: 2fr 1fr 1.5fr; gap: 50px; margin-bottom: 40px; max-width: 1400px; margin-left: auto; margin-right: auto; }
    .footer-col h4 { font-size: 0.85rem; font-weight: 800; color: var(--accent); margin: 0 0 16px 0; text-transform: uppercase; letter-spacing: 2.5px; }
    .footer-col p { font-size: 0.85rem; color: var(--text-muted); line-height: 1.8; margin: 0; }
    .footer-bottom { border-top: 1px solid rgba(255,255,255,0.05); padding-top: 24px; text-align: center; font-size: 0.75rem; color: var(--text-muted); font-weight: 500; letter-spacing: 1px; }
    
    .hs-divider { height: 1px; background: linear-gradient(90deg, transparent, var(--accent), transparent); border: none; margin: 40px 0; opacity: 0.3; }
    
    /* Enhance Cards */
    .feature-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; margin: 40px 0; }
    .feature-card {
        background: var(--bg-card); border: 1px solid var(--border-subtle); border-radius: 20px;
        padding: 36px 30px; transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275); position: relative; overflow: hidden;
        backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .feature-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 4px; background: var(--gradient-accent); opacity: 0; transition: opacity 0.4s; }
    .feature-card:hover { border-color: rgba(56,189,248,0.4); transform: translateY(-8px); box-shadow: 0 20px 40px rgba(56,189,248,0.15); }
    .feature-card:hover::before { opacity: 1; }
    .feature-card .card-icon { font-size: 2.5rem; margin-bottom: 20px; display: block; filter: drop-shadow(0 4px 6px rgba(0,0,0,0.3)); }
    .feature-card h3 { font-family: 'Inter', sans-serif; font-size: 1.2rem; font-weight: 800; color: var(--text-primary); margin: 0 0 12px 0; letter-spacing: -0.3px; }
    .feature-card p { font-family: 'Inter', sans-serif; font-size: 0.95rem; color: var(--text-secondary); line-height: 1.6; margin: 0; font-weight: 400; }
    
    /* Hero Section */
    .hero { text-align: center; padding: 60px 20px 80px 20px; animation: fadeIn 1s ease-out; }
    @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
    .hero h1 { font-family: 'Inter', sans-serif; font-weight: 900; font-size: 4rem; background: var(--gradient-accent); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0 0 16px 0; letter-spacing: -1.5px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.2)); }
    .hero .tagline { font-family: 'Inter', sans-serif; font-size: 1.4rem; color: var(--text-secondary); margin: 0 0 16px 0; font-weight: 500; }
    .hero .sub-tagline { font-family: 'Inter', sans-serif; font-size: 1rem; color: var(--text-muted); margin: 0 0 40px 0; max-width: 800px; margin-left: auto; margin-right: auto; line-height: 1.7; }
    
    .section-title { font-family: 'Inter', sans-serif; font-size: 1.5rem; font-weight: 800; color: var(--text-primary); margin: 32px 0 8px 0; display: flex; align-items: center; gap: 12px; letter-spacing: -0.5px; }
    .section-title .icon { font-size: 1.6rem; }
    .section-underline { height: 4px; width: 70px; background: var(--gradient-accent); border: none; border-radius: 4px; margin: 0 0 24px 0; }
    
    .info-box { background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 16px; padding: 24px 28px; margin: 16px 0; border-left: 4px solid var(--accent); }
    .info-box h4 { font-family: 'Inter', sans-serif; font-weight: 800; color: var(--text-primary); margin: 0 0 10px 0; font-size: 1.05rem; }
    .info-box p { font-family: 'Inter', sans-serif; font-size: 0.9rem; color: var(--text-secondary); line-height: 1.7; margin: 0; }
    
    /* Popover Settings Button styling */
    [data-testid="stPopover"] button { 
        width: 100%;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--border-subtle) !important;
        color: var(--text-secondary) !important;
        padding: 10px !important;
        cursor: pointer;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        font-size: 1rem !important;
    }
    [data-testid="stPopover"] button:hover { 
        background: var(--accent-glow) !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
        color: var(--text-primary) !important;
    }
    
    html { scroll-behavior: smooth; }
    @media (max-width: 768px) { .feature-grid { grid-template-columns: 1fr; } .footer-grid { grid-template-columns: 1fr; } .hero h1 { font-size: 2.5rem; } }
    </style>
    """, unsafe_allow_html=True)



def render_header():
    st.markdown("""
    <div class="hs-header">
        <div class="brand">
            <span class="brand-icon">🩸</span>
            <div class="title-container">
                <h1>HemoSense</h1>
                <p>WHO-Aligned CCHF Decision Support</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_nav(active_key="app"):
    nav_items = [
        ("🏠 Home", PAGE_HOME, "app"),
        ("🔬 AI Parser", PAGE_AI_PARSER, "ai_parser"),
        ("🎯 Risk", PAGE_RISK, "risk"),
        ("🤖 HemoBot", PAGE_CHATBOT, "chatbot"),
        ("📊 Outbreak", PAGE_OUTBREAK, "outbreak"),
        ("👤 Account", PAGE_ACCOUNT, "account"),
        ("ℹ️ About", PAGE_ABOUT, "about"),
        ("❓ Help", PAGE_HELP, "help"),
    ]
    
    st.markdown('<div class="nav-wrapper">', unsafe_allow_html=True)
    cols = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])
    
    for i, (label, page_path, key) in enumerate(nav_items):
        with cols[i]:
            st.page_link(page_path, label=label, use_container_width=True)
            
    # Add Settings Popover to the last column of the nav bar for a clean design
    with cols[8]:
        with st.popover("⚙️ Settings", use_container_width=True):
            st.markdown("#### App Settings")
            mode = st.radio("Display Mode", ["Public", "Doctor"], index=0, key="hs_display_mode")
            st.markdown("---")
            st.markdown("#### 🤖 AI Configuration")
            
            # Fetch default key from secrets or env if not in session state
            default_key = "AIzaSyDjw60zzeUO0oEo-F8Ubv5LQ5eSX_wS110"
            if "GEMINI_API_KEY" in os.environ:
                default_key = os.environ["GEMINI_API_KEY"]
            else:
                try: 
                    if st.secrets.get("GEMINI_API_KEY", ""):
                        default_key = st.secrets.get("GEMINI_API_KEY", "")
                except Exception: 
                    pass
            
            # Ensure it's in session state
            if "hs_api_key" not in st.session_state or not st.session_state["hs_api_key"]:
                st.session_state["hs_api_key"] = default_key

            api_key = st.text_input("Gemini API Key", value=st.session_state["hs_api_key"], type="password", help="Enter your Gemini API key.", key="hs_api_key_input")
            
            if api_key != st.session_state["hs_api_key"]:
                st.session_state["hs_api_key"] = api_key
            st.session_state["hs_mode"] = mode
    st.markdown('</div>', unsafe_allow_html=True)


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
        <div class="footer-bottom">🩸 &copy; 2026 HemoSense &mdash; Team Logicraft &middot; All rights reserved</div>
    </div>
    """, unsafe_allow_html=True)


def setup_ai():
    try:
        from google import genai
        HAS_GENAI = True
    except ImportError:
        HAS_GENAI = False
    
    # Hardcode the API key so the user never has to enter it
    api_key = "AIzaSyDjw60zzeUO0oEo-F8Ubv5LQ5eSX_wS110"
    
    if HAS_GENAI and api_key:
        return True, genai.Client(api_key=api_key)
    return False, None


def section_title(icon, title):
    st.markdown(f'<div class="section-title"><span class="icon">{icon}</span>{title}</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-underline"></div>', unsafe_allow_html=True)

def divider():
    st.markdown('<div class="hs-divider"></div>', unsafe_allow_html=True)

"""HemoSense — Account"""
import streamlit as st
from components import inject_css, render_header, render_nav, render_footer, section_title, divider, PAGE_RISK, PAGE_AI_PARSER, PAGE_CHATBOT

inject_css()
render_header()
render_nav(active_key="account")

if "hs_users" not in st.session_state: st.session_state.hs_users={}
if "hs_logged_in" not in st.session_state: st.session_state.hs_logged_in=False
if "hs_username" not in st.session_state: st.session_state.hs_username=""

section_title("👤","Account")

if st.session_state.hs_logged_in:
    st.markdown(f'<div class="info-box" style="text-align:center;max-width:440px;margin:20px auto;"><div style="font-size:3rem;">👋</div><h4>Welcome, {st.session_state.hs_username}!</h4><p>You are logged in to HemoSense.</p></div>',unsafe_allow_html=True)
    _,cb,_ = st.columns([1,1,1])
    with cb:
        if st.button("🚪 Log Out",type="primary",use_container_width=True):
            st.session_state.hs_logged_in=False; st.session_state.hs_username=""; st.rerun()
    divider(); st.markdown("##### 🚀 Quick Actions")
    q1,q2,q3 = st.columns(3)
    with q1: st.page_link(PAGE_RISK,label="🎯 Risk Assessment",use_container_width=True)
    with q2: st.page_link(PAGE_AI_PARSER,label="🔬 AI Parser",use_container_width=True)
    with q3: st.page_link(PAGE_CHATBOT,label="🤖 HemoBot",use_container_width=True)
else:
    t1,t2 = st.tabs(["🔐 Login","📝 Register"])
    with t1:
        st.markdown("#### 🔐 Sign In")
        lu = st.text_input("Username",key="login_u"); lp = st.text_input("Password",type="password",key="login_p")
        if st.button("Sign In",type="primary",use_container_width=True,key="btn_login"):
            if lu and lp:
                if lu in st.session_state.hs_users and st.session_state.hs_users[lu]==lp:
                    st.session_state.hs_logged_in=True; st.session_state.hs_username=lu; st.rerun()
                else: st.error("❌ Invalid credentials.")
            else: st.warning("Fill in both fields.")
    with t2:
        st.markdown("#### 📝 Create Account")
        ru = st.text_input("Username",key="reg_u"); re = st.text_input("Email",key="reg_e")
        rp = st.text_input("Password",type="password",key="reg_p"); rp2 = st.text_input("Confirm Password",type="password",key="reg_p2")
        if st.button("Create Account",type="primary",use_container_width=True,key="btn_reg"):
            if not all([ru,re,rp,rp2]): st.warning("Fill all fields.")
            elif rp!=rp2: st.error("❌ Passwords don't match.")
            elif ru in st.session_state.hs_users: st.error("❌ Username taken.")
            elif len(rp)<6: st.error("❌ Password must be 6+ chars.")
            else:
                st.session_state.hs_users[ru]=rp; st.session_state.hs_logged_in=True; st.session_state.hs_username=ru; st.rerun()

render_footer()

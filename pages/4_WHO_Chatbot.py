"""HemoSense — WHO CCHF Knowledge Chatbot"""
import streamlit as st
from components import inject_css, render_header, render_nav, render_footer, section_title, divider, setup_ai

inject_css()
render_header()
render_nav(active_key="chatbot")

@st.cache_data
def load_who():
    try:
        with open('who_cchf_guidelines.txt','r') as f: return f.read()
    except: return "WHO guidelines not found."
WHO = load_who()

section_title("🤖","WHO CCHF Knowledge Assistant")
st.markdown('<p style="font-family:\'Inter\',sans-serif;font-size:0.9rem;color:#94a3b8;margin-bottom:20px;line-height:1.7;">Ask any CCHF question. Powered by <strong style="color:#cbd5e1;">RAG</strong> using official WHO fact sheets.</p>',unsafe_allow_html=True)

AI_ENABLED, ai_client = setup_ai()
if AI_ENABLED:
    if "messages" not in st.session_state: st.session_state.messages=[]
    for m in st.session_state.messages:
        with st.chat_message(m["role"]): st.markdown(m["content"])
    if prompt := st.chat_input("E.g., 'Is CCHF contagious?'"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"): st.markdown(prompt)
        with st.chat_message("assistant"):
            try:
                r = ai_client.models.generate_content(model='gemini-2.5-flash',contents=f"You are a medical assistant. Use WHO info to answer about CCHF. Do NOT diagnose. WHO INFO: {WHO}\n\nQuestion: {prompt}")
                st.markdown(r.text); st.session_state.messages.append({"role":"assistant","content":r.text})
            except Exception as e: st.error(f"Chat error: {e}")
    if not st.session_state.messages:
        divider(); st.markdown("##### 💬 Try asking...")
        q1,q2 = st.columns(2)
        with q1: st.markdown('<div class="info-box"><p>• What is CCHF?</p><p>• How is CCHF transmitted?</p><p>• What are the symptoms?</p></div>',unsafe_allow_html=True)
        with q2: st.markdown('<div class="info-box"><p>• What is the mortality rate?</p><p>• How is CCHF diagnosed?</p><p>• Treatment options?</p></div>',unsafe_allow_html=True)
else:
    st.markdown('<div class="info-box"><h4>🔑 AI Key Required</h4><p>Enter your Gemini API key in the <strong>☰ menu</strong> to enable the Chatbot.</p></div>',unsafe_allow_html=True)

render_footer()

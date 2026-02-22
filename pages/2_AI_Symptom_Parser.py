"""HemoSense — AI Symptom Parser"""
import streamlit as st
import json
from components import inject_css, render_header, render_nav, render_footer, section_title, divider, setup_ai, PAGE_RISK

inject_css()
render_header()
render_nav(active_key="ai_parser")

section_title("🔬", "AI Symptom Parser")
st.markdown('<p style="font-family:\'Inter\',sans-serif;font-size:0.9rem;color:#94a3b8;margin-bottom:20px;line-height:1.7;">Describe your symptoms in plain language. Our Gemini NLP engine extracts <strong style="color:#cbd5e1;">18 clinical variables</strong> aligned with WHO criteria.</p>', unsafe_allow_html=True)

AI_ENABLED, ai_client = setup_ai()
session_def = {"symptom_days":0,"fever":False,"fever_days":0,"bleeding":False,"bleeding_days":0,"headache":False,"muscle_pain":False,"vomiting":False,"dizziness":False,"neck_pain":False,"photophobia":False,"abdominal_pain":False,"diarrhea":False,"tick_bite":False,"days_since_tick":0,"livestock_contact":False,"slaughter_exposure":False,"healthcare_exposure":False,"human_contact":False}
for k, v in session_def.items():
    if k not in st.session_state:
        st.session_state[k] = v

if AI_ENABLED:
    user_narrative = st.text_area("✍️ Describe your situation", placeholder="Example: 'I have had a high fever and severe headache for 3 days. I was bitten by a tick about a week ago.'", height=160)
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        extract_btn = st.button("🔬  Extract Symptoms", type="primary", use_container_width=True)
    if extract_btn and user_narrative.strip():
        with st.spinner("Analyzing text with Gemini AI..."):
            from google.genai import types
            prompt = f'You are a medical NLP extractor. Extract from this text: "{user_narrative}"\nOutput ONLY valid JSON with keys: "symptom_days"(int),"fever"(bool),"fever_days"(int),"bleeding"(bool),"bleeding_days"(int),"headache"(bool),"muscle_pain"(bool),"vomiting"(bool),"dizziness"(bool),"neck_pain"(bool),"photophobia"(bool),"abdominal_pain"(bool),"diarrhea"(bool),"tick_bite"(bool),"days_since_tick"(int),"livestock_contact"(bool),"slaughter_exposure"(bool),"healthcare_exposure"(bool),"human_contact"(bool). If not mentioned, use false/0.'
            try:
                response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
                extracted = json.loads(response.text)
                for k in extracted:
                    if k in st.session_state: st.session_state[k] = extracted[k]
                st.success("✅ Extracted! Data available on Risk Assessment page.")
                divider()
                section_title("📋", "Extracted Data")
                c1,c2,c3 = st.columns(3)
                with c1:
                    st.markdown("**Symptoms**")
                    for key in ["fever","headache","bleeding","muscle_pain","vomiting","dizziness","neck_pain","photophobia","abdominal_pain","diarrhea"]:
                        st.markdown(f"{'🔴' if extracted.get(key) else '⚪'} {key.replace('_',' ').title()}: **{extracted.get(key,False)}**")
                with c2:
                    st.markdown("**Timing**")
                    for key in ["symptom_days","fever_days","bleeding_days","days_since_tick"]:
                        st.metric(key.replace("_"," ").title(), extracted.get(key,0))
                with c3:
                    st.markdown("**Exposure**")
                    for key in ["tick_bite","livestock_contact","slaughter_exposure","healthcare_exposure","human_contact"]:
                        st.markdown(f"{'🟠' if extracted.get(key) else '⚪'} {key.replace('_',' ').title()}: **{extracted.get(key,False)}**")
                divider()
                st.page_link(PAGE_RISK, label="🎯  Continue to Risk Assessment →", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")
    elif extract_btn:
        st.warning("Please enter a description first.")
else:
    st.markdown('<div class="info-box"><h4>🔑 AI Key Required</h4><p>Enter your Gemini API key in the <strong>☰ menu</strong> (top-right) to enable the AI Parser.</p></div>', unsafe_allow_html=True)

render_footer()

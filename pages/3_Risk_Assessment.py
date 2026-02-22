"""HemoSense — Risk Assessment"""
import streamlit as st
import pickle
import numpy as np
import plotly.graph_objects as go
from components import inject_css, render_header, render_nav, render_footer, section_title, divider, setup_ai

inject_css()
render_header()
render_nav(active_key="risk")

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        m1 = pickle.load(f)
    with open('stage_model.pkl', 'rb') as f:
        m2 = pickle.load(f)
    return m1, m2

model_risk, model_stage = load_models()
OCC = {"Urban/Office Worker":0.05,"Farmer":0.30,"Butcher":0.40,"Healthcare Worker":0.35}
REG = {"USA North":(0.10,0),"USA South":(0.20,0),"Canada":(0.05,0),"Latin America":(0.15,0),"Western Europe":(0.30,0),"Spain":(0.50,1),"Eastern Europe":(0.60,1),"Greece":(0.70,2),"Balkans":(0.80,2),"Turkey":(0.90,2)}

sd = {"symptom_days":0,"fever":False,"fever_days":0,"bleeding":False,"bleeding_days":0,"headache":False,"muscle_pain":False,"vomiting":False,"dizziness":False,"neck_pain":False,"photophobia":False,"abdominal_pain":False,"diarrhea":False,"tick_bite":False,"days_since_tick":0,"livestock_contact":False,"slaughter_exposure":False,"healthcare_exposure":False,"human_contact":False}
for k,v in sd.items():
    if k not in st.session_state: st.session_state[k]=v

section_title("🎯", "Risk Assessment")
st.markdown('<p style="font-family:\'Inter\',sans-serif;font-size:0.9rem;color:#94a3b8;margin-bottom:20px;line-height:1.7;">Enter symptoms, exposure, and lab data. Model uses <strong style="color:#cbd5e1;">26 WHO-aligned features</strong>.</p>', unsafe_allow_html=True)

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown("##### 🩺 Symptoms & Timing")
    symptom_days = st.number_input("Days since onset",min_value=0,max_value=30,value=st.session_state.symptom_days)
    fever = st.checkbox("Fever",value=st.session_state.fever)
    fever_days = st.number_input("Fever Duration",min_value=0,max_value=30,value=st.session_state.fever_days,disabled=not fever)
    bleeding = st.checkbox("Bleeding",value=st.session_state.bleeding)
    bleeding_days = st.number_input("Bleeding Duration",min_value=0,max_value=30,value=st.session_state.bleeding_days,disabled=not bleeding)
    headache = st.checkbox("Headache",value=st.session_state.headache)
    muscle_pain = st.checkbox("Muscle Pain",value=st.session_state.muscle_pain)
    vomiting = st.checkbox("Vomiting",value=st.session_state.vomiting)
    dizziness = st.checkbox("Dizziness",value=st.session_state.dizziness)
    neck_pain = st.checkbox("Neck Pain",value=st.session_state.neck_pain)
    photophobia = st.checkbox("Photophobia",value=st.session_state.photophobia)
    abdominal_pain = st.checkbox("Abdominal Pain",value=st.session_state.abdominal_pain)
    diarrhea = st.checkbox("Diarrhea",value=st.session_state.diarrhea)
with c2:
    st.markdown("##### 🐄 Exposure Factors")
    tick_bite = st.checkbox("Tick Bite",value=st.session_state.tick_bite)
    days_since_tick = st.number_input("Days since tick bite",min_value=0,max_value=60,value=st.session_state.days_since_tick,disabled=not tick_bite)
    livestock_contact = st.checkbox("Livestock Contact",value=st.session_state.livestock_contact)
    slaughter_exposure = st.checkbox("Slaughterhouse Exposure",value=st.session_state.slaughter_exposure)
    healthcare_exposure = st.checkbox("Healthcare Setting Exposure",value=st.session_state.healthcare_exposure)
    human_contact = st.checkbox("Contact with CCHF Patient",value=st.session_state.human_contact)
    contact_occurred = livestock_contact or slaughter_exposure or healthcare_exposure or human_contact
    days_since_contact = st.number_input("Days since contact",min_value=0,max_value=60,value=0,disabled=not contact_occurred)
    occupation = st.selectbox("Occupation",list(OCC.keys()))
with c3:
    st.markdown("##### 🧪 Clinical & Lab")
    platelet_count = st.number_input("Platelet Count (cells/μL)",min_value=0,max_value=500000,value=200000)
    platelet_low = platelet_count < 150000
    wbc_low = st.checkbox("Leukopenia (Low WBC)")
    ast_alt_high = st.checkbox("Elevated Liver Enzymes (AST/ALT)")
    liver_impairment = st.checkbox("Signs of Liver Impairment/Failure")
    shock_signs = st.checkbox("Signs of Shock")
    selected_region = st.selectbox("Geographic Region",list(REG.keys()))

divider()

if st.button("🎯  Predict Risk Level",type="primary",use_container_width=True):
    orv = OCC[occupation]
    rrv, elv = REG[selected_region]
    feat = np.array([[int(fever),int(bleeding),int(headache),int(muscle_pain),int(vomiting),int(dizziness),int(neck_pain),int(photophobia),int(abdominal_pain),int(diarrhea),int(tick_bite),int(livestock_contact),int(slaughter_exposure),int(healthcare_exposure),int(human_contact),int(platelet_low),int(wbc_low),int(ast_alt_high),int(liver_impairment),int(shock_signs),orv,rrv,elv,days_since_tick,days_since_contact,symptom_days]])
    rp = model_risk.predict(feat)[0]; rpr = model_risk.predict_proba(feat)[0]; rc = model_risk.classes_
    pd_ = {rc[i]:rpr[i] for i in range(len(rc))}; mp = pd_[rp]*100
    sp = model_stage.predict(feat)[0]; spr = model_stage.predict_proba(feat)[0]; sc = model_stage.classes_
    sd_ = {sc[i]:spr[i] for i in range(len(sc))}; smp = sd_[sp]*100

    divider(); section_title("📊","Risk & Stage Assessment")
    cr1,cr2 = st.columns(2)
    with cr1:
        gv = pd_.get(1,0)*100
        if "high" in pd_: gv = pd_["high"]*100
        fg = go.Figure(go.Indicator(mode="gauge+number",value=gv,domain={'x':[0,1],'y':[0,1]},title={'text':"CCHF Risk",'font':{'size':24,'color':'#f1f5f9'}},number={'suffix':"%",'font':{'size':40,'color':'#f1f5f9'}},gauge={'axis':{'range':[None,100],'tickcolor':"#38bdf8"},'bar':{'color':"#38bdf8"},'bgcolor':'#1e293b','steps':[{'range':[0,33],'color':'#065f46'},{'range':[33,66],'color':'#92400e'},{'range':[66,100],'color':'#991b1b'}],'threshold':{'line':{'color':"#fb7185",'width':4},'thickness':0.75,'value':gv}}))
        fg.update_layout(height=300,margin=dict(l=20,r=20,t=50,b=20),paper_bgcolor='rgba(0,0,0,0)',font_color='#f1f5f9')
        st.plotly_chart(fg,use_container_width=True); st.metric("Risk Confidence",f"{mp:.1f}%")
    with cr2:
        st.markdown(f"### Predicted Stage: **{str(sp).upper()}**"); st.metric("Stage Confidence",f"{smp:.1f}%")
        fs = go.Figure(data=[go.Bar(x=[str(k).capitalize() for k in sd_.keys()],y=[v*100 for v in sd_.values()],marker_color=['#34d399' if 'early' in str(k).lower() else '#fbbf24' if 'hemorrhagic' in str(k).lower() else '#fb7185' for k in sd_.keys()],text=[f"{v*100:.1f}%" for v in sd_.values()],textposition='auto')])
        fs.update_layout(title={'text':"Stage Probabilities",'font':{'color':'#f1f5f9'}},yaxis_range=[0,100],height=250,showlegend=False,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font_color='#94a3b8')
        st.plotly_chart(fs,use_container_width=True)

    AI_ENABLED, ai_client = setup_ai()
    if AI_ENABLED:
        divider(); section_title("💡","AI Medical Explanation")
        with st.spinner("Generating explanation..."):
            prompt = f"You are a helpful medical assistant. A ML model predicted CCHF Risk={rp} ({mp:.1f}%), Stage={sp} ({smp:.1f}%). Patient: Fever={fever}({fever_days}d), Bleeding={bleeding}({bleeding_days}d), Tick={tick_bite}({days_since_tick}d), Labs: Platelets<150k={platelet_low}, WBC Low={wbc_low}, AST/ALT High={ast_alt_high}. Write 2-3 paragraphs explaining WHY based on WHO guidelines. State it is non-diagnostic."
            try:
                ex = ai_client.models.generate_content(model='gemini-2.5-flash',contents=prompt)
                st.info(ex.text)
            except Exception as e:
                st.error(f"Error: {e}")

render_footer()

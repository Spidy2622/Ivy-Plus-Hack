"""HemoSense — Risk Assessment"""
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from components import inject_css, render_header, render_nav, render_footer, section_title, divider, setup_ai
from report_generator import generate_cchf_report, generate_report_pdf

inject_css()
render_header()
render_nav(active_key="risk")

@st.cache_resource
def load_models():
    """Load trained models and evaluation metrics."""
    try:
        with open('model_v2.pkl', 'rb') as f:
            m1 = pickle.load(f)
        with open('stage_model_v2.pkl', 'rb') as f:
            m2 = pickle.load(f)
    except FileNotFoundError:
        st.error("Model files not found. Please run `python train_model.py` first.")
        return None, None, None, None, None
    
    try:
        with open('evaluation_metrics.json', 'r') as f:
            metrics = json.load(f)
    except Exception:
        metrics = None
    
    try:
        with open('roc_data.json', 'r') as f:
            roc_data = json.load(f)
    except Exception:
        roc_data = None
        
    try:
        with open('feature_importance.json', 'r') as f:
            feature_importance = json.load(f)
    except Exception:
        feature_importance = None
    
    return m1, m2, metrics, roc_data, feature_importance

model_risk, model_stage, eval_metrics, roc_data, feature_importance = load_models()

if model_risk is None:
    st.stop()

# Occupation and Region mappings (used as model features)
OCC = {
    "Urban/Office Worker": 0.05,
    "Farmer": 0.30,
    "Butcher": 0.40,
    "Healthcare Worker": 0.35,
    "Veterinarian": 0.38,
    "Laboratory Worker": 0.32
}

REG = {
    "USA North": (0.10, 0),
    "USA South": (0.20, 0),
    "Canada": (0.05, 0),
    "Latin America": (0.15, 0),
    "Western Europe": (0.30, 0),
    "Spain": (0.50, 1),
    "Eastern Europe": (0.60, 1),
    "Greece": (0.70, 2),
    "Balkans": (0.80, 2),
    "Turkey": (0.90, 2),
    "Central Asia": (0.95, 2),
    "Middle East": (0.85, 2),
    "Africa": (0.88, 2)
}

MONTHS = {
    "January": 1, "February": 2, "March": 3, "April": 4,
    "May": 5, "June": 6, "July": 7, "August": 8,
    "September": 9, "October": 10, "November": 11, "December": 12
}

# Session state defaults
sd = {
    "symptom_days": 0, "fever": False, "fever_days": 0, "bleeding": False, "bleeding_days": 0,
    "headache": False, "muscle_pain": False, "vomiting": False, "dizziness": False,
    "neck_pain": False, "photophobia": False, "abdominal_pain": False, "diarrhea": False,
    "tick_bite": False, "days_since_tick": 0, "livestock_contact": False,
    "slaughter_exposure": False, "healthcare_exposure": False, "human_contact": False
}
for k, v in sd.items():
    if k not in st.session_state:
        st.session_state[k] = v

section_title("🎯", "Risk Assessment")
st.markdown('''<p style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#94a3b8;margin-bottom:20px;line-height:1.7;">
Enter symptoms, exposure, and lab data. Model uses <strong style="color:#cbd5e1;">28 WHO-aligned features</strong> 
including cyclical seasonal encoding. Predictions are <strong style="color:#38bdf8;">fully model-driven</strong> with no manual adjustments.
</p>''', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("##### 🩺 Symptoms & Timing")
    symptom_days = st.number_input("Days since onset", min_value=0, max_value=30, value=st.session_state.symptom_days)
    fever = st.checkbox("Fever", value=st.session_state.fever)
    fever_days = st.number_input("Fever Duration", min_value=0, max_value=30, value=st.session_state.fever_days, disabled=not fever)
    bleeding = st.checkbox("Bleeding", value=st.session_state.bleeding)
    bleeding_days = st.number_input("Bleeding Duration", min_value=0, max_value=30, value=st.session_state.bleeding_days, disabled=not bleeding)
    headache = st.checkbox("Headache", value=st.session_state.headache)
    muscle_pain = st.checkbox("Muscle Pain", value=st.session_state.muscle_pain)
    vomiting = st.checkbox("Vomiting", value=st.session_state.vomiting)
    dizziness = st.checkbox("Dizziness", value=st.session_state.dizziness)
    neck_pain = st.checkbox("Neck Pain", value=st.session_state.neck_pain)
    photophobia = st.checkbox("Photophobia", value=st.session_state.photophobia)
    abdominal_pain = st.checkbox("Abdominal Pain", value=st.session_state.abdominal_pain)
    diarrhea = st.checkbox("Diarrhea", value=st.session_state.diarrhea)

with c2:
    st.markdown("##### 🐄 Exposure Factors")
    tick_bite = st.checkbox("Tick Bite", value=st.session_state.tick_bite)
    days_since_tick = st.number_input("Days since tick bite", min_value=0, max_value=60, value=st.session_state.days_since_tick, disabled=not tick_bite)
    livestock_contact = st.checkbox("Livestock Contact", value=st.session_state.livestock_contact)
    slaughter_exposure = st.checkbox("Slaughterhouse Exposure", value=st.session_state.slaughter_exposure)
    healthcare_exposure = st.checkbox("Healthcare Setting Exposure", value=st.session_state.healthcare_exposure)
    human_contact = st.checkbox("Contact with CCHF Patient", value=st.session_state.human_contact)
    contact_occurred = livestock_contact or slaughter_exposure or healthcare_exposure or human_contact
    days_since_contact = st.number_input("Days since contact", min_value=0, max_value=60, value=0, disabled=not contact_occurred)
    occupation = st.selectbox("Occupation", list(OCC.keys()))
    selected_month = st.selectbox("Month of Symptom Onset", list(MONTHS.keys()), index=6)

with c3:
    st.markdown("##### 🧪 Clinical & Lab")
    platelet_count = st.number_input("Platelet Count (cells/μL)", min_value=0, max_value=500000, value=200000)
    platelet_low = platelet_count < 150000
    wbc_low = st.checkbox("Leukopenia (Low WBC)")
    ast_alt_high = st.checkbox("Elevated Liver Enzymes (AST/ALT)")
    liver_impairment = st.checkbox("Signs of Liver Impairment/Failure")
    shock_signs = st.checkbox("Signs of Shock")
    selected_region = st.selectbox("Geographic Region", list(REG.keys()))
    
    st.markdown("---")
    st.markdown("##### 📊 Lab Indicators")
    if platelet_low:
        st.warning("⚠️ Low Platelets (<150k)")
    if wbc_low:
        st.warning("⚠️ Leukopenia Detected")
    if ast_alt_high:
        st.warning("⚠️ Elevated Liver Enzymes")

divider()

if st.button("🎯  Predict Risk Level", type="primary", use_container_width=True):
    # Get feature values
    orv = OCC[occupation]
    rrv, elv = REG[selected_region]
    month_num = MONTHS[selected_month]
    
    # Cyclical encoding for month
    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)
    
    # Build feature vector (28 features)
    feat = np.array([[
        int(fever), int(bleeding), int(headache), int(muscle_pain), int(vomiting),
        int(dizziness), int(neck_pain), int(photophobia), int(abdominal_pain), int(diarrhea),
        int(tick_bite), int(livestock_contact), int(slaughter_exposure), int(healthcare_exposure), int(human_contact),
        int(platelet_low), int(wbc_low), int(ast_alt_high), int(liver_impairment), int(shock_signs),
        orv, rrv, elv,
        days_since_tick, days_since_contact, symptom_days,
        month_sin, month_cos
    ]])
    
    # Risk prediction (direct from model, no adjustments)
    rp = model_risk.predict(feat)[0]
    rpr = model_risk.predict_proba(feat)[0]
    rc = model_risk.classes_
    pd_ = {rc[i]: rpr[i] for i in range(len(rc))}
    mp = pd_[rp] * 100
    
    # Stage prediction
    sp = model_stage.predict(feat)[0]
    spr = model_stage.predict_proba(feat)[0]
    sc = model_stage.classes_
    sd_ = {sc[i]: spr[i] for i in range(len(sc))}
    smp = sd_[sp] * 100

    divider()
    section_title("📊", "Risk & Stage Assessment")
    
    cr1, cr2 = st.columns(2)
    
    with cr1:
        # Determine gauge value based on high risk probability
        gv = pd_.get(2, 0) * 100
        if gv == 0:
            for k, v in pd_.items():
                if "high" in str(k).lower():
                    gv = v * 100
                    break
        if gv == 0:
            gv = max(pd_.values()) * 100

        # Risk label mapping
        risk_labels = {0: "Low", 1: "Medium", 2: "High"}
        pred_label = risk_labels.get(rp, str(rp).title())
        
        st.markdown(f"### Predicted Risk: **{pred_label.upper()}**")
        
        fg = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gv,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "CCHF Risk (Severity)", 'font': {'size': 24, 'color': '#f1f5f9'}},
            number={'suffix': "%", 'font': {'size': 40, 'color': '#f1f5f9'}},
            gauge={
                'axis': {'range': [None, 100], 'tickcolor': "#38bdf8"},
                'bar': {'color': "#38bdf8"},
                'bgcolor': '#1e293b',
                'steps': [
                    {'range': [0, 33], 'color': '#065f46'},
                    {'range': [33, 66], 'color': '#92400e'},
                    {'range': [66, 100], 'color': '#991b1b'}
                ],
                'threshold': {'line': {'color': "#fb7185", 'width': 4}, 'thickness': 0.75, 'value': gv}
            }
        ))
        fg.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f5f9')
        st.plotly_chart(fg, use_container_width=True)
        st.metric("Model Confidence", f"{mp:.1f}%")
        
        # Probability distribution
        st.markdown("#### Risk Probability Distribution")
        prob_df = pd.DataFrame([
            {'Risk Level': risk_labels.get(k, str(k)), 'Probability': v * 100}
            for k, v in pd_.items()
        ])
        fig_prob = px.bar(prob_df, x='Risk Level', y='Probability', 
                          color='Risk Level',
                          color_discrete_map={'Low': '#34d399', 'Medium': '#fbbf24', 'High': '#fb7185'})
        fig_prob.update_layout(height=200, showlegend=False, paper_bgcolor='rgba(0,0,0,0)', 
                               plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8',
                               yaxis_range=[0, 100])
        st.plotly_chart(fig_prob, use_container_width=True)

    with cr2:
        st.markdown(f"### Predicted Stage: **{str(sp).upper()}**")
        st.metric("Stage Confidence", f"{smp:.1f}%")
        
        fs = go.Figure(data=[go.Bar(
            x=[str(k).capitalize() for k in sd_.keys()],
            y=[v * 100 for v in sd_.values()],
            marker_color=['#34d399' if 'early' in str(k).lower() 
                         else '#fbbf24' if 'hemorrhagic' in str(k).lower() 
                         else '#fb7185' for k in sd_.keys()],
            text=[f"{v*100:.1f}%" for v in sd_.values()],
            textposition='auto'
        )])
        fs.update_layout(
            title={'text': "Stage Probabilities", 'font': {'color': '#f1f5f9'}},
            yaxis_range=[0, 100], height=250, showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8'
        )
        st.plotly_chart(fs, use_container_width=True)
        
        # Clinical summary
        st.markdown("#### Clinical Summary")
        clinical_factors = []
        if fever: clinical_factors.append(f"Fever ({fever_days}d)")
        if bleeding: clinical_factors.append(f"Bleeding ({bleeding_days}d)")
        if platelet_low: clinical_factors.append("Thrombocytopenia")
        if tick_bite: clinical_factors.append(f"Tick bite ({days_since_tick}d ago)")
        if shock_signs: clinical_factors.append("Shock signs")
        if liver_impairment: clinical_factors.append("Liver impairment")
        
        if clinical_factors:
            for factor in clinical_factors:
                st.markdown(f"• {factor}")
        else:
            st.info("No major clinical factors identified")

    # AI Explanation
    AI_ENABLED, ai_client = setup_ai()
    if AI_ENABLED:
        divider()
        section_title("💡", "AI Medical Explanation")
        with st.spinner("Generating explanation..."):
            prompt = f"""You are a medical assistant. A ML model predicted:
- CCHF Risk: {pred_label} ({mp:.1f}% confidence)
- Disease Stage: {sp} ({smp:.1f}% confidence)

Patient presentation:
- Symptoms: Fever={fever}({fever_days}d), Bleeding={bleeding}({bleeding_days}d), Headache={headache}
- Exposure: Tick bite={tick_bite}({days_since_tick}d ago), Livestock={livestock_contact}, Region={selected_region}
- Labs: Platelets<150k={platelet_low}, WBC Low={wbc_low}, AST/ALT High={ast_alt_high}
- Season: {selected_month}

Write 2-3 paragraphs explaining WHY these predictions make clinical sense based on WHO CCHF guidelines.
Emphasize this is a decision-support tool and not a diagnosis. Be specific about which factors contributed most."""
            
            try:
                ex = ai_client.models.generate_content(model='gemini-2.5-flash', contents=prompt)
                st.info(ex.text)
            except Exception as e:
                st.error(f"Error generating explanation: {e}")

    # ── CCHF Risk & Precaution Report ─────────────────────────────────
    divider()
    section_title("📄", "CCHF Risk & Precaution Report")

    # Gather patient inputs for the report generator
    _report_inputs = {
        "fever": fever, "bleeding": bleeding, "headache": headache,
        "muscle_pain": muscle_pain, "vomiting": vomiting, "dizziness": dizziness,
        "neck_pain": neck_pain, "photophobia": photophobia,
        "abdominal_pain": abdominal_pain, "diarrhea": diarrhea,
        "tick_bite": tick_bite, "livestock_contact": livestock_contact,
        "slaughter_exposure": slaughter_exposure,
        "healthcare_exposure": healthcare_exposure, "human_contact": human_contact,
        "platelet_low": platelet_low, "wbc_low": wbc_low,
        "ast_alt_high": ast_alt_high, "liver_impairment": liver_impairment,
        "shock_signs": shock_signs,
        "symptom_days": symptom_days, "days_since_tick": days_since_tick,
        "days_since_contact": days_since_contact,
        "region": selected_region, "occupation": occupation,
        "month": selected_month,
    }

    # Risk label probabilities as {"Low": p, "Medium": p, "High": p}
    _risk_labels = {0: "Low", 1: "Medium", 2: "High"}
    _risk_probs_labeled = {
        _risk_labels.get(k, str(k)): v for k, v in pd_.items()
    }
    _stage_probs_labeled = {
        str(k).capitalize(): v for k, v in sd_.items()
    }

    _fi_list = feature_importance.get("risk", []) if feature_importance else None

    _report = generate_cchf_report(
        inputs=_report_inputs,
        prediction=pred_label,
        stage=str(sp).capitalize(),
        risk_probabilities=_risk_probs_labeled,
        stage_probabilities=_stage_probs_labeled,
        feature_importance=_fi_list,
    )

    # ── Render report in-app ──────────────────────────────────────────
    # Risk level badge
    _badge_colors = {"Low": ("#065f46", "#d1fae5"), "Medium": ("#92400e", "#fef3c7"), "High": ("#991b1b", "#fee2e2")}
    _bg, _fg = _badge_colors.get(_report["risk_level"], ("#334155", "#f1f5f9"))
    st.markdown(
        f'<div style="display:inline-block;padding:6px 18px;border-radius:8px;'
        f'background:{_fg};margin-bottom:10px;">'
        f'<span style="color:{_bg};font-weight:700;font-size:1.1rem;">'
        f'Risk Level: {_report["risk_level"].upper()}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(_report["confidence_text"])

    with st.expander("👤 Patient Summary", expanded=True):
        st.markdown(_report["patient_summary"])

    with st.expander("🔑 Key Risk Factors", expanded=True):
        if _report["key_factors"]:
            for i, fct in enumerate(_report["key_factors"], 1):
                _bar_pct = min(fct["importance"] * 400, 100)  # scale for visual bar
                st.markdown(
                    f'<div style="margin-bottom:6px;">'
                    f'<span style="color:#f1f5f9;font-weight:600;">{i}. {fct["label"]}</span>'
                    f' <span style="color:#64748b;font-size:0.8rem;">(importance: {fct["importance"]:.4f})</span>'
                    f'<div style="background:#1e293b;border-radius:4px;height:6px;margin-top:3px;">'
                    f'<div style="background:#38bdf8;width:{_bar_pct:.0f}%;height:100%;border-radius:4px;"></div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No dominant risk factors identified from model importance data.")

    with st.expander("🔍 Risk Interpretation", expanded=True):
        st.markdown(_report["interpretation"])

    with st.expander("🛡️ Recommended Precautions", expanded=True):
        for p_item in _report["precautions"]:
            st.markdown(f"• {p_item}")

    with st.expander("🏥 Medical Guidance", expanded=True):
        st.markdown(_report["medical_guidance"])

    # Disclaimer
    st.markdown(
        f'<div style="margin-top:12px;padding:10px 14px;border-left:3px solid #f59e0b;'
        f'background:rgba(245,158,11,0.08);border-radius:4px;'
        f'font-size:0.8rem;color:#94a3b8;line-height:1.5;">'
        f'{_report["disclaimer"]}</div>',
        unsafe_allow_html=True,
    )

    # ── PDF download button ───────────────────────────────────────────
    _pdf_bytes = generate_report_pdf(_report)
    st.download_button(
        label="📥  Download Risk Report PDF",
        data=_pdf_bytes,
        file_name=f"CCHF_Risk_Report_{_report['timestamp'].replace(':', '-').replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# Model Transparency Dashboard
if eval_metrics:
    with st.expander("🔬 Model Evaluation & Transparency", expanded=False):
        st.markdown("""
        <div style="font-family:'Inter', sans-serif; color:#cbd5e1; margin-bottom: 20px;">
        HemoSense uses <strong>strictly model-driven predictions</strong> trained on clinical and exposure features. 
        No manual probability adjustments are applied. All predictions come directly from the model's learned patterns.
        </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "📈 ROC Curves", "🔢 Confusion Matrix", "⚡ Feature Importance"])
        
        with tab1:
            st.markdown("### Cross-Validation & Test Metrics")
            
            # Training info
            training_info = eval_metrics.get('training_info', {})
            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Training Samples", f"{training_info.get('train_samples', 'N/A'):,}")
            with col_info2:
                st.metric("Test Samples", f"{training_info.get('test_samples', 'N/A'):,}")
            with col_info3:
                st.metric("Features", training_info.get('n_features', 28))
            with col_info4:
                st.metric("CV Folds", training_info.get('cv_folds', 5))
            
            st.markdown("---")
            
            # Risk model metrics
            st.markdown("#### Risk Model Performance")
            risk_info = eval_metrics.get('risk_model', {})
            best_risk = risk_info.get('best_model', eval_metrics.get('risk_metrics', {}).get('model', 'Unknown'))
            st.caption(f"Best Model: **{best_risk}**")
            
            all_risk_results = risk_info.get('all_results', {})
            if all_risk_results:
                metrics_data = []
                for model_name, results in all_risk_results.items():
                    cv = results.get('cv_metrics', {})
                    test = results.get('test_metrics', {})
                    metrics_data.append({
                        'Model': model_name,
                        'CV Accuracy': f"{cv.get('accuracy', 0)*100:.2f}%",
                        'CV F1': f"{cv.get('f1', 0)*100:.2f}%",
                        'Test Accuracy': f"{test.get('accuracy', 0)*100:.2f}%",
                        'Test F1': f"{test.get('f1', 0)*100:.2f}%",
                        'Test AUC': f"{test.get('roc_auc', 0)*100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
            else:
                # Fallback to legacy format
                rm = eval_metrics.get('risk_metrics', {})
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("CV Accuracy", f"{rm.get('accuracy', 0)*100:.1f}%")
                with col2:
                    st.metric("CV F1 (Macro)", f"{rm.get('f1', 0)*100:.1f}%")
                with col3:
                    st.metric("CV ROC-AUC", f"{rm.get('roc_auc', 0)*100:.1f}%")
                with col4:
                    st.metric("CV Precision", f"{rm.get('precision', 0)*100:.1f}%")
            
            st.markdown("---")
            
            # Stage model metrics
            st.markdown("#### Stage Model Performance")
            stage_info = eval_metrics.get('stage_model', {})
            best_stage = stage_info.get('best_model', eval_metrics.get('stage_metrics', {}).get('model', 'Unknown'))
            st.caption(f"Best Model: **{best_stage}**")
            
            all_stage_results = stage_info.get('all_results', {})
            if all_stage_results:
                metrics_data = []
                for model_name, results in all_stage_results.items():
                    cv = results.get('cv_metrics', {})
                    test = results.get('test_metrics', {})
                    metrics_data.append({
                        'Model': model_name,
                        'CV Accuracy': f"{cv.get('accuracy', 0)*100:.2f}%",
                        'CV F1': f"{cv.get('f1', 0)*100:.2f}%",
                        'Test Accuracy': f"{test.get('accuracy', 0)*100:.2f}%",
                        'Test F1': f"{test.get('f1', 0)*100:.2f}%",
                        'Test AUC': f"{test.get('roc_auc', 0)*100:.2f}%"
                    })
                st.dataframe(pd.DataFrame(metrics_data), use_container_width=True, hide_index=True)
        
        with tab2:
            st.markdown("### ROC Curves (Test Set)")
            
            if roc_data:
                col_roc1, col_roc2 = st.columns(2)
                
                with col_roc1:
                    st.markdown("#### Risk Model ROC")
                    risk_roc = roc_data.get('risk', {})
                    if risk_roc:
                        fig_roc = go.Figure()
                        colors = {'0': '#34d399', '1': '#fbbf24', '2': '#fb7185'}
                        labels = {'0': 'Low', '1': 'Medium', '2': 'High'}
                        for cls, data in risk_roc.items():
                            fig_roc.add_trace(go.Scatter(
                                x=data['fpr'], y=data['tpr'],
                                mode='lines',
                                name=f"{labels.get(cls, cls)} (AUC={data['auc']:.3f})",
                                line=dict(color=colors.get(cls, '#38bdf8'))
                            ))
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random',
                            line=dict(color='gray', dash='dash')
                        ))
                        fig_roc.update_layout(
                            height=350,
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#94a3b8',
                            legend=dict(x=0.6, y=0.1)
                        )
                        st.plotly_chart(fig_roc, use_container_width=True)
                
                with col_roc2:
                    st.markdown("#### Stage Model ROC")
                    stage_roc = roc_data.get('stage', {})
                    if stage_roc:
                        fig_roc2 = go.Figure()
                        colors = {'Early': '#34d399', 'Hemorrhagic': '#fbbf24', 'Severe': '#fb7185'}
                        for cls, data in stage_roc.items():
                            fig_roc2.add_trace(go.Scatter(
                                x=data['fpr'], y=data['tpr'],
                                mode='lines',
                                name=f"{cls} (AUC={data['auc']:.3f})",
                                line=dict(color=colors.get(cls, '#38bdf8'))
                            ))
                        fig_roc2.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            name='Random',
                            line=dict(color='gray', dash='dash')
                        ))
                        fig_roc2.update_layout(
                            height=350,
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#94a3b8',
                            legend=dict(x=0.6, y=0.1)
                        )
                        st.plotly_chart(fig_roc2, use_container_width=True)
            else:
                st.info("ROC data not available. Run train_model.py to generate.")
        
        with tab3:
            st.markdown("### Confusion Matrices (Test Set)")
            
            col_cm1, col_cm2 = st.columns(2)
            
            with col_cm1:
                st.markdown("#### Risk Model")
                risk_info = eval_metrics.get('risk_model', {})
                cm = np.array(risk_info.get('confusion_matrix', eval_metrics.get('risk_cm', [])))
                classes = risk_info.get('classes', eval_metrics.get('risk_classes', []))
                
                if len(cm) > 0 and len(classes) > 0:
                    class_labels = ['Low', 'Medium', 'High'] if classes == [0, 1, 2] else [str(c) for c in classes]
                    fig_cm = px.imshow(
                        cm, x=class_labels, y=class_labels,
                        text_auto=True, color_continuous_scale="Blues", aspect="auto"
                    )
                    fig_cm.update_layout(
                        height=300, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f5f9',
                        xaxis_title="Predicted", yaxis_title="Actual"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            with col_cm2:
                st.markdown("#### Stage Model")
                stage_info = eval_metrics.get('stage_model', {})
                cm_stage = np.array(stage_info.get('confusion_matrix', []))
                classes_stage = stage_info.get('classes', [])
                
                if len(cm_stage) > 0 and len(classes_stage) > 0:
                    fig_cm2 = px.imshow(
                        cm_stage, x=classes_stage, y=classes_stage,
                        text_auto=True, color_continuous_scale="Purples", aspect="auto"
                    )
                    fig_cm2.update_layout(
                        height=300, margin=dict(l=0, r=0, t=30, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', font_color='#f1f5f9',
                        xaxis_title="Predicted", yaxis_title="Actual"
                    )
                    st.plotly_chart(fig_cm2, use_container_width=True)
        
        with tab4:
            st.markdown("### Feature Importance (Model-Derived)")
            st.caption("Importance scores are derived directly from the trained model, not from heuristic rules.")
            
            col_fi1, col_fi2 = st.columns(2)
            
            with col_fi1:
                st.markdown("#### Risk Model Features")
                if feature_importance:
                    risk_fi = feature_importance.get('risk', [])[:15]
                else:
                    risk_fi = eval_metrics.get('feature_importance', [])
                
                if risk_fi:
                    idf = pd.DataFrame(risk_fi).sort_values(by="importance", ascending=True)
                    fig_f = px.bar(idf, x="importance", y="feature", orientation='h')
                    fig_f.update_layout(
                        height=400, margin=dict(l=0, r=0, t=0, b=0),
                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                        font_color='#94a3b8'
                    )
                    fig_f.update_traces(marker_color='#38bdf8')
                    st.plotly_chart(fig_f, use_container_width=True)
            
            with col_fi2:
                st.markdown("#### Stage Model Features")
                if feature_importance:
                    stage_fi = feature_importance.get('stage', [])[:15]
                    if stage_fi:
                        idf2 = pd.DataFrame(stage_fi).sort_values(by="importance", ascending=True)
                        fig_f2 = px.bar(idf2, x="importance", y="feature", orientation='h')
                        fig_f2.update_layout(
                            height=400, margin=dict(l=0, r=0, t=0, b=0),
                            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            font_color='#94a3b8'
                        )
                        fig_f2.update_traces(marker_color='#a78bfa')
                        st.plotly_chart(fig_f2, use_container_width=True)
                else:
                    st.info("Stage feature importance not available.")

render_footer()

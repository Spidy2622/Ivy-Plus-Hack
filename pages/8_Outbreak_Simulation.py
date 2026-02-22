"""HemoSense — Outbreak Simulation"""
import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from components import inject_css, render_header, render_nav, render_footer, section_title, divider

inject_css()
render_header()
render_nav(active_key="outbreak")

@st.cache_resource
def load_models():
    """Load trained models."""
    try:
        with open('model_v2.pkl', 'rb') as f:
            m1 = pickle.load(f)
        with open('stage_model_v2.pkl', 'rb') as f:
            m2 = pickle.load(f)
        return m1, m2
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_dataset():
    """Load the synthetic dataset for simulation."""
    try:
        df = pd.read_csv('synthetic_cchf_who.csv')
        return df
    except FileNotFoundError:
        return None

model_risk, model_stage = load_models()
dataset = load_dataset()

section_title("📊", "Outbreak Simulation")
st.markdown('''<p style="font-family:'Inter',sans-serif;font-size:0.9rem;color:#94a3b8;margin-bottom:20px;line-height:1.7;">
Simulate outbreak scenarios by sampling patients from the dataset distribution. 
Analyze <strong style="color:#cbd5e1;">risk distributions</strong>, <strong style="color:#cbd5e1;">regional patterns</strong>, 
and <strong style="color:#cbd5e1;">seasonal trends</strong> to understand CCHF epidemiology.
</p>''', unsafe_allow_html=True)

if dataset is None or model_risk is None:
    st.error("Required files not found. Please ensure synthetic_cchf_who.csv and model files exist.")
    st.stop()

# Simulation controls
st.markdown("### Simulation Parameters")
col1, col2, col3 = st.columns(3)

with col1:
    n_patients = st.slider("Number of Patients", min_value=50, max_value=2000, value=500, step=50)

with col2:
    region_filter = st.multiselect(
        "Filter by Region",
        options=dataset['region'].unique().tolist(),
        default=[]
    )

with col3:
    occupation_filter = st.multiselect(
        "Filter by Occupation",
        options=dataset['occupation'].unique().tolist(),
        default=[]
    )

# Month selection for seasonal analysis
month_range = st.select_slider(
    "Month Range",
    options=list(range(1, 13)),
    value=(1, 12),
    format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1]
)

divider()

if st.button("🔬 Run Outbreak Simulation", type="primary", use_container_width=True):
    # Filter dataset
    df_filtered = dataset.copy()
    
    if region_filter:
        df_filtered = df_filtered[df_filtered['region'].isin(region_filter)]
    
    if occupation_filter:
        df_filtered = df_filtered[df_filtered['occupation'].isin(occupation_filter)]
    
    if len(df_filtered) < n_patients:
        st.warning(f"Filtered dataset has only {len(df_filtered)} samples. Adjusting simulation size.")
        n_patients = min(n_patients, len(df_filtered))
    
    if len(df_filtered) == 0:
        st.error("No data matches the selected filters.")
        st.stop()
    
    # Sample patients
    sample = df_filtered.sample(n=n_patients, random_state=np.random.randint(0, 10000))
    
    # Generate month distribution (CCHF peaks in summer)
    month_weights = np.array([0.03, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.10, 0.05, 0.04, 0.02])
    # Filter by month range
    month_mask = np.zeros(12)
    month_mask[month_range[0]-1:month_range[1]] = 1
    month_weights = month_weights * month_mask
    month_weights = month_weights / month_weights.sum()
    
    sample['month'] = np.random.choice(range(1, 13), size=len(sample), p=month_weights)
    sample['month_sin'] = np.sin(2 * np.pi * sample['month'] / 12)
    sample['month_cos'] = np.cos(2 * np.pi * sample['month'] / 12)
    
    # Prepare features for prediction
    feature_columns = [
        'fever', 'bleeding', 'headache', 'muscle_pain', 'vomiting', 
        'dizziness', 'neck_pain', 'photophobia', 'abdominal_pain', 'diarrhea',
        'tick_bite', 'livestock_contact', 'slaughter_exposure', 'healthcare_exposure', 'human_contact',
        'platelet_low', 'wbc_low', 'ast_alt_high', 'liver_impairment', 'shock_signs',
        'occupation_risk', 'region_risk', 'endemic_level',
        'days_since_tick', 'days_since_contact', 'symptom_days',
        'month_sin', 'month_cos'
    ]
    
    X = sample[feature_columns].values
    
    # Get predictions
    risk_preds = model_risk.predict(X)
    risk_probas = model_risk.predict_proba(X)
    stage_preds = model_stage.predict(X)
    stage_probas = model_stage.predict_proba(X)
    
    # Map predictions
    risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    sample['predicted_risk'] = [risk_labels.get(r, str(r)) for r in risk_preds]
    sample['predicted_stage'] = stage_preds
    sample['risk_confidence'] = np.max(risk_probas, axis=1) * 100
    sample['stage_confidence'] = np.max(stage_probas, axis=1) * 100
    
    # Display results
    section_title("📈", "Simulation Results")
    
    # Key metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        high_risk_pct = (sample['predicted_risk'] == 'High').mean() * 100
        st.metric("High Risk Cases", f"{high_risk_pct:.1f}%", 
                  delta=f"{int((sample['predicted_risk'] == 'High').sum())} patients")
    with col_m2:
        severe_pct = (sample['predicted_stage'] == 'Severe').mean() * 100
        st.metric("Severe Stage", f"{severe_pct:.1f}%",
                  delta=f"{int((sample['predicted_stage'] == 'Severe').sum())} patients")
    with col_m3:
        avg_conf = sample['risk_confidence'].mean()
        st.metric("Avg. Confidence", f"{avg_conf:.1f}%")
    with col_m4:
        st.metric("Total Simulated", f"{n_patients:,}")
    
    divider()
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Risk Distribution", "🗺️ Regional Analysis", "📅 Seasonal Trends", "📋 Patient Data"])
    
    with tab1:
        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("#### Risk Level Distribution")
            risk_counts = sample['predicted_risk'].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                color=risk_counts.index,
                color_discrete_map={'Low': '#34d399', 'Medium': '#fbbf24', 'High': '#fb7185'}
            )
            fig_risk.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8'
            )
            st.plotly_chart(fig_risk, use_container_width=True)
        
        with col_r2:
            st.markdown("#### Disease Stage Distribution")
            stage_counts = sample['predicted_stage'].value_counts()
            fig_stage = px.pie(
                values=stage_counts.values,
                names=stage_counts.index,
                color=stage_counts.index,
                color_discrete_map={'Early': '#34d399', 'Hemorrhagic': '#fbbf24', 'Severe': '#fb7185'}
            )
            fig_stage.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8'
            )
            st.plotly_chart(fig_stage, use_container_width=True)
        
        # Risk by occupation
        st.markdown("#### Risk Level by Occupation")
        risk_occ = pd.crosstab(sample['occupation'], sample['predicted_risk'], normalize='index') * 100
        fig_occ = px.bar(
            risk_occ.reset_index().melt(id_vars='occupation'),
            x='occupation', y='value', color='predicted_risk',
            color_discrete_map={'Low': '#34d399', 'Medium': '#fbbf24', 'High': '#fb7185'},
            labels={'value': 'Percentage (%)', 'predicted_risk': 'Risk Level'}
        )
        fig_occ.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            barmode='stack'
        )
        st.plotly_chart(fig_occ, use_container_width=True)
    
    with tab2:
        st.markdown("#### Risk Distribution by Region")
        
        # Regional risk analysis
        region_risk = sample.groupby('region').agg({
            'predicted_risk': lambda x: (x == 'High').mean() * 100,
            'region_risk': 'first'
        }).rename(columns={'predicted_risk': 'high_risk_pct'}).reset_index()
        region_risk = region_risk.sort_values('high_risk_pct', ascending=True)
        
        fig_region = px.bar(
            region_risk,
            y='region',
            x='high_risk_pct',
            orientation='h',
            color='high_risk_pct',
            color_continuous_scale=['#34d399', '#fbbf24', '#fb7185']
        )
        fig_region.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            xaxis_title='High Risk Cases (%)',
            yaxis_title='Region',
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_region, use_container_width=True)
        
        # Endemic level analysis
        st.markdown("#### Cases by Endemic Level")
        endemic_counts = sample.groupby('endemic_level').size().reset_index(name='count')
        endemic_counts['level_name'] = endemic_counts['endemic_level'].map({0: 'Non-Endemic', 1: 'Low Endemic', 2: 'High Endemic'})
        
        fig_endemic = px.bar(
            endemic_counts,
            x='level_name',
            y='count',
            color='level_name',
            color_discrete_map={'Non-Endemic': '#34d399', 'Low Endemic': '#fbbf24', 'High Endemic': '#fb7185'}
        )
        fig_endemic.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            showlegend=False
        )
        st.plotly_chart(fig_endemic, use_container_width=True)
    
    with tab3:
        st.markdown("#### Monthly Case Distribution")
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_cases = sample.groupby('month').size().reindex(range(1, 13), fill_value=0)
        
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(
            x=month_names,
            y=monthly_cases.values,
            marker_color=['#065f46' if m in [12, 1, 2] 
                         else '#92400e' if m in [3, 4, 5, 9, 10, 11]
                         else '#991b1b' for m in range(1, 13)],
            text=monthly_cases.values,
            textposition='auto'
        ))
        fig_monthly.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            xaxis_title='Month',
            yaxis_title='Number of Cases'
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
        
        # Monthly risk breakdown
        st.markdown("#### High Risk Cases by Month")
        monthly_high_risk = sample[sample['predicted_risk'] == 'High'].groupby('month').size().reindex(range(1, 13), fill_value=0)
        monthly_total = sample.groupby('month').size().reindex(range(1, 13), fill_value=0)
        monthly_high_pct = (monthly_high_risk / monthly_total * 100).fillna(0)
        
        fig_monthly_risk = go.Figure()
        fig_monthly_risk.add_trace(go.Scatter(
            x=month_names,
            y=monthly_high_pct.values,
            mode='lines+markers',
            line=dict(color='#fb7185', width=3),
            marker=dict(size=10),
            fill='tozeroy',
            fillcolor='rgba(251, 113, 133, 0.2)'
        ))
        fig_monthly_risk.update_layout(
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            xaxis_title='Month',
            yaxis_title='High Risk Cases (%)',
            yaxis_range=[0, 100]
        )
        st.plotly_chart(fig_monthly_risk, use_container_width=True)
        
        st.info("🔬 **Seasonal Pattern**: CCHF transmission peaks during warm months (May-September) when tick activity is highest.")
    
    with tab4:
        st.markdown("#### Simulated Patient Data")
        
        # Display sample data
        display_cols = ['region', 'occupation', 'predicted_risk', 'risk_confidence', 'predicted_stage', 'stage_confidence', 
                       'fever', 'bleeding', 'tick_bite', 'platelet_low']
        display_df = sample[display_cols].copy()
        display_df['risk_confidence'] = display_df['risk_confidence'].round(1).astype(str) + '%'
        display_df['stage_confidence'] = display_df['stage_confidence'].round(1).astype(str) + '%'
        
        st.dataframe(
            display_df.head(100),
            use_container_width=True,
            hide_index=True
        )
        
        if len(sample) > 100:
            st.caption(f"Showing first 100 of {len(sample)} patients")
        
        # Download option
        csv = sample.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Simulation Data",
            data=csv,
            file_name=f"outbreak_simulation_{n_patients}_patients.csv",
            mime="text/csv"
        )

# Information section
with st.expander("ℹ️ About Outbreak Simulation", expanded=False):
    st.markdown("""
    ### How It Works
    
    This simulation tool samples patients from the synthetic WHO-aligned CCHF dataset and runs them through 
    the trained ML models to predict risk levels and disease stages.
    
    **Key Features:**
    - **Random Sampling**: Patients are randomly selected from the dataset distribution
    - **Seasonal Modeling**: Month distribution follows realistic CCHF epidemiological patterns (peaks in summer)
    - **Model-Driven**: All predictions come directly from the trained GradientBoosting models
    - **Regional Analysis**: Analyze outbreak patterns across different endemic regions
    
    **Use Cases:**
    - Training and education for healthcare workers
    - Understanding CCHF risk patterns
    - Resource planning for outbreak scenarios
    - Evaluating model behavior across different populations
    
    **Limitations:**
    - Based on synthetic data, not real clinical cases
    - For educational and research purposes only
    - Should not be used for actual clinical decision-making
    """)

render_footer()

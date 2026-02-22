# CCHF Risk Prediction Tool

A comprehensive Streamlit web application for predicting Crimean-Congo Hemorrhagic Fever (CCHF) risk levels with advanced analytics and reporting.

## Features

### Core Functionality
- **Interactive symptom checklist** with duration tracking (fever_days, bleeding_days)
- **Exposure factor assessment** (tick bite, livestock contact, rural area)
- **Occupation-based risk adjustment** (Farmer, Veterinarian, Butcher, etc.)
- **Seasonal risk modeling** based on month of symptoms
- **Clinical parameters** including platelet count monitoring
- **Regional risk mapping** with endemic area identification

### Advanced Analytics
- **Risk Gauge** - Visual dial showing CCHF risk probability with color-coded zones (green/yellow/red)
- **Probability Distribution Chart** - Bar chart displaying predicted class probabilities
- **Confidence Indicator** - Model confidence percentage for predictions
- **Risk Factor Analysis** - Rule-based explanation of contributing factors
- **Regional Risk Overview** - Card-based map highlighting selected region

### Clinical Decision Support
- **Risk-stratified recommendations** (Low/Medium/High)
- **Clinical action protocols** (monitoring, testing, isolation)
- **Doctor vs Public Mode** - Toggle between simplified and detailed views
- **PDF Report Export** - Comprehensive patient report with all inputs, predictions, and recommendations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Train the model:
```bash
python train_model.py
```

3. Run the Streamlit app:
```bash
streamlit run app.py
```

## Feature Architecture

```
UI Inputs → Feature Conversion → ML Model
            ↓
   Season/Occupation Adjust
   Explanation Engine
   Advice Engine
   Charts + Gauge + Map
   PDF Export
```

## Risk Adjustment Logic

- **Occupation Risk**: Farmer (+0.15), Veterinarian (+0.20), Butcher (+0.25)
- **Seasonal Risk**: Summer (+0.10), Spring (+0.05), Winter (-0.05), Fall (0.0)
- **Platelet Threshold**: Low if < 150,000 cells/μL
- **Regional Risk Scores**: Central Asia (0.9), Africa (0.85), Eastern Europe (0.8), etc.

## Display Modes

- **Public Mode**: Simplified interface with essential information
- **Doctor Mode**: Detailed clinical data and comprehensive patient history

## PDF Report Contents

- Patient symptoms and duration
- Exposure history and occupation
- Risk assessment with confidence levels
- Probability distribution across risk categories
- Clinical recommendations
- Medical advice tailored to risk level

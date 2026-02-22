# CCHF Risk Prediction Tool - Feature Guide

## ✅ Implemented Features

### 1. Extended Input Features
- **fever_days**: Duration of fever in days (0-30)
- **bleeding_days**: Duration of bleeding in days (0-30)
- **occupation**: Dropdown with risk-weighted occupations
- **month**: Month of symptom onset for seasonal analysis
- **platelet_count**: Numeric input converted to platelet_low (< 150,000)

### 2. Risk Gauge 🎯
- Plotly gauge chart showing CCHF risk probability
- Color zones: Green (0-33%), Yellow (33-66%), Red (66-100%)
- Real-time needle indicator
- Large percentage display

### 3. Regional Risk Map 🗺️
- Card-based display of all regions
- Color-coded risk indicators (🔴🟡🟢)
- Highlighted border for selected region
- Risk scores displayed for each region

### 4. Explanation Panel 🔍
- Rule-based factor analysis
- Identifies major contributors:
  - Bleeding symptoms (major risk)
  - Prolonged fever
  - Tick bite exposure
  - Low platelet count
  - High endemic regions
  - High-risk occupations
  - Peak transmission seasons

### 5. Seasonal Risk Adjustment 🌦️
- Month → Season mapping
- Risk multipliers:
  - Summer: +0.10
  - Spring: +0.05
  - Fall: 0.0
  - Winter: -0.05

### 6. Occupation Risk Adjustment 👷
- Risk multipliers by occupation:
  - Butcher: +0.25
  - Veterinarian: +0.20
  - Farmer: +0.15
  - Healthcare Worker: +0.10
  - Other: +0.05
  - Urban/Office: 0.0

### 7. Probability Distribution Chart 📊
- Plotly bar chart
- Shows Low/Medium/High probabilities
- Color-coded bars matching risk levels
- Percentage labels on bars

### 8. Clinical Recommendation Engine 🏥
- Risk-stratified protocols:
  - **High**: Immediate hospitalization, PCR testing, isolation, ribavirin
  - **Medium**: 24-hour evaluation, lab testing, monitoring, follow-up
  - **Low**: Home monitoring, temperature tracking, symptom watch

### 9. Confidence Indicator 📈
- Displays model confidence percentage
- Based on maximum predicted probability
- Prominent metric display

### 10. Doctor vs Public Mode 👨‍⚕️
- **Public Mode**: Simplified, patient-friendly interface
- **Doctor Mode**: Detailed clinical data including:
  - Complete patient input summary
  - Exposure history breakdown
  - Temporal context (season, month)
  - All clinical parameters

### 11. PDF Report Export 📄
- Comprehensive patient report including:
  - Patient information table
  - Exposure and context data
  - Risk assessment results with color coding
  - Medical advice
  - Clinical recommendations
  - Timestamp and disclaimer
- Download button with auto-generated filename
- Professional formatting with ReportLab

## Feature Integration

All features work together seamlessly:
1. User enters symptoms and context
2. Model makes base prediction
3. Occupation and seasonal adjustments applied
4. Probabilities normalized
5. Multiple visualizations generated
6. Explanations computed from rules
7. Recommendations matched to risk level
8. PDF report available for download

## No Retraining Required

All new features use:
- Feature conversion (platelet_count → platelet_low)
- Post-prediction adjustments (occupation, season)
- Rule-based logic (explanations, recommendations)
- Visualization enhancements

The original trained model remains unchanged and fully functional.

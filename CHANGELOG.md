# Changelog

## Version 2.0.0 - Enhanced CCHF Risk Prediction Tool

### 🎉 Major Features Added

#### Data Model Extensions
- ✅ **fever_days**: Track duration of fever symptoms (0-30 days)
- ✅ **bleeding_days**: Track duration of bleeding symptoms (0-30 days)
- ✅ **occupation**: Dropdown selection with risk-weighted occupations
- ✅ **month**: Month of symptom onset for seasonal risk analysis
- ✅ **platelet_count**: Numeric input with automatic conversion to platelet_low threshold

#### Visualizations
- ✅ **Risk Gauge**: Plotly gauge chart with color-coded zones (green/yellow/red)
- ✅ **Probability Distribution Chart**: Bar chart showing Low/Medium/High probabilities
- ✅ **Regional Risk Map**: Card-based display with highlighted selected region
- ✅ **Confidence Indicator**: Model confidence percentage display

#### Analytics & Intelligence
- ✅ **Risk Factor Analysis**: Rule-based explanation of contributing factors
- ✅ **Seasonal Risk Adjustment**: Month-to-season mapping with risk multipliers
- ✅ **Occupation Risk Adjustment**: Profession-based risk weighting
- ✅ **Clinical Recommendation Engine**: Risk-stratified medical protocols

#### User Experience
- ✅ **Doctor vs Public Mode**: Toggle between simplified and detailed views
- ✅ **PDF Report Export**: Comprehensive patient report generation
- ✅ **Enhanced Layout**: Three-column design for better organization
- ✅ **Dynamic Form Fields**: Duration inputs auto-enable with symptom selection

### 🔧 Technical Improvements

#### Model Integration
- No retraining required - all features use post-prediction adjustments
- Feature conversion logic (platelet_count → platelet_low)
- Probability normalization after adjustments
- Backward compatible with existing model.pkl

#### Risk Calculation
- Occupation multipliers: Butcher (+0.25), Vet (+0.20), Farmer (+0.15)
- Seasonal multipliers: Summer (+0.10), Spring (+0.05), Winter (-0.05)
- Regional risk scores: Central Asia (0.9), Africa (0.85), Eastern Europe (0.8)
- Platelet threshold: < 150,000 cells/μL

#### Code Quality
- Modular configuration dictionaries
- Cached model loading with @st.cache_resource
- Comprehensive error handling
- Professional PDF generation with ReportLab

### 📦 Dependencies Added
- plotly: Interactive visualizations
- reportlab: PDF report generation

### 📚 Documentation
- ✅ README.md: Updated with comprehensive feature list
- ✅ FEATURES.md: Detailed feature documentation
- ✅ USAGE_GUIDE.md: Complete user guide with examples
- ✅ CHANGELOG.md: Version history
- ✅ test_app.py: Pre-flight verification script

### 🎯 Implementation Order Followed
1. ✅ Extended input features
2. ✅ Maintained backward compatibility
3. ✅ Added risk gauge visualization
4. ✅ Implemented explanation engine
5. ✅ Added probability chart
6. ✅ Applied occupation/season adjustments
7. ✅ Integrated PDF export
8. ✅ Created regional risk map

### 🔒 Safety Features
- Input validation (min/max ranges)
- Probability normalization
- Graceful degradation if model files missing
- Clear disclaimers about educational use

### 🎨 UI/UX Enhancements
- Color-coded risk indicators throughout
- Emoji icons for visual clarity
- Responsive three-column layout
- Conditional field enabling
- Professional styling with custom CSS
- Organized information hierarchy

---

## Version 1.0.0 - Initial Release

### Features
- Basic symptom checklist (fever, bleeding, headache, vomiting, muscle_pain)
- Exposure factors (tick_bite, livestock_contact, rural)
- Binary platelet_low indicator
- Region selection
- RandomForest classification
- Simple risk level prediction (Low/Medium/High)
- Basic medical advice

### Files
- train_model.py: Model training script
- app.py: Streamlit application
- requirements.txt: Dependencies
- README.md: Basic documentation

# Changelog

## Version 2.1.0 - Model-Driven Architecture

### Architecture Upgrade
- **Fully model-driven predictions** — no post-prediction probability manipulation
- All contextual risk factors (occupation, region, season) are now **input features learned during training**
- Removed legacy heuristic adjustment logic
- Direct `model.predict_proba()` output used for all predictions

### Enhanced Training Pipeline
- Added **GradientBoosting** to model comparison (now best performer)
- Implemented **80/20 train/test split** with stratification
- **5-fold cross-validation** on training set
- Comprehensive evaluation metrics saved to JSON
- ROC curve data exported for visualization
- Feature importance extracted from trained models

### New Features
- **28 input features** (up from 26)
- **Cyclical month encoding** (month_sin, month_cos) for seasonality
- **Model Transparency Dashboard** with 4 tabs:
  - Metrics: CV and test set performance
  - ROC Curves: Per-class curves with AUC
  - Confusion Matrix: Heatmaps for both models
  - Feature Importance: Model-derived rankings
- **Outbreak Simulation** page for scenario modeling
- **Stage prediction** with separate trained model

### Documentation Updates
- All documentation updated to reflect model-driven architecture
- Removed references to post-prediction multipliers
- Added sections on model validation and transparency
- Updated system diagram with accurate ML pipeline

---

## Version 2.0.0 - Enhanced CCHF Risk Prediction Tool

### Major Features Added

#### Data Model Extensions
- **fever_days**: Track duration of fever symptoms (0-30 days)
- **bleeding_days**: Track duration of bleeding symptoms (0-30 days)
- **occupation**: Dropdown selection with risk-weighted occupations
- **month**: Month of symptom onset for seasonal risk analysis
- **platelet_count**: Numeric input with automatic conversion to platelet_low threshold

#### Visualizations
- **Risk Gauge**: Plotly gauge chart with color-coded zones (green/yellow/red)
- **Probability Distribution Chart**: Bar chart showing Low/Medium/High probabilities
- **Regional Risk Map**: Card-based display with highlighted selected region
- **Confidence Indicator**: Model confidence percentage display

#### Analytics & Intelligence
- **Risk Factor Analysis**: Explanation of contributing factors
- **Seasonal Risk Encoding**: Month-to-season cyclical encoding
- **Occupation Risk Encoding**: Profession-based risk scores as features
- **Clinical Recommendation Engine**: Risk-stratified medical protocols

#### User Experience
- **Doctor vs Public Mode**: Toggle between simplified and detailed views
- **PDF Report Export**: Comprehensive patient report generation
- **Enhanced Layout**: Three-column design for better organization
- **Dynamic Form Fields**: Duration inputs auto-enable with symptom selection

### Technical Improvements

#### Model Training
- Feature conversion logic (platelet_count → platelet_low)
- Occupation and region risk scores encoded as input features
- Seasonal patterns captured via cyclical encoding
- All risk factors learned by the model during training

#### Code Quality
- Modular configuration dictionaries
- Cached model loading with @st.cache_resource
- Comprehensive error handling
- Professional PDF generation with ReportLab

### Dependencies
- plotly: Interactive visualizations
- reportlab: PDF report generation
- google-genai: AI-powered explanations

### Documentation
- README.md: Comprehensive project overview
- FEATURES.md: Detailed feature documentation
- USAGE_GUIDE.md: Complete user guide with examples
- CHANGELOG.md: Version history
- PROJECT_SUMMARY.md: Architecture documentation
- SYSTEM_DIAGRAM.txt: Visual system architecture

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

---

## Migration Notes

### From v2.0.0 to v2.1.0
- Run `python train_model.py` to regenerate models with new architecture
- Models now include month_sin/month_cos features
- Old model.pkl replaced with model_v2.pkl
- New evaluation artifacts: roc_data.json, feature_importance.json

### Key Changes
- No code changes needed for existing UI usage
- Predictions may differ slightly due to model architecture changes
- All predictions now come directly from trained model (no post-processing)

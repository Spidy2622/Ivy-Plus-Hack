# HemoSense - Feature Guide

## Implemented Features

### 1. Extended Input Features (28 Total)
- **Symptoms (10)**: fever, bleeding, headache, muscle_pain, vomiting, dizziness, neck_pain, photophobia, abdominal_pain, diarrhea
- **Exposure (5)**: tick_bite, livestock_contact, slaughter_exposure, healthcare_exposure, human_contact
- **Clinical (5)**: platelet_low, wbc_low, ast_alt_high, liver_impairment, shock_signs
- **Contextual (3)**: occupation_risk, region_risk, endemic_level
- **Temporal (3)**: days_since_tick, days_since_contact, symptom_days
- **Seasonal (2)**: month_sin, month_cos (cyclical encoding)

### 2. Risk Gauge
- Plotly gauge chart showing CCHF risk probability
- Color zones: Green (0-33%), Yellow (33-66%), Red (66-100%)
- Real-time needle indicator
- Large percentage display
- **Direct model probability** — no adjustments applied

### 3. Stage Prediction
- Separate model for disease stage classification
- Categories: Early, Hemorrhagic, Severe
- Confidence score displayed
- Bar chart showing stage probabilities

### 4. AI Explanation
- Gemini-powered clinical reasoning
- Explains prediction based on WHO guidelines
- Considers patient presentation and risk factors
- States this is decision support, not diagnosis

### 5. Contextual Risk Factors (Model-Learned)
All contextual risks are **encoded as input features** and their influence is **learned by the model during training**:

- **Occupation Risk Score** (0.05–0.40): Butcher, Farmer, Healthcare Worker, etc.
- **Region Risk Score** (0.05–0.95): Based on endemic status
- **Endemic Level** (0, 1, 2): Non-endemic, Low, High
- **Seasonal Encoding**: month_sin = sin(2π × month/12), month_cos = cos(2π × month/12)

**Important**: These are **input features**, not post-prediction multipliers.

### 6. Probability Distribution Chart
- Plotly bar chart
- Shows Low/Medium/High probabilities
- Color-coded bars matching risk levels
- Percentage labels on bars
- **Direct output from model.predict_proba()**

### 7. Clinical Recommendation Engine
- Risk-stratified protocols based on WHO guidelines:
  - **High**: Immediate hospitalization, PCR testing, isolation, ribavirin consideration
  - **Medium**: 24-hour evaluation, lab testing, monitoring, follow-up
  - **Low**: Home monitoring, temperature tracking, symptom watch

### 8. Confidence Indicator
- Displays model confidence percentage
- Based on maximum predicted probability
- **Direct from model** — no manipulation

### 9. Doctor vs Public Mode
- **Public Mode**: Simplified, patient-friendly interface
- **Doctor Mode**: Detailed clinical data including:
  - Complete patient input summary
  - Exposure history breakdown
  - Temporal context (season, month)
  - All clinical parameters

### 10. PDF Report Export
- Comprehensive patient report including:
  - Patient information table
  - Exposure and context data
  - Risk assessment results with color coding
  - Medical advice
  - Clinical recommendations
  - Timestamp and disclaimer
- Download button with auto-generated filename
- Professional formatting with ReportLab

### 11. Model Transparency Dashboard
Expandable section showing:
- **Metrics Tab**: 5-fold CV and test set metrics for all models
- **ROC Curves Tab**: Per-class ROC curves with AUC values
- **Confusion Matrix Tab**: Heatmaps for Risk and Stage models
- **Feature Importance Tab**: Model-derived feature rankings

### 12. Outbreak Simulation
- Simulate N patients from dataset distribution
- Filter by region, occupation, month range
- Visualize risk distributions
- Analyze seasonal trends
- Export simulation data as CSV

### 13. HemoBot (RAG Chatbot)
- WHO CCHF guidelines as knowledge base
- Gemini-powered question answering
- Conversation history maintained
- Suggested questions provided

## Model-Driven Architecture

### How Prediction Works

1. User inputs are converted to 28 standardized features
2. Features are passed directly to the trained model
3. Model outputs probabilities via `predict_proba()`
4. Highest probability class becomes the prediction
5. **No post-processing or heuristic adjustments**

### Feature Engineering (Pre-Model)

| Input | Conversion | Output Feature |
|-------|------------|----------------|
| Platelet count | Threshold check | platelet_low (< 150k) |
| Month | Cyclical encoding | month_sin, month_cos |
| Occupation | Lookup table | occupation_risk score |
| Region | Lookup table | region_risk, endemic_level |

### What the Model Learns

During training, the model learns:
- How symptoms correlate with risk levels
- Impact of exposure factors
- Relationship between lab values and severity
- Geographic and seasonal patterns
- Occupational risk associations

All these patterns are **learned from data**, not hard-coded rules.

## Validation

### Training Pipeline
- 80/20 train/test split with stratification
- 5-fold cross-validation on training set
- Model comparison: GradientBoosting vs RandomForest vs LogisticRegression
- Best model selected by F1-score (macro)

### Metrics Computed
- Accuracy
- Precision (macro)
- Recall (macro)
- F1-score (macro)
- ROC-AUC (multi-class)

### Transparency
All validation metrics are:
- Saved to `evaluation_metrics.json`
- Displayed in the Model Transparency Dashboard
- Available for inspection and audit

## Integration Flow

All features work together seamlessly:

```
1. User enters symptoms and context
         ↓
2. Feature engineering (28 features)
         ↓
3. Model makes prediction (direct probability)
         ↓
4. Visualizations generated (gauge, charts)
         ↓
5. AI explanation computed (Gemini)
         ↓
6. Recommendations matched to risk level
         ↓
7. PDF report available for download
```

## Key Design Principles

- **Fully model-driven**: No heuristic probability manipulation
- **Transparent**: All metrics and feature importance visible
- **Validated**: Proper train/test split and cross-validation
- **Explainable**: AI-generated clinical reasoning
- **Reproducible**: Fixed random seeds, saved artifacts

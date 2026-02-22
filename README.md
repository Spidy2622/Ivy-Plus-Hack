# HemoSense

> **WHO-Aligned Crimean-Congo Hemorrhagic Fever (CCHF) Clinical Decision Support System**

HemoSense is an intelligent clinical decision support system designed to assist healthcare professionals in early detection, risk stratification, and patient management for Crimean-Congo Hemorrhagic Fever. It utilizes a rigorous machine learning pipeline with comprehensive validation, grounded in World Health Organization (WHO) guidelines.

## Key Features

### 1. Fully Model-Driven ML Architecture
- **No manual probability adjustments** or heuristic multipliers
- Risk prediction relies entirely on trained Gradient Boosting / Random Forest classifiers
- 5-fold cross-validation with holdout test set ensures robust evaluation
- Dual prediction: Risk Level (Low/Medium/High) and Disease Stage (Early/Hemorrhagic/Severe)
- 28 WHO-aligned features including cyclical seasonal encoding

### 2. Comprehensive Model Validation
- Train/Test split (80/20) with stratified sampling
- Cross-validation metrics: Accuracy, Precision, Recall, F1-score (macro), ROC-AUC
- Model comparison: RandomForest vs LogisticRegression vs GradientBoosting
- Confusion matrices and ROC curves for both models
- Feature importance derived directly from trained models

### 3. Gemini-Powered AI Symptom Parser
- Converts unstructured patient narratives into structured clinical variables
- Extracts 18 clinical features aligned with model inputs
- Example: "I had a high fever and tick bite 3 days ago" → structured JSON

### 4. Comprehensive Risk Dashboard
- Dynamic risk gauges and probability charts using Plotly
- Real-time predictions based on clinical symptoms, exposure factors, and lab results
- AI-generated explanations based on WHO guidelines

### 5. Model Transparency Dashboard
- View cross-validation and test set metrics
- ROC curves for both Risk and Stage models
- Confusion matrix heatmaps
- Feature importance charts (model-derived, not rule-based)

### 6. Outbreak Simulation Mode
- Simulate outbreak scenarios with configurable patient counts
- Analyze risk distributions across regions and occupations
- Seasonal trend analysis
- Export simulation data for further analysis

### 7. WHO Knowledge Bot ("HemoBot")
- RAG-powered chatbot using official WHO CCHF documentation
- Answers questions about symptoms, transmission, treatment, and prevention

## Model Validation Results

| Model | CV Accuracy | CV F1 | Test AUC |
|-------|-------------|-------|----------|
| GradientBoosting | 99.28% | 99.20% | 99.99% |
| RandomForest | 98.19% | 98.19% | 99.94% |
| LogisticRegression | 94.27% | 94.27% | 99.29% |

**Note**: High metrics reflect synthetic data patterns. Real clinical deployment requires validation on actual patient data.

## Feature Engineering

### 28 WHO-Aligned Features

**Symptoms (10):**
- Fever, Bleeding, Headache, Muscle Pain, Vomiting
- Dizziness, Neck Pain, Photophobia, Abdominal Pain, Diarrhea

**Exposure (5):**
- Tick Bite, Livestock Contact, Slaughterhouse Exposure
- Healthcare Setting Exposure, CCHF Patient Contact

**Clinical/Lab (5):**
- Thrombocytopenia (Platelet < 150k), Leukopenia
- Elevated AST/ALT, Liver Impairment, Shock Signs

**Contextual Risk (3):**
- Occupation Risk Score (0.05-0.40)
- Region Risk Score (0.05-0.95)
- Endemic Level (0, 1, 2)

**Temporal (3):**
- Days Since Tick Bite
- Days Since Contact
- Symptom Duration (Days)

**Seasonal (2):**
- Month Sine (cyclical encoding)
- Month Cosine (cyclical encoding)

## Quick Start

### Prerequisites
- Python 3.10+
- Google Gemini API Key (for AI features)

### Installation

```bash
git clone https://github.com/Spidy2622/Ivy-Plus-Hack.git
cd Ivy-Plus-Hack
pip install -r requirements.txt
```

### Train Models

```bash
python train_model.py
```

This generates:
- `model_v2.pkl` - Risk prediction model
- `stage_model_v2.pkl` - Stage prediction model
- `evaluation_metrics.json` - Comprehensive metrics
- `roc_data.json` - ROC curve data
- `feature_importance.json` - Feature importance scores

### Launch Application

```bash
streamlit run app.py
```

## Architecture

```
User Input (28 features)
    ↓
Feature Engineering
    - Cyclical month encoding (sin/cos)
    - Occupation/Region risk scores
    - Platelet threshold conversion
    ↓
Validated ML Models
    - GradientBoosting (primary)
    - 5-fold CV + holdout test
    - No post-prediction adjustments
    ↓
Direct Probability Output
    - predict_proba() only
    - No heuristic modifications
    ↓
Explainable Visualization
    - Risk gauges
    - Probability distributions
    - Feature importance
    - AI explanations
    ↓
Evaluation Transparency
    - Confusion matrices
    - ROC curves
    - Cross-validation metrics
```

## Clinical Scope

### Intended Use
- **Educational tool** for CCHF risk awareness
- **Research prototype** for ML-based clinical decision support
- **Training simulator** for healthcare workers

### Limitations
- Trained on **synthetic data** based on WHO descriptions
- Not validated on real clinical cases
- Should **NOT** replace professional medical diagnosis
- High metrics reflect synthetic data patterns

## Synthetic Dataset Justification

The model is trained on a 20,000-sample synthetic dataset generated based on:
- WHO CCHF fact sheet symptoms and risk factors
- Epidemiological patterns from literature (seasonal peaks, regional endemic levels)
- Realistic feature correlations (e.g., bleeding + thrombocytopenia = higher risk)

This approach enables demonstration of ML capabilities while avoiding:
- Patient privacy concerns
- Data acquisition challenges
- Regulatory requirements for real clinical data

**For clinical deployment, the model must be retrained and validated on real patient data.**

## Technology Stack

- **Frontend**: Streamlit (multi-page navigation)
- **ML**: Scikit-learn (GradientBoosting, RandomForest, LogisticRegression)
- **Visualization**: Plotly (interactive charts)
- **NLP**: Google Gemini AI (symptom parsing, explanations)
- **RAG**: WHO guidelines for HemoBot
- **Reports**: ReportLab (PDF generation)

## Display Modes

- **Public Mode**: Simplified interface for patients
- **Doctor Mode**: Detailed clinical data and confidence intervals

## Files Structure

```
├── app.py                    # Main Streamlit router
├── components.py             # Shared UI components
├── train_model.py            # ML training pipeline
├── model_v2.pkl              # Trained risk model
├── stage_model_v2.pkl        # Trained stage model
├── evaluation_metrics.json   # Model evaluation data
├── roc_data.json             # ROC curve data
├── feature_importance.json   # Feature importance scores
├── synthetic_cchf_who.csv    # Training dataset
├── who_cchf_guidelines.txt   # WHO fact sheet for RAG
├── pages/
│   ├── 1_Home.py
│   ├── 2_AI_Symptom_Parser.py
│   ├── 3_Risk_Assessment.py
│   ├── 4_HemoBot.py
│   ├── 5_Account.py
│   ├── 6_About.py
│   ├── 7_Help.py
│   └── 8_Outbreak_Simulation.py
└── requirements.txt
```

## License & Disclaimer

This tool is for **educational and research purposes only**. It should not replace professional medical diagnosis or treatment decisions. Always consult qualified healthcare professionals for actual clinical cases.

## Acknowledgments

Built by **Team Logicraft** for the Ivy Plus Hackathon 2026.

Technologies: Streamlit, Scikit-learn, Plotly, Google Gemini AI, ReportLab

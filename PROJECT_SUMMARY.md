# HemoSense — Project Summary

## Project Overview

HemoSense is a scientifically rigorous, explainable ML clinical decision-support system for Crimean-Congo Hemorrhagic Fever (CCHF) risk assessment. The system is built with a focus on **model transparency**, **validation rigor**, and **WHO alignment**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface (Streamlit)                   │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐ │
│  │ AI Parser   │ Risk Assess │ HemoBot     │ Outbreak Sim    │ │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘ │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Feature Engineering Layer                     │
│  • 28 WHO-aligned features                                      │
│  • Cyclical month encoding (sin/cos)                            │
│  • Occupation/Region risk scores                                │
│  • No post-prediction adjustments                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Validated ML Models                           │
│  ┌────────────────────┐    ┌────────────────────┐              │
│  │ Risk Model         │    │ Stage Model        │              │
│  │ GradientBoosting   │    │ GradientBoosting   │              │
│  │ CV F1: 99.20%      │    │ CV F1: 99.20%      │              │
│  │ Test AUC: 99.99%   │    │ Test AUC: 99.99%   │              │
│  └────────────────────┘    └────────────────────┘              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Output & Explainability                       │
│  • Direct probability output (no heuristics)                    │
│  • Model-derived feature importance                             │
│  • ROC curves and confusion matrices                            │
│  • AI-generated clinical explanations                           │
└─────────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Fully Model-Driven Predictions
- **No manual probability adjustments**
- **No occupation/seasonal multipliers** applied post-prediction
- All risk factors are learned features, not heuristic rules
- Predictions come directly from `model.predict_proba()`

### 2. Rigorous Validation
- 80/20 train/test split with stratified sampling
- 5-fold cross-validation on training set
- Comparison of 3 model types (GradientBoosting, RandomForest, LogisticRegression)
- Best model selected by test F1 score
- All metrics saved for transparency

### 3. Feature Engineering
- 28 features aligned with WHO CCHF guidelines
- Cyclical encoding for month (captures seasonality without discontinuity)
- Continuous risk scores for occupation and region
- No feature is a heuristic adjustment

### 4. Explainability
- Feature importance derived from trained model
- ROC curves for each class
- Confusion matrices for error analysis
- AI-generated explanations using WHO knowledge

## Validation Metrics

### Risk Model (GradientBoosting)
| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| Accuracy | 99.28% | 99.28% |
| F1 Score (Macro) | 99.20% | 99.28% |
| Precision (Macro) | 99.20% | 99.28% |
| Recall (Macro) | 99.20% | 99.28% |
| ROC-AUC (Macro) | 99.99% | 99.99% |

### Stage Model (GradientBoosting)
| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| Accuracy | 99.28% | 99.28% |
| F1 Score (Macro) | 99.20% | 99.28% |
| ROC-AUC (Macro) | 99.99% | 99.99% |

**Note**: High metrics are expected given the synthetic data with clear patterns. Real deployment requires validation on clinical data.

## Features Implemented

### Core ML Pipeline
- [x] 28-feature model with seasonal encoding
- [x] 5-fold cross-validation
- [x] Holdout test set evaluation
- [x] Model comparison (3 algorithms)
- [x] Feature importance extraction
- [x] ROC curve generation
- [x] Confusion matrix computation

### Transparency Dashboard
- [x] Cross-validation metrics display
- [x] Test set metrics display
- [x] ROC curves visualization
- [x] Confusion matrix heatmaps
- [x] Feature importance charts
- [x] Model comparison table

### User Interface
- [x] Risk Assessment with 28 inputs
- [x] Month selection for seasonal encoding
- [x] AI Symptom Parser (Gemini NLP)
- [x] HemoBot (RAG chatbot)
- [x] Outbreak Simulation
- [x] Doctor/Public display modes
- [x] PDF report export

## Synthetic Dataset

### Justification
The 20,000-sample dataset was generated based on:
- WHO CCHF fact sheet symptom descriptions
- Epidemiological literature (seasonal patterns, endemic regions)
- Clinical correlation patterns (e.g., bleeding + low platelets = severe)

### Limitations
- Does not represent real patient data
- May not capture rare edge cases
- Risk scores are approximations

### For Clinical Deployment
The model must be:
1. Retrained on real clinical data
2. Validated by medical professionals
3. Assessed for regulatory compliance
4. Calibrated for real-world prevalence

## Files Generated by Training

| File | Description |
|------|-------------|
| `model_v2.pkl` | Trained risk prediction model |
| `stage_model_v2.pkl` | Trained stage prediction model |
| `evaluation_metrics.json` | All validation metrics |
| `roc_data.json` | ROC curve data points |
| `feature_importance.json` | Feature importance scores |

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Train models (generates all artifacts)
python train_model.py

# Launch application
streamlit run app.py
```

## Clinical Scope Statement

HemoSense is a **decision-support prototype** designed for:
- Healthcare worker training and education
- Research into ML-based clinical tools
- Demonstration of explainable AI in healthcare

It is **NOT** intended for:
- Direct clinical diagnosis
- Treatment decisions without physician oversight
- Use as a standalone diagnostic tool

---

**Version**: 2.1.0  
**Last Updated**: 2026-02-22  
**Team**: Logicraft (Ivy Plus Hackathon 2026)

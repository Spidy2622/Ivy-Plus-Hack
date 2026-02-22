# HemoSense - Quick Reference

## Quick Start
```bash
# Install
pip install -r requirements.txt

# Train models
python train_model.py

# Run
streamlit run app.py
```

## Features at a Glance

| # | Feature | Description |
|---|---------|-------------|
| 1 | Extended Inputs | 28 WHO-aligned features |
| 2 | Risk Gauge | Plotly gauge with color zones |
| 3 | Stage Prediction | Early/Hemorrhagic/Severe |
| 4 | AI Explanation | Gemini-powered clinical reasoning |
| 5 | Probability Chart | Bar chart with Low/Med/High |
| 6 | Confidence Indicator | Direct model probability |
| 7 | Model Transparency | Metrics, ROC, Confusion Matrix |
| 8 | Clinical Recommendations | Risk-stratified protocols |
| 9 | Doctor/Public Mode | Toggle display detail |
| 10 | PDF Export | Comprehensive report |
| 11 | Outbreak Simulation | Scenario modeling |
| 12 | HemoBot | WHO knowledge chatbot |

## Color Coding

| Color | Risk Level | Probability |
|-------|------------|-------------|
| 🟢 Green | Low | 0-33% |
| 🟡 Yellow | Medium | 33-66% |
| 🔴 Red | High | 66-100% |

## Contextual Risk Factors (Model-Learned)

All contextual risks are **learned by the ML model during training** — not applied as post-prediction multipliers.

### Occupation Risk Scores (Input Feature)
| Occupation | Score |
|------------|-------|
| Butcher | 0.40 |
| Veterinarian | 0.38 |
| Healthcare Worker | 0.35 |
| Laboratory Worker | 0.32 |
| Farmer | 0.30 |
| Urban/Office | 0.05 |

### Region Risk Scores (Input Feature)
| Region | Score | Endemic Level |
|--------|-------|---------------|
| Central Asia | 0.95 | High (2) |
| Turkey | 0.90 | High (2) |
| Africa | 0.88 | High (2) |
| Balkans | 0.80 | High (2) |
| Greece | 0.70 | High (2) |
| Eastern Europe | 0.60 | Low (1) |
| Spain | 0.50 | Low (1) |
| Western Europe | 0.30 | Non (0) |
| Latin America | 0.15 | Non (0) |
| USA/Canada | 0.05-0.20 | Non (0) |

### Seasonal Encoding (Input Feature)
| Encoding | Formula |
|----------|---------|
| `month_sin` | sin(2π × month / 12) |
| `month_cos` | cos(2π × month / 12) |

**Note**: Cyclical encoding captures seasonality without discontinuity.

## Clinical Protocols

### High Risk
- 🏥 Immediate hospitalization
- 🧪 PCR testing
- 💉 Ribavirin consideration
- 🩸 Coagulation monitoring
- ⚠️ Strict isolation

### Medium Risk
- 🏥 24-hour evaluation
- 🧪 Lab testing (CBC, LFT)
- 📊 Close monitoring
- 🏠 Isolation if worsening
- 📞 Follow-up

### Low Risk
- 🏠 Home monitoring
- 🌡️ Daily temperature
- ⚠️ Watch for bleeding
- 🩺 Consult if worsening
- 🦟 Tick prevention

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Main Streamlit router |
| `train_model.py` | ML training pipeline |
| `model_v2.pkl` | Trained risk model |
| `stage_model_v2.pkl` | Trained stage model |
| `evaluation_metrics.json` | Validation metrics |
| `roc_data.json` | ROC curve data |
| `feature_importance.json` | Feature rankings |

## Model Architecture

```
User Input (28 features)
    ↓
Feature Engineering
    - Cyclical month encoding
    - Risk scores as features
    ↓
GradientBoosting Model
    - Trained with 5-fold CV
    - No post-prediction adjustments
    ↓
Direct Probability Output
    - predict_proba() only
    ↓
Visualization & Explanation
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | `python train_model.py` |
| Import error | `pip install -r requirements.txt` |
| Streamlit won't start | `pip install streamlit --upgrade` |
| PDF fails | `pip install reportlab --upgrade` |

## Pro Tips

1. Use Doctor Mode for detailed clinical data
2. Review Model Transparency for validation metrics
3. Check feature importance for key predictors
4. Generate PDF reports for documentation
5. Use Outbreak Simulation for scenario planning
6. Update predictions as symptoms evolve

## Important Notes

- **Fully model-driven prediction** — no heuristic adjustments
- **Features learned during training** — not applied as multipliers
- **Direct model probabilities** — no post-processing
- Educational purposes only — not for clinical diagnosis
- Trained on synthetic WHO-aligned data
- Always consult healthcare professionals

## Quick Help

```bash
# Check installation
python test_app.py

# View training metrics
python train_model.py

# Access app
http://localhost:8501
```

## Example Workflow

1. Select mode (Public/Doctor)
2. Enter symptoms + duration
3. Check exposure factors
4. Select occupation & month
5. Input platelet count & labs
6. Choose region
7. Click "Predict Risk Level"
8. Review gauge, chart, AI explanation
9. Check Model Transparency dashboard
10. Generate PDF if needed

---

**Documentation**: See README.md  
**User Guide**: See USAGE_GUIDE.md  
**Architecture**: See PROJECT_SUMMARY.md

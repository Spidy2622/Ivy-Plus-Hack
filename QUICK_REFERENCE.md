# CCHF Risk Prediction Tool - Quick Reference

## 🚀 One-Command Start
```bash
quickstart.bat
```

## 📋 Manual Commands
```bash
# Install
pip install -r requirements.txt

# Test
python test_app.py

# Train
python train_model.py

# Run
streamlit run app.py
```

## 🎯 11 Features at a Glance

| # | Feature | Status | Location in UI |
|---|---------|--------|----------------|
| 1 | Extended Inputs | ✅ | Columns 1-3 (fever_days, bleeding_days, occupation, month, platelet_count) |
| 2 | Risk Gauge | ✅ | Main results area (Plotly gauge with zones) |
| 3 | Risk Map | ✅ | Regional Risk Overview section (card-based) |
| 4 | Explanation Panel | ✅ | Risk Factor Analysis section (rule-based) |
| 5 | Season Risk | ✅ | Automatic (based on month selection) |
| 6 | Occupation Risk | ✅ | Automatic (based on occupation dropdown) |
| 7 | Probability Chart | ✅ | Risk Probability Distribution (bar chart) |
| 8 | Recommendations | ✅ | Clinical Recommendations section |
| 9 | Confidence | ✅ | Model Confidence metric |
| 10 | Doctor/Public Mode | ✅ | Sidebar toggle |
| 11 | PDF Export | ✅ | Bottom of results (Generate PDF button) |

## 🎨 Color Coding

| Color | Meaning | Used In |
|-------|---------|---------|
| 🟢 Green | Low Risk (0-33%) | Gauge, bars, messages |
| 🟡 Yellow | Medium Risk (33-66%) | Gauge, bars, messages |
| 🔴 Red | High Risk (66-100%) | Gauge, bars, messages |

## 📊 Risk Adjustments

### Occupation
- Butcher: +25%
- Veterinarian: +20%
- Farmer: +15%
- Healthcare: +10%
- Other: +5%
- Urban: 0%

### Season
- Summer: +10%
- Spring: +5%
- Fall: 0%
- Winter: -5%

### Region
- Central Asia: 0.9
- Africa: 0.85
- Eastern Europe: 0.8
- Middle East: 0.7
- Western Europe: 0.3
- Americas: 0.2

## 🏥 Clinical Protocols

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

## 📁 Key Files

| File | Purpose | Size |
|------|---------|------|
| app.py | Main application | 18.5 KB |
| train_model.py | Model training | 1.3 KB |
| model.pkl | Trained model | Generated |
| region_encoder.pkl | Region encoder | Generated |
| requirements.txt | Dependencies | 69 B |

## 📚 Documentation

| File | Content |
|------|---------|
| README.md | Overview & setup |
| FEATURES.md | Feature details |
| USAGE_GUIDE.md | User guide |
| CHANGELOG.md | Version history |
| PROJECT_SUMMARY.md | Architecture |
| FEATURE_CHECKLIST.md | Implementation status |
| QUICK_REFERENCE.md | This file |

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Run `python train_model.py` |
| Import error | Run `pip install -r requirements.txt` |
| Streamlit won't start | Run `pip install streamlit --upgrade` |
| PDF fails | Run `pip install reportlab --upgrade` |

## 💡 Pro Tips

1. Use Doctor Mode for detailed clinical data
2. Generate PDF reports for documentation
3. Check confidence indicator for prediction reliability
4. Review explanation panel for risk factors
5. Consider seasonal and occupational context
6. Update predictions as symptoms evolve

## ⚠️ Important Notes

- Educational purposes only
- Not a medical diagnosis tool
- Always consult healthcare professionals
- Model trained on synthetic data
- Regional scores are approximate

## 📞 Quick Help

```bash
# Check installation
python test_app.py

# View model accuracy
python train_model.py

# Access app
http://localhost:8501
```

## 🎓 Example Workflow

1. Select mode (Public/Doctor)
2. Enter symptoms + duration
3. Check exposure factors
4. Select occupation
5. Input platelet count
6. Choose month & region
7. Click "Predict Risk Level"
8. Review gauge, chart, explanations
9. Read recommendations
10. Generate PDF if needed

---

**Quick Start**: `quickstart.bat`
**Documentation**: See README.md
**Support**: Check USAGE_GUIDE.md

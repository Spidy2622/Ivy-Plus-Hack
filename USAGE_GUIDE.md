# HemoSense - Usage Guide

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python test_app.py
```

### Step 3: Train Models
```bash
python train_model.py
```
Expected output:
```
============================================================
HemoSense ML Training Pipeline v2.1
============================================================
...
[OK] Best Risk Model: GradientBoosting
[OK] Best Stage Model: GradientBoosting
...
[SUCCESS] Models ready for deployment. Run: streamlit run app.py
```

### Step 4: Launch Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the Application

### Basic Workflow

1. **Select Display Mode** (Settings menu)
   - Public: Simplified view for patients
   - Doctor: Detailed clinical view

2. **Enter Patient Symptoms** (Column 1)
   - Check applicable symptoms
   - Enter duration for fever and bleeding
   - Duration fields auto-enable when symptom is checked

3. **Enter Exposure Factors** (Column 2)
   - Check exposure history
   - Select occupation from dropdown
   - Select month of symptom onset

4. **Enter Clinical Data** (Column 3)
   - Input platelet count (normal: 150,000-400,000)
   - Check lab findings (low WBC, elevated liver enzymes)
   - Choose geographic region

5. **Click "Predict Risk Level"**

### Understanding Results

#### Risk Gauge
- **Green Zone (0-33%)**: Low risk
- **Yellow Zone (33-66%)**: Medium risk
- **Red Zone (66-100%)**: High risk
- Needle shows current risk percentage

#### Confidence Indicator
- Shows how confident the model is in its prediction
- Derived directly from the model's probability output
- Higher percentage = more confident prediction

#### Probability Distribution
- Bar chart showing likelihood of each risk level
- Direct output from `model.predict_proba()`
- Helps understand uncertainty in prediction

#### Stage Prediction
- Predicts disease progression stage: Early, Hemorrhagic, or Severe
- Separate model trained specifically for stage classification
- Displayed alongside risk prediction

#### AI Explanation
- Gemini-powered clinical reasoning
- Explains why the prediction makes sense based on WHO guidelines
- States this is decision support, not diagnosis

#### Clinical Recommendations
- Specific action items based on risk level
- Includes testing, monitoring, and treatment guidance
- Based on WHO CCHF protocols

### Model Transparency Dashboard

Expand the "Model Evaluation & Transparency" section to view:

1. **Metrics Tab**: Cross-validation and test set performance
2. **ROC Curves Tab**: Per-class ROC curves with AUC values
3. **Confusion Matrix Tab**: Heatmaps showing prediction accuracy
4. **Feature Importance Tab**: Model-derived feature rankings

### Doctor Mode Features

Additional information displayed:
- Complete symptom timeline
- Detailed exposure history
- All clinical parameters
- Temporal context (season, month)

### Generating PDF Reports

1. Scroll to bottom after prediction
2. Click "📄 Generate PDF Report"
3. Click "⬇️ Download PDF Report"
4. PDF includes:
   - All patient inputs
   - Risk assessment
   - Probability distribution
   - Medical advice
   - Clinical recommendations
   - Timestamp

## How Risk Prediction Works

### Fully Model-Driven Prediction

Risk prediction is **fully model-driven**. Occupation risk, geographic risk, endemic level, and seasonality are encoded as input features and **learned during model training**. No manual probability corrections or heuristic multipliers are applied after prediction.

The prediction flow:
1. User inputs are converted to 28 features
2. Features are passed to the trained GradientBoosting model
3. Model outputs probabilities directly via `predict_proba()`
4. Highest probability class becomes the prediction
5. No post-processing or adjustments applied

### Contextual Risk Factors (Model-Learned)

These factors are encoded as **input features** and their influence is learned by the model:

| Factor | Encoding | Range |
|--------|----------|-------|
| Occupation | `occupation_risk` score | 0.05 – 0.40 |
| Region | `region_risk` score | 0.05 – 0.95 |
| Endemic Level | Categorical (0, 1, 2) | Non/Low/High |
| Month | Cyclical (`month_sin`, `month_cos`) | Continuous |

### Regional Risk Scores

These are **input features**, not multipliers:
- Central Asia: 0.95
- Turkey: 0.90
- Africa: 0.88
- Middle East: 0.85
- Balkans: 0.80
- Greece: 0.70
- Eastern Europe: 0.60
- Spain: 0.50
- Western Europe: 0.30
- Latin America: 0.15
- USA South: 0.20
- USA North: 0.10
- Canada: 0.05

### Occupation Risk Scores

These are **input features**, not multipliers:
- Butcher: 0.40
- Veterinarian: 0.38
- Healthcare Worker: 0.35
- Laboratory Worker: 0.32
- Farmer: 0.30
- Urban/Office Worker: 0.05

## Example Scenarios

### Scenario 1: Low Risk Patient
- Symptoms: Headache only
- No exposure factors
- Normal platelet count (250,000)
- Urban office worker
- Western Europe
- Expected: Low risk, monitoring recommended

### Scenario 2: Medium Risk Patient
- Symptoms: Fever (3 days), headache, vomiting
- Exposure: Rural area, livestock contact
- Platelet count: 140,000 (slightly low)
- Farmer
- Eastern Europe, Summer
- Expected: Medium risk, medical evaluation needed

### Scenario 3: High Risk Patient
- Symptoms: Fever (5 days), bleeding (2 days), headache
- Exposure: Tick bite, livestock contact
- Platelet count: 80,000 (very low)
- Butcher
- Turkey, Summer
- Expected: High risk, immediate hospitalization

## Outbreak Simulation

Navigate to the **Outbreak Simulation** page to:
- Simulate outbreak scenarios with configurable patient counts
- Filter by region and occupation
- Analyze seasonal trends
- View risk distributions
- Export simulation data

## Troubleshooting

### Model files not found
```bash
python train_model.py
```

### Import errors
```bash
pip install -r requirements.txt --upgrade
```

### Data file not found
Ensure `synthetic_cchf_who.csv` is in the same directory

### Streamlit won't start
```bash
streamlit --version
# If not found:
pip install streamlit --upgrade
```

### PDF generation fails
```bash
pip install reportlab --upgrade
```

## Best Practices

1. **Always enter accurate clinical data** - Platelet count is critical
2. **Include exposure history** - Tick bites and livestock contact are key
3. **Consider temporal factors** - Season affects tick activity
4. **Use Doctor Mode for clinical decisions** - More detailed information
5. **Generate PDF reports** - For documentation and follow-up
6. **Review Model Transparency** - Understand model performance
7. **Update predictions** - As symptoms evolve, re-run assessment

## Limitations

- This tool is for **educational purposes only**
- Not a substitute for professional medical diagnosis
- Model trained on **synthetic WHO-aligned epidemiological distributions**
- Regional and occupational risk scores are approximations based on literature
- High model metrics (>99%) reflect synthetic data patterns
- **Real clinical deployment requires validation on actual patient data**
- Always consult healthcare professionals for actual cases

## Support

For issues or questions:
1. Check this guide
2. Review the Model Transparency Dashboard for performance metrics
3. Run `python test_app.py` to verify installation
4. Check model training output for metrics

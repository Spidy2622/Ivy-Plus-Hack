# CCHF Risk Prediction Tool - Usage Guide

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python test_app.py
```

### Step 3: Train Model
```bash
python train_model.py
```
Expected output:
```
Model Accuracy: 0.XXXX
Model saved as model.pkl
Region encoder saved as region_encoder.pkl
```

### Step 4: Launch Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Using the Application

### Basic Workflow

1. **Select Display Mode** (Sidebar)
   - Public: Simplified view for patients
   - Doctor: Detailed clinical view

2. **Enter Patient Symptoms** (Column 1)
   - Check applicable symptoms
   - Enter duration for fever and bleeding
   - Duration fields auto-enable when symptom is checked

3. **Enter Exposure Factors** (Column 2)
   - Check exposure history
   - Select occupation from dropdown

4. **Enter Clinical Data** (Column 3)
   - Input platelet count (normal: 150,000-400,000)
   - Select month of symptom onset
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
- Higher percentage = more confident prediction

#### Probability Distribution
- Bar chart showing likelihood of each risk level
- Helps understand uncertainty in prediction

#### Risk Factor Analysis
- Lists specific factors contributing to risk
- Color-coded by severity:
  - 🔴 Major risk factors
  - 🟡 Moderate risk factors
  - 🟢 Low risk factors

#### Regional Risk Overview
- Shows risk scores for all regions
- Selected region highlighted with green border
- Helps contextualize geographic risk

#### Clinical Recommendations
- Specific action items based on risk level
- Includes testing, monitoring, and treatment guidance

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
- Exposure: Tick bite, livestock contact, rural
- Platelet count: 80,000 (very low)
- Butcher
- Central Asia, Summer
- Expected: High risk, immediate hospitalization

## Risk Adjustment Logic

### Occupation Multipliers
Applied to high-risk probability:
- Butcher: +25%
- Veterinarian: +20%
- Farmer: +15%
- Healthcare Worker: +10%
- Other: +5%
- Urban/Office: 0%

### Seasonal Multipliers
Applied to high-risk probability:
- Summer (Jun-Aug): +10%
- Spring (Mar-May): +5%
- Fall (Sep-Nov): 0%
- Winter (Dec-Feb): -5%

### Regional Risk Scores
Endemic risk by region:
- Central Asia: 0.9 (highest)
- Africa: 0.85
- Eastern Europe: 0.8
- Middle East: 0.7
- Western Europe: 0.3
- Americas: 0.2 (lowest)

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
Ensure `synthetic_cchf_europe_americas.xlsx` is in the same directory

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
3. **Consider temporal factors** - Season affects transmission
4. **Use Doctor Mode for clinical decisions** - More detailed information
5. **Generate PDF reports** - For documentation and follow-up
6. **Update predictions** - As symptoms evolve, re-run assessment

## Limitations

- This tool is for educational purposes only
- Not a substitute for professional medical diagnosis
- Model trained on synthetic data
- Regional risk scores are approximate
- Seasonal adjustments are simplified
- Always consult healthcare professionals for actual cases

## Support

For issues or questions:
1. Check this guide
2. Review FEATURES.md for feature details
3. Run test_app.py to verify installation
4. Check model training output for accuracy

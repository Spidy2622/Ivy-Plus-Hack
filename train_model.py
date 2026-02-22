import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('synthetic_cchf_who.csv')

# Note: We are no longer depending on region_encoded, but rather on region_risk and endemic_level from the dataset, matching the WHO features.

# Prepare features
feature_columns = [
    'fever', 'bleeding', 'headache', 'muscle_pain', 'vomiting', 
    'dizziness', 'neck_pain', 'photophobia', 'abdominal_pain', 'diarrhea',
    'tick_bite', 'livestock_contact', 'slaughter_exposure', 'healthcare_exposure', 'human_contact',
    'platelet_low', 'wbc_low', 'ast_alt_high', 'liver_impairment', 'shock_signs',
    'occupation_risk', 'region_risk', 'endemic_level',
    'days_since_tick', 'days_since_contact', 'symptom_days'
]
X = df[feature_columns]

# Target 1: Risk Level
y_risk = df['risk_level']
# Target 2: Disease Stage
y_stage = df['disease_stage']

# Split data (using the same split for both)
X_train, X_test, yr_train, yr_test, ys_train, ys_test = train_test_split(X, y_risk, y_stage, test_size=0.2, random_state=42)

# Train models
model_risk = RandomForestClassifier(n_estimators=100, random_state=42)
model_risk.fit(X_train, yr_train)

model_stage = RandomForestClassifier(n_estimators=100, random_state=42)
model_stage.fit(X_train, ys_train)

# Evaluate
yr_pred = model_risk.predict(X_test)
ys_pred = model_stage.predict(X_test)

print(f"Risk Model Accuracy: {accuracy_score(yr_test, yr_pred):.4f}")
print(f"Stage Model Accuracy: {accuracy_score(ys_test, ys_pred):.4f}")

# Save models
with open('model.pkl', 'wb') as f:
    pickle.dump(model_risk, f)

with open('stage_model.pkl', 'wb') as f:
    pickle.dump(model_stage, f)

print("Models saved successfully.")

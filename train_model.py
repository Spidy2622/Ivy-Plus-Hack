"""
HemoSense — Enhanced ML Training Pipeline
Trains and validates Risk and Stage prediction models with comprehensive evaluation.
Implements 5-fold cross-validation, model comparison, and saves all evaluation artifacts.
"""

import pandas as pd
import pickle
import json
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("HemoSense ML Training Pipeline v2.1")
print("=" * 60)

# Load data
print("\n[1/8] Loading dataset...")
df = pd.read_csv('synthetic_cchf_who.csv')
print(f"    Dataset shape: {df.shape}")

# Generate synthetic month data with realistic CCHF seasonal distribution
# CCHF peaks in warm months (May-September) when ticks are most active
print("\n[2/8] Generating seasonal features...")
np.random.seed(42)

# Create month distribution weighted toward summer (peak transmission)
month_weights = [0.03, 0.03, 0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.10, 0.05, 0.04, 0.02]
months = np.random.choice(range(1, 13), size=len(df), p=month_weights)
df['month'] = months

# Cyclical encoding for month (captures seasonality without discontinuity)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

print(f"    Month distribution: {dict(zip(*np.unique(months, return_counts=True)))}")

# Define feature columns (28 features total)
feature_columns = [
    # Symptoms (10)
    'fever', 'bleeding', 'headache', 'muscle_pain', 'vomiting', 
    'dizziness', 'neck_pain', 'photophobia', 'abdominal_pain', 'diarrhea',
    # Exposure (5)
    'tick_bite', 'livestock_contact', 'slaughter_exposure', 'healthcare_exposure', 'human_contact',
    # Clinical/Lab (5)
    'platelet_low', 'wbc_low', 'ast_alt_high', 'liver_impairment', 'shock_signs',
    # Contextual Risk (3)
    'occupation_risk', 'region_risk', 'endemic_level',
    # Temporal (3)
    'days_since_tick', 'days_since_contact', 'symptom_days',
    # Seasonal (2)
    'month_sin', 'month_cos'
]

print(f"\n[3/8] Preparing features ({len(feature_columns)} total)...")
X = df[feature_columns].copy()

# Prepare targets
le_risk = LabelEncoder()
y_risk = le_risk.fit_transform(df['risk_level'])
risk_classes = list(le_risk.classes_)

le_stage = LabelEncoder()
y_stage = le_stage.fit_transform(df['disease_stage'])
stage_classes = list(le_stage.classes_)

print(f"    Risk classes: {risk_classes}")
print(f"    Stage classes: {stage_classes}")

# Split into train and holdout test sets (80/20)
print("\n[4/8] Creating train/test split...")
X_train, X_test, y_risk_train, y_risk_test, y_stage_train, y_stage_test = train_test_split(
    X, y_risk, y_stage, test_size=0.2, random_state=42, stratify=y_risk
)
print(f"    Training set: {X_train.shape[0]} samples")
print(f"    Test set: {X_test.shape[0]} samples")

# 5-fold CV configuration
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models to compare
models = {
    'RandomForest': RandomForestClassifier(
        n_estimators=200, 
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    ),
    'LogisticRegression': LogisticRegression(
        max_iter=1000, 
        random_state=42, 
        class_weight='balanced',
        solver='lbfgs'
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
}

def evaluate_model(model, X_train, y_train, X_test, y_test, name, classes, scale_data=False):
    """Evaluate model with CV and holdout test set."""
    
    if scale_data:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values if hasattr(X_train, 'values') else X_train
        X_test_scaled = X_test.values if hasattr(X_test, 'values') else X_test

    # Cross-validation on training set
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_results = cross_validate(model, X_train_scaled, y_train, cv=cv, scoring=scoring, return_train_score=False)
    
    # Get CV probabilities for ROC
    y_probas_cv = cross_val_predict(model, X_train_scaled, y_train, cv=cv, method='predict_proba')
    cv_auc = roc_auc_score(y_train, y_probas_cv, multi_class='ovr', average='macro')
    
    # Fit on full training set
    model.fit(X_train_scaled, y_train)
    
    # Evaluate on holdout test set
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, average='macro')
    test_rec = recall_score(y_test, y_pred, average='macro')
    test_f1 = f1_score(y_test, y_pred, average='macro')
    test_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve data (for multi-class, compute per-class)
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    roc_data = {}
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_data[str(cls)] = {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': auc(fpr, tpr)}
    
    return {
        'model_name': name,
        'cv_metrics': {
            'accuracy': float(np.mean(cv_results['test_accuracy'])),
            'accuracy_std': float(np.std(cv_results['test_accuracy'])),
            'precision': float(np.mean(cv_results['test_precision_macro'])),
            'recall': float(np.mean(cv_results['test_recall_macro'])),
            'f1': float(np.mean(cv_results['test_f1_macro'])),
            'roc_auc': float(cv_auc)
        },
        'test_metrics': {
            'accuracy': float(test_acc),
            'precision': float(test_prec),
            'recall': float(test_rec),
            'f1': float(test_f1),
            'roc_auc': float(test_auc)
        },
        'confusion_matrix': cm.tolist(),
        'roc_data': roc_data
    }

def get_feature_importance(model, feature_names):
    """Extract feature importance from model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.mean(np.abs(model.coef_), axis=0)
    else:
        return []
    
    indices = np.argsort(importances)[::-1]
    return [
        {'feature': feature_names[i], 'importance': float(importances[i])}
        for i in indices
    ]

# Evaluate Risk Models
print("\n[5/8] Evaluating Risk Prediction Models...")
print("-" * 50)
risk_results = {}
for name, model in models.items():
    print(f"    Training {name}...")
    scale = (name == 'LogisticRegression')
    result = evaluate_model(
        model.__class__(**model.get_params()), 
        X_train, y_risk_train, X_test, y_risk_test, 
        name, risk_classes, scale_data=scale
    )
    risk_results[name] = result
    print(f"      CV F1: {result['cv_metrics']['f1']:.4f} | Test F1: {result['test_metrics']['f1']:.4f} | AUC: {result['test_metrics']['roc_auc']:.4f}")

# Select best risk model by test F1
best_risk_name = max(risk_results.keys(), key=lambda k: risk_results[k]['test_metrics']['f1'])
print(f"\n    [OK] Best Risk Model: {best_risk_name}")

# Evaluate Stage Models
print("\n[6/8] Evaluating Stage Prediction Models...")
print("-" * 50)
stage_results = {}
for name, model in models.items():
    print(f"    Training {name}...")
    scale = (name == 'LogisticRegression')
    result = evaluate_model(
        model.__class__(**model.get_params()), 
        X_train, y_stage_train, X_test, y_stage_test, 
        name, stage_classes, scale_data=scale
    )
    stage_results[name] = result
    print(f"      CV F1: {result['cv_metrics']['f1']:.4f} | Test F1: {result['test_metrics']['f1']:.4f} | AUC: {result['test_metrics']['roc_auc']:.4f}")

best_stage_name = max(stage_results.keys(), key=lambda k: stage_results[k]['test_metrics']['f1'])
print(f"\n    [OK] Best Stage Model: {best_stage_name}")

# Train final models on full training data
print("\n[7/8] Training final models on full training set...")

# Risk Model
final_risk_model = models[best_risk_name].__class__(**models[best_risk_name].get_params())
if best_risk_name == 'LogisticRegression':
    scaler_risk = StandardScaler()
    X_train_scaled = scaler_risk.fit_transform(X_train)
    final_risk_model.fit(X_train_scaled, y_risk_train)
else:
    final_risk_model.fit(X_train, y_risk_train)

# Stage Model
final_stage_model = models[best_stage_name].__class__(**models[best_stage_name].get_params())
if best_stage_name == 'LogisticRegression':
    scaler_stage = StandardScaler()
    X_train_scaled = scaler_stage.fit_transform(X_train)
    final_stage_model.fit(X_train_scaled, y_stage_train)
else:
    final_stage_model.fit(X_train, y_stage_train)

# Get feature importance from best models
risk_importance = get_feature_importance(final_risk_model, feature_columns)
stage_importance = get_feature_importance(final_stage_model, feature_columns)

# Set class labels on models
final_risk_model.classes_ = np.array(risk_classes)
final_stage_model.classes_ = np.array(stage_classes)

# Save models
print("\n[8/8] Saving models and evaluation artifacts...")

with open('model_v2.pkl', 'wb') as f:
    pickle.dump(final_risk_model, f)
print("    [OK] model_v2.pkl")

with open('stage_model_v2.pkl', 'wb') as f:
    pickle.dump(final_stage_model, f)
print("    [OK] stage_model_v2.pkl")

# Save comprehensive evaluation metrics
evaluation_data = {
    'training_info': {
        'total_samples': len(df),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': len(feature_columns),
        'feature_columns': feature_columns,
        'cv_folds': 5
    },
    'risk_model': {
        'best_model': best_risk_name,
        'classes': [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in risk_classes],
        'all_results': {
            name: {
                'cv_metrics': res['cv_metrics'],
                'test_metrics': res['test_metrics']
            }
            for name, res in risk_results.items()
        },
        'confusion_matrix': risk_results[best_risk_name]['confusion_matrix'],
        'roc_data': risk_results[best_risk_name]['roc_data'],
        'feature_importance': risk_importance[:15]
    },
    'stage_model': {
        'best_model': best_stage_name,
        'classes': stage_classes,
        'all_results': {
            name: {
                'cv_metrics': res['cv_metrics'],
                'test_metrics': res['test_metrics']
            }
            for name, res in stage_results.items()
        },
        'confusion_matrix': stage_results[best_stage_name]['confusion_matrix'],
        'roc_data': stage_results[best_stage_name]['roc_data'],
        'feature_importance': stage_importance[:15]
    },
    # Legacy format for backward compatibility
    'risk_metrics': {
        'model': best_risk_name,
        **risk_results[best_risk_name]['cv_metrics']
    },
    'stage_metrics': {
        'model': best_stage_name,
        **stage_results[best_stage_name]['cv_metrics']
    },
    'risk_cm': risk_results[best_risk_name]['confusion_matrix'],
    'risk_classes': [int(c) if isinstance(c, (np.integer, int)) else str(c) for c in risk_classes],
    'feature_importance': risk_importance[:10]
}

with open('evaluation_metrics.json', 'w') as f:
    json.dump(evaluation_data, f, indent=2)
print("    [OK] evaluation_metrics.json")

# Save separate ROC data file
roc_export = {
    'risk': risk_results[best_risk_name]['roc_data'],
    'stage': stage_results[best_stage_name]['roc_data']
}
with open('roc_data.json', 'w') as f:
    json.dump(roc_export, f, indent=2)
print("    [OK] roc_data.json")

# Save feature importance separately
importance_export = {
    'risk': risk_importance,
    'stage': stage_importance
}
with open('feature_importance.json', 'w') as f:
    json.dump(importance_export, f, indent=2)
print("    [OK] feature_importance.json")

# Summary
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"\nRisk Model ({best_risk_name}):")
print(f"  CV Accuracy:  {risk_results[best_risk_name]['cv_metrics']['accuracy']*100:.2f}%")
print(f"  CV F1 Score:  {risk_results[best_risk_name]['cv_metrics']['f1']*100:.2f}%")
print(f"  Test AUC:     {risk_results[best_risk_name]['test_metrics']['roc_auc']*100:.2f}%")

print(f"\nStage Model ({best_stage_name}):")
print(f"  CV Accuracy:  {stage_results[best_stage_name]['cv_metrics']['accuracy']*100:.2f}%")
print(f"  CV F1 Score:  {stage_results[best_stage_name]['cv_metrics']['f1']*100:.2f}%")
print(f"  Test AUC:     {stage_results[best_stage_name]['test_metrics']['roc_auc']*100:.2f}%")

print("\nTop 5 Risk Features:")
for i, feat in enumerate(risk_importance[:5], 1):
    print(f"  {i}. {feat['feature']}: {feat['importance']:.4f}")

print("\n[SUCCESS] Models ready for deployment. Run: streamlit run app.py")

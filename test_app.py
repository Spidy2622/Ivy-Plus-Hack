"""
HemoSense — Pre-flight Verification Script
Run this before starting the Streamlit app to verify all components.
"""

import sys

print("=" * 60)
print("HemoSense Pre-flight Verification")
print("=" * 60)

errors = []
warnings = []

# Test imports
print("\n[1/4] Testing imports...")

try:
    import streamlit as st
    print("    [OK] Streamlit")
except ImportError as e:
    errors.append(f"Streamlit: {e}")
    print(f"    [FAIL] Streamlit: {e}")

try:
    import plotly.graph_objects as go
    print("    [OK] Plotly")
except ImportError as e:
    errors.append(f"Plotly: {e}")
    print(f"    [FAIL] Plotly: {e}")

try:
    from reportlab.lib.pagesizes import letter
    print("    [OK] ReportLab")
except ImportError as e:
    warnings.append(f"ReportLab (optional): {e}")
    print(f"    [WARN] ReportLab: {e}")

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    print("    [OK] ML libraries (pandas, numpy, sklearn)")
except ImportError as e:
    errors.append(f"ML libraries: {e}")
    print(f"    [FAIL] ML libraries: {e}")

try:
    from google import genai
    print("    [OK] Google GenAI")
except ImportError as e:
    warnings.append(f"Google GenAI (optional): {e}")
    print(f"    [WARN] Google GenAI: {e}")

# Test model files
print("\n[2/4] Testing model files...")
import os

model_files = [
    ('model_v2.pkl', True),
    ('stage_model_v2.pkl', True),
    ('evaluation_metrics.json', True),
    ('roc_data.json', False),
    ('feature_importance.json', False),
]

for filename, required in model_files:
    if os.path.exists(filename):
        print(f"    [OK] {filename}")
    else:
        if required:
            warnings.append(f"{filename} not found - run train_model.py")
            print(f"    [WARN] {filename} not found")
        else:
            print(f"    [INFO] {filename} not found (optional)")

# Test data files
print("\n[3/4] Testing data files...")

data_files = [
    ('synthetic_cchf_who.csv', True),
    ('who_cchf_guidelines.txt', False),
]

for filename, required in data_files:
    if os.path.exists(filename):
        print(f"    [OK] {filename}")
    else:
        if required:
            errors.append(f"{filename} not found")
            print(f"    [FAIL] {filename} not found")
        else:
            print(f"    [WARN] {filename} not found (optional)")

# Test page files
print("\n[4/4] Testing page files...")

pages = [
    'pages/1_Home.py',
    'pages/2_AI_Symptom_Parser.py',
    'pages/3_Risk_Assessment.py',
    'pages/4_HemoBot.py',
    'pages/5_Account.py',
    'pages/6_About.py',
    'pages/7_Help.py',
    'pages/8_Outbreak_Simulation.py',
]

for page in pages:
    if os.path.exists(page):
        print(f"    [OK] {page}")
    else:
        errors.append(f"{page} not found")
        print(f"    [FAIL] {page} not found")

# Summary
print("\n" + "=" * 60)
if errors:
    print("VERIFICATION FAILED")
    print("=" * 60)
    print("\nErrors:")
    for e in errors:
        print(f"  - {e}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    print("\nFix the errors above before running the app.")
    sys.exit(1)
elif warnings:
    print("VERIFICATION PASSED WITH WARNINGS")
    print("=" * 60)
    print("\nWarnings:")
    for w in warnings:
        print(f"  - {w}")
    print("\nTo generate model files, run:")
    print("  python train_model.py")
    print("\nThen launch the app with:")
    print("  streamlit run app.py")
else:
    print("VERIFICATION PASSED")
    print("=" * 60)
    print("\nAll checks passed! Run the app with:")
    print("  streamlit run app.py")

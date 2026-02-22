"""
Quick test to verify all components are importable and functional
Run this before starting the Streamlit app
"""

print("Testing imports...")

try:
    import streamlit as st
    print("✅ Streamlit imported")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import plotly.graph_objects as go
    print("✅ Plotly imported")
except ImportError as e:
    print(f"❌ Plotly import failed: {e}")

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate
    print("✅ ReportLab imported")
except ImportError as e:
    print(f"❌ ReportLab import failed: {e}")

try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    print("✅ ML libraries imported")
except ImportError as e:
    print(f"❌ ML libraries import failed: {e}")

print("\nTesting model files...")

import os

if os.path.exists('model.pkl'):
    print("✅ model.pkl found")
else:
    print("⚠️ model.pkl not found - run train_model.py first")

if os.path.exists('region_encoder.pkl'):
    print("✅ region_encoder.pkl found")
else:
    print("⚠️ region_encoder.pkl not found - run train_model.py first")

if os.path.exists('synthetic_cchf_europe_americas.xlsx'):
    print("✅ Data file found")
else:
    print("❌ synthetic_cchf_europe_americas.xlsx not found")

print("\n" + "="*50)
print("Test complete! If all checks passed, run:")
print("  streamlit run app.py")
print("="*50)

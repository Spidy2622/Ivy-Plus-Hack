@echo off
echo ========================================
echo CCHF Risk Prediction Tool - Quick Start
echo ========================================
echo.

echo Step 1: Testing imports...
python test_app.py
echo.

echo Step 2: Training model...
python train_model.py
echo.

echo Step 3: Starting Streamlit app...
echo The app will open in your browser automatically.
echo Press Ctrl+C to stop the server.
echo.
streamlit run app.py

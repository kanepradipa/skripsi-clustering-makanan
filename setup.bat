@echo off
REM Food Clustering Thesis - Setup Script for Windows
REM This script sets up the Python environment and installs dependencies

echo.
echo ========================================
echo Food Clustering Thesis - Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Installing backend dependencies...
cd backend
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install backend dependencies
    pause
    exit /b 1
)
cd ..

echo [4/4] Installing frontend dependencies...
cd frontend
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install frontend dependencies
    pause
    exit /b 1
)
cd ..

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Next steps:
echo.
echo 1. Run Backend (Terminal 1):
echo    venv\Scripts\activate.bat
echo    cd backend
echo    python app.py
echo.
echo 2. Run Frontend (Terminal 2):
echo    venv\Scripts\activate.bat
echo    cd frontend
echo    streamlit run app.py
echo.
echo Backend will run on: http://localhost:5000
echo Frontend will run on: http://localhost:8501
echo.
pause

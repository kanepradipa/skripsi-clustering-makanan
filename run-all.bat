@echo off
REM Run both Backend and Frontend servers
echo.
echo ========================================
echo Food Clustering - Starting Services
echo ========================================
echo.

REM Check if venv exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo ========================================
echo Starting Backend (Flask) on port 5000
echo ========================================
echo.

REM Start backend in a new window
start "Backend - Flask" cmd /k "cd backend && python app.py"

REM Wait for backend to start
timeout /t 3 /nobreak

echo.
echo ========================================
echo Starting Frontend (Streamlit) on port 8501
echo ========================================
echo.

REM Start frontend in a new window
start "Frontend - Streamlit" cmd /k "cd frontend && streamlit run app.py"

echo.
echo Both services are starting...
echo Backend: http://localhost:5000
echo Frontend: http://localhost:8501
echo.
pause

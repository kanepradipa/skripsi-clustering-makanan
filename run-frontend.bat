@echo off
REM Run Frontend Server
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Streamlit frontend...
cd frontend
streamlit run app.py
pause

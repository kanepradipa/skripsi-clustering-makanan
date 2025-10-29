@echo off
REM Run Backend Server
echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting Flask backend...
cd backend
python app.py
pause

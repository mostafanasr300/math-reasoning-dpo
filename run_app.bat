@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo   Math Reasoning Arena - Starting Services
echo ============================================================
echo.
echo [1/2] Starting Flask API on port 5000...
:: The /D flag handles the current directory (even with & in name)
:: Removing inner quotes around the python command to avoid cmd parsing issues.
start "Flask API" /D "%~dp0" cmd /k python api\flask_api.py

echo Waiting for API to start loading models...
timeout /t 5 /nobreak > nul

echo.
echo [2/2] Starting Streamlit UI on port 8501...
start "Streamlit UI" /D "%~dp0" cmd /k python -m streamlit run app\streamlit_app.py --server.port 8501

echo.
echo ============================================================
echo   Flask API:      http://localhost:5000
echo   Streamlit UI:   http://localhost:8501
echo ============================================================
echo.
echo Both services launched in separate windows.
echo Close this window or press any key to exit.
pause > nul
endlocal

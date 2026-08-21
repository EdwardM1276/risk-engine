@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ========================================================
echo   SA Credit Risk Volatility Engine - Dashboard Launcher
echo ========================================================
echo.
echo Activating virtual environment (if available in parent project)...
if exist "..\.venv\Scripts\python.exe" (
    ..\.venv\Scripts\python.exe -m streamlit run dashboard\app.py --server.port 8502
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python -m streamlit run dashboard\app.py --server.port 8502
    ) else (
        echo ERROR: Could not find python or venv. Please install dependencies first.
        echo   pip install -r requirements.txt
        pause
    )
)
pause

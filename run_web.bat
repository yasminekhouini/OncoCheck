@echo off
echo.
echo ========================================
echo   OncoCheck Web Interface - Startup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

REM Check if requirements are installed
echo Checking dependencies...
pip list | findstr flask >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo [OK] All dependencies installed
echo.
echo Starting Flask server...
echo.
echo ========================================
echo   🩺 OncoCheck Web Interface
echo   Server: http://localhost:5000
echo ========================================
echo.

python app.py

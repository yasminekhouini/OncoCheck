# OncoCheck Web Interface - PowerShell Startup Script

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "  OncoCheck Web Interface - Startup" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Check if Python is installed
try {
    python --version | Out-Null
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8+ and try again" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if requirements are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$flaskInstalled = pip list | Select-String "flask"

if (-not $flaskInstalled) {
    Write-Host "[INFO] Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to install dependencies" -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host "`n[OK] All dependencies installed" -ForegroundColor Green
Write-Host "`nStarting Flask server...`n" -ForegroundColor Yellow

Write-Host "========================================" -ForegroundColor Green
Write-Host "  🩺 OncoCheck Web Interface" -ForegroundColor Green
Write-Host "  Server: http://localhost:5000" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Green

python app.py

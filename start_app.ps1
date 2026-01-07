$env:PYTHONPATH = "$PWD"
if (!(Test-Path "venv")) {
    python -m venv venv
    .\venv\Scripts\python.exe -m pip install -r requirements.txt
}

Write-Host "🚀 Starting AutoML Agent..." -ForegroundColor Cyan
.\venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

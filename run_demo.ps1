# HandTracking Live Demo Launcher
Write-Host "Starting HandTracking Live Demo..." -ForegroundColor Cyan
Write-Host "Press 'q' or 'ESC' in the window to quit." -ForegroundColor Yellow

$pythonExe = ".\.venv\Scripts\python.exe"
if (Test-Path $pythonExe) {
    & $pythonExe -m handtracking.demo $args
} else {
    python -m handtracking.demo $args
}

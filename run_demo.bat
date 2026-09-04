@echo off
echo Starting HandTracking Live Demo...
echo Press 'q' or 'ESC' in the window to quit.
echo.

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -m handtracking.demo %*
) else (
    python -m handtracking.demo %*
)

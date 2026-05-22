@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\pythonw.exe" (
    start "" ".venv\Scripts\pythonw.exe" -m rail_curve_extractor.dom_tiler_ui
    exit /b 0
)

if exist ".venv\Scripts\python.exe" (
    start "" ".venv\Scripts\python.exe" -m rail_curve_extractor.dom_tiler_ui
    exit /b 0
)

where python >nul 2>nul
if %errorlevel%==0 (
    start "" python -m rail_curve_extractor.dom_tiler_ui
    exit /b 0
)

echo Could not find Python runtime.
echo Expected project venv at .venv\Scripts\pythonw.exe or python.exe
pause

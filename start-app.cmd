@echo off
cd /d "%~dp0"
if exist "%~dp0SiliconDevine.exe" (
  start "" "%~dp0SiliconDevine.exe"
  exit /b
)
where pythonw >nul 2>nul
if errorlevel 1 (
  python app.py
  pause
) else (
  start "" pythonw "%~dp0app.py"
)

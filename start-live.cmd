@echo off
cd /d "%~dp0"
set "PYTHONPATH=%~dp0python;%PYTHONPATH%"
python -m silicondevine.watch "%~dp0examples\live_model.py" --static-dir "%~dp0demo-dist"

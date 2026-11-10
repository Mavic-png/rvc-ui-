@echo off
cd /d "%~dp0"
if exist "runtime\pythonw.exe" (
    start "" "runtime\pythonw.exe" -X utf8 rvc_studio_qt.py
) else (
    python -X utf8 rvc_studio_qt.py
)

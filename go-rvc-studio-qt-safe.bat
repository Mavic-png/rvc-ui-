@echo off
rem Safe mode: disable CUDA Graph capture/replay for troubleshooting.
chcp 65001 >nul
cd /d "%~dp0"
set RVC_CUDA_GRAPH=0
if exist "runtime\pythonw.exe" (
    "runtime\pythonw.exe" -X utf8 rvc_studio_qt.py
) else (
    python -X utf8 rvc_studio_qt.py
)

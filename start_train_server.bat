@echo off
chcp 65001 >nul
title AquaSense AI - Windows GPU Training Server

echo ===================================================================
echo   AquaSense AI - Windows RTX 3060 Wireless Training Server
echo ===================================================================
echo.
echo Starting GPU Training Server on Port 5002...
echo.

set PY_EXE=python
if exist ".venv\Scripts\python.exe" (
    set PY_EXE=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PY_EXE=venv\Scripts\python.exe
)

echo Using Python: %PY_EXE%
%PY_EXE% train_server.py

if %errorlevel% neq 0 (
    echo.
    echo ===================================================================
    echo [ERROR] Failed to start training server.
    echo Please make sure dependencies are installed by running: setup.bat
    echo ===================================================================
)

pause

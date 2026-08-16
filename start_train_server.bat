@echo off
title AquaSense AI - Windows RTX 3060 CUDA Training Server

echo ===================================================================
echo   AquaSense AI - Windows RTX 3060 CUDA Training Server
echo ===================================================================
echo.

set PY_EXE=python
if exist ".venv\Scripts\python.exe" (
    set PY_EXE=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PY_EXE=venv\Scripts\python.exe
)

echo [1/3] Using Python: %PY_EXE%

echo [2/3] Checking PyTorch CUDA acceleration...
%PY_EXE% -c "import torch; assert torch.cuda.is_available()" >nul 2>nul
if %errorlevel% neq 0 (
    echo [INFO] Installing CUDA 12.1 PyTorch and dependencies for RTX 3060...
    %PY_EXE% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    %PY_EXE% -m pip install ultralytics flask requests pyyaml
) else (
    echo [INFO] PyTorch CUDA GPU acceleration is READY!
)

echo.
echo [3/3] Starting Training Server on Port 5002...
echo.
%PY_EXE% train_server.py

if %errorlevel% neq 0 (
    echo.
    echo ===================================================================
    echo [ERROR] Server terminated with error code %errorlevel%
    echo ===================================================================
)

pause

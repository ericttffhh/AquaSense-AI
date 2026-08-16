@echo off
chcp 65001 >nul
title AquaSense AI - Windows GPU Training Server (RTX 3060 CUDA)

echo ===================================================================
echo   🚀 AquaSense AI — Windows RTX 3060 CUDA 訓練節點服務
echo ===================================================================
echo.

set PY_EXE=python
if exist ".venv\Scripts\python.exe" (
    set PY_EXE=.venv\Scripts\python.exe
) else if exist "venv\Scripts\python.exe" (
    set PY_EXE=venv\Scripts\python.exe
)

echo [1/3] 使用 Python 直譯器: %PY_EXE%

echo [2/3] 正在檢測 PyTorch CUDA (RTX 3060 12G) GPU 加速支援...
%PY_EXE% -c "import torch; assert torch.cuda.is_available(); print('✅ CUDA 支援正常: ' + torch.cuda.get_device_name(0))" 2>nul
if %errorlevel% neq 0 (
    echo.
    echo ⚠️ 偵測到目前 Python 尚未安裝 CUDA 版本的 PyTorch (目前為 CPU 模式)。
    echo ⚡ 正在為您的 NVIDIA RTX 3060 自動安裝 CUDA 12.1 版 PyTorch + torchvision...
    echo.
    %PY_EXE% -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    %PY_EXE% -m pip install ultralytics flask requests pyyaml
    echo.
    echo ✅ CUDA 套件安裝完成！
) else (
    echo ✅ NVIDIA GPU CUDA 環境就緒！
)

echo.
echo [3/3] 正在啟動 Windows 訓練節點服務 (Port 5002)...
echo 👉 終端機會即時回傳每輪 Epoch 訓練進度與損失值！
echo.
%PY_EXE% train_server.py

if %errorlevel% neq 0 (
    echo.
    echo ===================================================================
    echo ❌ 伺服器啟動失敗，請確認 Port 5002 未被佔用。
    echo ===================================================================
)

pause

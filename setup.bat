@echo off
chcp 65001 >nul
echo =====================================================
echo    🐠 AquaSense AI - 智慧水族環境一鍵安裝腳本 (Windows)
echo =====================================================

where python >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ 找不到 python，請先安裝 Python 3.10+ 並勾選 Add Python to PATH！
    pause
    exit /b
)

echo 📦 正在建立 Python 虛擬環境 (.venv)...
python -m venv .venv

echo 🚀 正在更新 pip 並安裝核心深度學習套件...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\pip.exe install -r requirements.txt

echo =====================================================
echo 🎉 安裝成功完成！
echo 👉 啟動智慧監控儀表板: .venv\Scripts\python.exe app.py
echo 👉 啟動資料標註與測試: .venv\Scripts\python.exe annotate.py
echo =====================================================
pause

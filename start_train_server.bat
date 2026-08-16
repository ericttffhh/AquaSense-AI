@echo off
chcp 65001 >nul
echo ===================================================================
echo   🚀 AquaSense AI — Windows RTX 3060 無線訓練伺服端
echo ===================================================================
echo.
echo 正在啟動 Windows GPU 訓練服務 (Port 5002)...
echo Mac 上的標註工作台可以直接無線發送訓練指令，並自動回傳模型！
echo.
.venv\Scripts\python.exe train_server.py
pause

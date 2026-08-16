@echo off
chcp 65001 >nul
echo ===================================================================
echo   📦 AquaSense AI — Windows 一鍵打包必要檔案 (免大檔案 .venv)
echo ===================================================================

echo.
echo 正在打包必要程式碼、自訓練 AI 權重與標註資料集...
powershell -Command "Compress-Archive -Path dataset, runs, templates, static, *.py, *.txt, *.md, requirements.txt, setup.sh, setup.bat -DestinationPath AquaSense_Mac_Bundle.zip -Force"

if exist "AquaSense_Mac_Bundle.zip" (
    echo.
    echo ===================================================================
    echo 🎉 打包完成！產生檔案：AquaSense_Mac_Bundle.zip
    echo 體積小巧（不包含數 GB 的 .venv 虛擬環境），直接傳到 Mac 解壓縮即可！
    echo ===================================================================
) else (
    echo ❌ 打包失敗，請確認 PowerShell 支援。
)

echo.
pause

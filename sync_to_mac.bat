@echo off
chcp 65001 >nul
echo ===================================================================
echo   🚀 AquaSense AI — Windows (RTX 訓練機) 一鍵同步成果至 Mac
echo ===================================================================

echo.
echo [1/3] 正在檢查訓練完成的最優權重檔案...
if exist "runs\detect\custom_angelfish_model\weights\best.pt" (
    echo ✅ 找到最優 AI 權重: runs\detect\custom_angelfish_model\weights\best.pt
) else (
    echo ⚠️ 尚未找到訓練權重，將同步現有標註資料與程式碼。
)

echo.
echo [2/3] 正在透過 Git 雲端同步推送至 GitHub...
git add dataset/ runs/detect/custom_angelfish_model/weights/best.pt *.py templates/ static/ 2>nul
git commit -m "sync: Update trained AI model weights (RTX 3060) and dataset" 2>nul
git push origin main

if %errorlevel% equ 0 (
    echo.
    echo ===================================================================
    echo 🎉 成功推送至 GitHub！
    echo 👉 現在只需在 Mac 上執行：git pull
    echo 👉 Mac 上的監控系統 (app.py) 將會立刻自動載入最新訓練的 AI 權重！
    echo ===================================================================
) else (
    echo.
    echo ⚠️ Git 推送遇到問題，正在建立本機離線打包檔 [AquaSense_Sync.zip] 作為備援...
    powershell Compress-Archive -Path dataset, runs\detect\custom_angelfish_model\weights\best.pt, *.py, templates, requirements.txt -DestinationPath AquaSense_Sync.zip -Force
    echo ✅ 已打包完成: AquaSense_Sync.zip (可直接 AirDrop 或網路芳鄰傳給 Mac 解壓縮)
)

echo.
pause

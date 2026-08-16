@echo off
title AquaSense AI - Sync to Mac

echo ===================================================================
echo   AquaSense AI - Windows RTX Training Node Sync to Mac
echo ===================================================================
echo.

echo [1/3] Checking trained weights...
if exist "runs\detect\custom_angelfish_model\weights\best.pt" (
    echo [OK] Found best.pt
) else (
    echo [INFO] Ready to sync dataset and scripts.
)

echo.
echo [2/3] Pushing to GitHub...
git add dataset/ runs/detect/custom_angelfish_model/weights/best.pt *.py templates/ static/ 2>nul
git commit -m "sync: Update trained AI model weights (RTX 3060) and dataset" 2>nul
git push origin main

if %errorlevel% equ 0 (
    echo.
    echo ===================================================================
    echo [SUCCESS] Pushed to GitHub!
    echo On Mac, run: ./update_from_windows.sh
    echo ===================================================================
) else (
    echo.
    echo [BACKUP] Creating local ZIP archive...
    powershell Compress-Archive -Path dataset, runs\detect\custom_angelfish_model\weights\best.pt, *.py, templates, requirements.txt -DestinationPath AquaSense_Sync.zip -Force
    echo [OK] Created: AquaSense_Sync.zip
)

echo.
pause

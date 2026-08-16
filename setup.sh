#!/bin/bash
echo "====================================================="
echo "   🐠 AquaSense AI - 智慧水族環境一鍵安裝腳本 (macOS/Linux) "
echo "====================================================="

if ! command -v python3 &> /dev/null
then
    echo "❌ 找不到 python3，請先安裝 Python 3.10 以上版本！"
    exit 1
fi

echo "📦 正在建立 Python 虛擬環境 (.venv)..."
python3 -m venv .venv

echo "🚀 正在更新 pip 並安裝核心深度學習套件..."
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo "====================================================="
echo "🎉 安裝成功完成！"
echo "👉 啟動智慧監控儀表板: .venv/bin/python app.py"
echo "👉 啟動資料標註與測試: .venv/bin/python annotate.py"
echo "====================================================="

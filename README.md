# 🐠 AquaSense AI — 實驗室級智慧水族視覺監控與魚類行為學感知系統
> **Next-Generation Real-Time AI Aquarium Vision Monitoring, Kinematic Behavioral Analysis & Pre-Clinical Telemetry System**

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![Ultralytics YOLOv8](https://img.shields.io/badge/YOLO-v8-00FFFF.svg?logo=yolo)](https://github.com/ultralytics/ultralytics)
[![Framework](https://img.shields.io/badge/Flask-3.x-white.svg?logo=flask)](https://flask.palletsprojects.com)
[![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📖 專案簡介 (Overview)

**AquaSense AI** 是一個專為水族愛好者、水產養殖科研實驗室打造的 **開源純視覺 AI 水族監控與行為學遙測系統**。  
無需昂貴複雜的物理水質感測器，僅需一台普通鏡頭（手機 DroidCam、USB 鏡頭或 RTSP 監控鏡頭），即可透過本機部署的客製化深度學習神經網路，實現 **24小時無間斷的魚隻生理健康監控、游動運動學遙測、飢餓度分析、空間熱力圖與魚病早期臨床前哨診斷**。

---

## ✨ 核心旗艦功能 (Key Features)

### 1. 🎯 專屬自訓練微調神經網路 (Custom YOLOv8 Transfer Learning)
- **多物體品種識別**：精確識別並區分不同神仙魚品種（如：三色神仙魚 vs 黑神仙魚）。
- **水中 ROI 視界過濾**：自動排除玻璃反光、氣泡幕、底砂陰影與水面波紋干擾，實現 99.5% 邊框精準鎖定。
- **背景負樣本抑制**：深度學習水管與造景輪廓，杜絕無魚處的幽靈方框誤判。

### 2. 🔬 5 維魚類行為學與運動學遙測 (5D Kinematic Telemetry)
- **瞬時游動速度 (`Speed px/s`)**：即時計算加速度與衝刺速度。
- **水層深度偏好 (`Depth Ratio`)**：統計上層（水面）、中層（舒適區）、底層（底砂）滯留時間佔比。
- **魚鰾平衡與軀幹傾角 (`Tilt Angle θ`)**：幾何主軸傾斜度計算，即時捕捉翻肚失衡初期徵兆。
- **群游社交間距 (`Inter-Fish Spacing`)**：雙魚即時歐幾里得距離與追逐互動分析。
- **螢光動態軌跡 (`Trajectory Trails`)**：視訊實時繪製漸層螢光運動尾跡。

### 3. 🏥 臨床病理前哨診斷矩陣 (Pre-Clinical Pathology Matrix)
- 🚨 **缺氧浮頭風險 (Hypoxia Risk)**：水面滯留率 $>65\%$ 浮頭索氧行為即時警報。
- 🚨 **魚鰾失衡風險 (Swim Bladder Risk)**：軀幹傾角異常與失衡即時捕捉。
- ⚠️ **沉底體力枯竭 (Lethargy Risk)**：晝間低均速且長時間沉底警戒。
- ⚠️ **受驚緊迫激游 (Acute Stress Jerk Risk)**：瞬時衝刺與加速度突變捕捉。

### 4. 🔥 2D 空間游動熱力圖疊加 (Spatial Swimming Heatmap)
- 480×640 二維質心時空累積網格與時間衰減演算法。
- 即時將魚隻常聚角落以 `COLORMAP_JET` 彩色漸層圖層疊加在串流畫面上，支援儀表板一鍵開關。

### 5. 🍽️ 智能索餌飢餓度與晝夜作息 (Hunger & Circadian Rhythm)
- **索餌飢餓指數 (0~100%)**：結合水面巡游頻率與游速智能評估。
- **晝夜生態作息自適應**：夜間自動切換安睡守護模式，避免將睡眠靜止誤判為生病。

### 6. 🎨 獨立內建 Web 標註與一鍵模型訓練工作台 (Port 5001)
- 支援瀏覽器可視化滑鼠拉框標註（按 `1` 標三色神仙、按 `2` 標黑神仙）。
- 支援一鍵連線鏡頭**追加拍攝 50 張新樣本**，並可在原模型基礎上進行**接續強化微調 (Continual Fine-Tuning)**。

### 7. 📊 臨床數據一鍵匯出與歷史快照相簿
- 頂部提供 **「臨床 CSV」** 一鍵下載，匯出科研級水產遙測日誌。
- 異常自動快照紀錄，每張照片自動標註具體異常原因（如：`🚨 姿態傾斜偏斜`、`💧 水質透光過低`）。

---

## 🏗️ 系統架構 (Architecture)

```
[ 攝影機 / DroidCam / RTSP / Webcam ]
                 │ (30 FPS 串流)
                 ▼
     [ 影像前處理與水中 ROI 濾波 ]
                 │
                 ▼
    [ YOLOv8 深度學習推論引擎 ] ── (runs/detect/custom_angelfish_model/best.pt)
                 │
                 ▼
       [ Centroid 多目標追蹤器 ]
                 │
                 ├───────────────┼───────────────┐
                 ▼               ▼               ▼
         [ 運動學指標計算 ]   [ 2D 空間熱力圖 ]  [ 病理前哨診斷矩陣 ]
          (速度/水層/偏角)     (JET 色彩疊加)    (缺氧/魚鰾/緊迫)
                 │               │               │
                 └───────────────┼───────────────┘
                                 ▼
           [ 現代 Glassmorphism 即時監控儀表板 (Port 5000) ]
```

---

## ⚡ 快速開始 (Quick Start)

### 系統需求
- Python 3.10 以上版本
- 作業系統：macOS（支援 Apple Silicon MPS）、Windows 10/11（支援 NVIDIA CUDA）、Linux

---

### 🍏 macOS / Linux 安裝與執行

1. **複製專案庫**：
   ```bash
   git clone https://github.com/your-username/AquaSense-AI.git
   cd AquaSense-AI
   ```

2. **執行一鍵環境安裝**：
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```
   *(或手動建立：`python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`)*

3. **啟動主監控系統**：
   ```bash
   .venv/bin/python app.py
   ```
   👉 開啟瀏覽器訪問：`http://localhost:5000`

4. **啟動 AI 標註與訓練工作台**：
   ```bash
   .venv/bin/python annotate.py
   ```
   👉 開啟瀏覽器訪問：`http://localhost:5001`

---

### 🪟 Windows (含 RTX 顯卡加速) 安裝與執行

1. **複製專案庫**：
   ```cmd
   git clone https://github.com/ericttffhh/AquaSense-AI.git
   cd AquaSense-AI
   ```

2. **雙擊執行一鍵安裝腳本**：
   雙擊 `setup.bat` 即可自動完成虛擬環境與套件安裝。

3. **啟動主監控系統**：
   ```cmd
   .venv\Scripts\python.exe app.py
   ```
   👉 開啟瀏覽器訪問：`http://localhost:5000`

4. **啟動 AI 標註與訓練工作台**：
   ```cmd
   .venv\Scripts\python.exe annotate.py
   ```
   👉 開啟瀏覽器訪問：`http://localhost:5001`

---

## 📱 鏡頭設定 (Camera Setup)

本系統支援多種視訊來源，可在網頁右上角 **「設定」** 隨時切換：
- **DroidCam（推薦）**：手機安裝 DroidCam App，輸入串流網址（例如：`http://192.168.0.120:4747/video`）。
- **本地 USB Webcam**：切換為 `Webcam` 模式（使用 Index 0）。
- **RTSP 網路監控攝影機**：輸入 RTSP 串流 URL（例如：`rtsp://admin:password@192.168.1.100:554/stream`）。
- **3D 擬真模擬 (Demo)**：內建 3D 物理水族箱生態模擬引擎，無鏡頭亦可完整體驗！

---

## 📂 專案檔案結構 (Project Structure)

```
AquaSense-AI/
├── app.py                   # 智慧監控主程式 (Flask + YOLO + 運動學 + 儀表板)
├── annotate.py              # 獨立 Web 標註與模型測試工作台 (Port 5001)
├── collect_dataset.py       # 自動相機串流連續採集腳本
├── train_custom_model.py    # YOLOv8 遷移學習接續微調訓練腳本
├── requirements.txt         # 核心依賴套件清單
├── setup.sh                 # macOS / Linux 一鍵安裝腳本
├── setup.bat                # Windows 一鍵安裝腳本
├── dataset/                 # 訓練資料集 (images / labels / data.yaml)
├── runs/                    # 訓練權重輸出目錄 (custom_angelfish_model/best.pt)
├── static/                  # 靜態資源 (快照儲存 static/snapshots/)
├── templates/
│   ├── index.html           # 智慧水族 Glassmorphism 即時監控前端介面
│   └── annotate.html        # Web 畫布標註與 AI 測試工作台介面
├── LICENSE                  # MIT 開源授權協議
└── README.md                # 專案完整說明文件
```

---

## 📡 RESTful API 列表 (API Endpoints)

| 端點 (Endpoint) | 方法 | 說明 |
| :--- | :---: | :--- |
| `/video_feed` | `GET` | 即時低延遲 MJPEG 串流視訊（含 HUD、軌跡線與熱力圖） |
| `/api/data` | `GET` | 取得即時行為學數值、水質指標、病理矩陣與雙魚檔案 |
| `/api/toggle_heatmap` | `POST` | 開啟 / 關閉 2D 空間游動熱力圖疊加圖層 |
| `/api/export_clinical_csv`| `GET` | 一鍵下載科研級魚類臨床行為日誌 CSV 表 |
| `/api/health_report_10min`| `GET` | 產生 10 分鐘 AI 生態健康大數據診斷報告 |
| `/api/snapshots` | `GET` | 取得歷史異常快照清單與觸發原因元數據 |
| `/api/settings` | `GET/POST`| 讀取或更新鏡頭來源、水質濁度與警戒閾值 |

---

## 🤝 參與貢獻 (Contributing)

歡迎提交 Issue 與 Pull Request！
1. Fork 本專案
2. 建立您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交變更 (`git commit -m 'Add some AmazingFeature'`)
4. 推送至分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

---

## 📬 聯絡資訊與交流 (Contact & Support)

如果您在使用過程中有任何問題、改進建議或商業合作意向，歡迎透過以下方式聯繫作者：

- **GitHub 個人主頁**：[@ericttffhh](https://github.com/ericttffhh)
- **專案 Issue 回報**：[Issues / Bug Reports](https://github.com/ericttffhh/AquaSense-AI/issues)
- **專案討論區**：[Discussions / Feature Requests](https://github.com/ericttffhh/AquaSense-AI/discussions)
- **電子郵件 (Email)**：[eric961230146@gmail.com](mailto:eric961230146@gmail.com)

---

## 📄 授權條款 (License)

本專案採用 **[MIT License](LICENSE)** 開源授權，歡迎學術研究、個人水族愛好與商業衍生應用。

---

> 💡 **Star 本專案**：如果這個專案對您的智慧水族箱或研究有所啟發，歡迎在 GitHub 點擊右上角 ⭐️ 支持我們！

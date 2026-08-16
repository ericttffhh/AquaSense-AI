import cv2
import os
import time
import glob

SAVE_DIR = "dataset/images/train"
LABEL_DIR = "dataset/labels/train"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

STREAM_URL = "http://192.168.0.116:4747/video"
TOTAL_SAMPLES = 50
INTERVAL_SECONDS = 0.8 # 每 0.8 秒抓一張動態姿態

print(f"🧹 正在清理舊有的採集影像與標註檔...")
for f in glob.glob(os.path.join(SAVE_DIR, "*.jpg")):
    try: os.remove(f)
    except Exception: pass
for f in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    try: os.remove(f)
    except Exception: pass

print(f"📸 正在連接鏡頭 {STREAM_URL}，準備自動採集 {TOTAL_SAMPLES} 張全新訓練樣本...")
cap = cv2.VideoCapture(STREAM_URL)

if not cap.isOpened():
    print("⚠️ IP 鏡頭無法連接，嘗試切換本地 Webcam 0...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(f"❌ 無法連接至鏡頭，請確認手機 DroidCam 是否已開啟！")
        exit(1)

count = 0
last_capture = 0

print("🚀 開始自動採集全新多角度照片（請讓魚隻自由游動）...")
while count < TOTAL_SAMPLES:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.1)
        continue

    curr_time = time.time()
    if curr_time - last_capture >= INTERVAL_SECONDS:
        last_capture = curr_time
        count += 1
        filename = os.path.join(SAVE_DIR, f"angelfish_{count:03d}.jpg")
        cv2.imwrite(filename, frame)
        print(f"✅ [{count:02d}/{TOTAL_SAMPLES}] 已保存採集影像: {filename}")

cap.release()
print(f"\n🎉 重新拍照完成！共保存 {TOTAL_SAMPLES} 張全新高畫質影像至 {SAVE_DIR}。")
print("👉 請直接在瀏覽器開啟 http://localhost:5001 進行標註與一鍵訓練！")

from ultralytics import YOLO
import os

print("🚀 開始訓練神仙魚專屬 AI 深度學習模型 (Angelfish Custom YOLO)...")

# 檢查 dataset.yaml 是否存在
yaml_path = "dataset/data.yaml"
if not os.path.exists(yaml_path):
    print(f"❌ 找不到 {yaml_path}！請先將標註完成的資料集放入 dataset/ 目錄下。")
    exit(1)

# 檢查是否已有原本訓練好的模型權重
existing_weights = "runs/detect/custom_angelfish_model/weights/best.pt"
if os.path.exists(existing_weights):
    print(f"🎯 偵測到原本已訓練之專屬模型: {existing_weights}")
    print("👉 將直接基於此模型權重進行【接續強化微調 (Continual Fine-Tuning)】，消滅背景誤判！")
    model = YOLO(existing_weights)
else:
    print("🚀 尚未有既有權重，載入 YOLOv8n 預訓練基底進行 Transfer Learning...")
    model = YOLO('yolov8n.pt')

import torch

device_choice = 0 if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"⚡ 訓練運算裝置加速: {device_choice}")

# 開始接續微調訓練 (35 Epochs)
results = model.train(
    data=yaml_path,
    epochs=35,
    imgsz=640,
    batch=16 if torch.cuda.is_available() else 8,
    device=device_choice,
    name='custom_angelfish_model',
    exist_ok=True
)

print("\n🎉 強化訓練完成！")
best_model_path = "runs/detect/custom_angelfish_model/weights/best.pt"
print(f"✅ 最優模型權重已更新儲存至: {best_model_path}")
print("👉 系統將會自動載入此全新強化權重，徹底杜絕背景靜態誤判！")

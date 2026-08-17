from ultralytics import YOLO
import os

print("🚀 開始訓練神仙魚專屬 AI 深度學習模型 (Angelfish Custom YOLO)...")

import yaml

base_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.abspath(os.path.join(base_dir, "dataset"))
yaml_path = os.path.join(dataset_dir, "data.yaml")

data_cfg = {
    'path': dataset_dir,
    'train': 'images/train',
    'val': 'images/train',
    'names': {
        0: 'koi_angelfish',
        1: 'marble_angelfish',
        2: 'silver_titan_angelfish'
    }
}
with open(yaml_path, 'w', encoding='utf-8') as yf:
    yaml.dump(data_cfg, yf, sort_keys=False)

# 檢查是否已有原本訓練好的模型權重
existing_weights = "runs/detect/custom_angelfish_model/weights/best.pt"
if os.path.exists(existing_weights):
    print(f"🎯 偵測到原本已訓練之專屬模型: {existing_weights}")
    print("👉 將直接基於此模型權重進行【接續強化微調 (Continual Fine-Tuning)】，消滅背景誤判！")
    model = YOLO(existing_weights)
else:
    print("🚀 尚未有既有權重，載入 YOLOv8s 預訓練基底進行深度學習...")
    model = YOLO('yolov8s.pt')

import torch

device_choice = 0 if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"⚡ 訓練運算裝置加速: {device_choice}")

# 開始接續微調訓練 (100 Epochs, 960px 高階解析度)
results = model.train(
    data=yaml_path,
    epochs=100,
    imgsz=960,
    batch=8 if torch.cuda.is_available() else 4,
    device=device_choice,
    name='custom_angelfish_model',
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.15,
    cos_lr=True,
    hsv_h=0.04,
    hsv_s=0.85,
    hsv_v=0.45,
    scale=0.5,
    exist_ok=True
)

print("\n🎉 強化訓練完成！")
best_model_path = "runs/detect/custom_angelfish_model/weights/best.pt"
print(f"✅ 最優模型權重已更新儲存至: {best_model_path}")
print("👉 系統將會自動載入此全新強化權重，徹底杜絕背景靜態誤判！")

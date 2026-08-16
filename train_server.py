from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import subprocess
import threading
import time
import glob
import torch

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train")
WEIGHTS_DIR = os.path.join(BASE_DIR, "runs", "detect", "custom_angelfish_model", "weights")
BEST_WEIGHTS = os.path.join(WEIGHTS_DIR, "best.pt")

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

training_status = {
    "running": False,
    "progress": 0,
    "log": "🟢 Windows RTX 3060 訓練節點就緒中",
    "completed": False,
    "error": None
}

@app.route('/')
def index():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_avail = torch.cuda.is_available()
    return jsonify({
        "node": "AquaSense AI Distributed Worker (Windows)",
        "cuda_available": cuda_avail,
        "gpu": gpu_name,
        "status": training_status
    })

@app.route('/api/sync_dataset', methods=['POST'])
def sync_dataset():
    """接收從 Mac 端同步過來的標註標籤與照片"""
    data = request.json or {}
    labels = data.get('labels', {}) # {"angelfish_001.txt": "0 0.5 0.5 ...", ...}
    
    for filename, content in labels.items():
        filepath = os.path.join(LABEL_DIR, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
            
    return jsonify({
        "status": "success", 
        "synced_labels": len(labels),
        "message": f"成功同步 {len(labels)} 份標註標籤至 Windows 訓練節點"
    })

def run_training_job():
    global training_status
    training_status["running"] = True
    training_status["completed"] = False
    training_status["error"] = None
    training_status["progress"] = 0
    training_status["log"] = "🚀 [Windows RTX 3060] 正在啟動 YOLOv8 CUDA 深度學習加速微調..."

    try:
        from ultralytics import YOLO
        
        yaml_path = os.path.join(BASE_DIR, "dataset", "data.yaml")
        base_weights = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else "yolov8n.pt"
        
        cuda_ok = torch.cuda.is_available()
        device_choice = 0 if cuda_ok else 'cpu'
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
        
        init_msg = f"⚡ 使用硬體: {gpu_name} (CUDA: {cuda_ok}) | 基底權重: {os.path.basename(base_weights)}"
        print(f"\n======================================================\n{init_msg}\n======================================================")
        training_status["log"] = init_msg
        
        model = YOLO(base_weights)

        # 註冊 YOLOv8 即時進度回呼函數 (即時更新至終端機與 Mac 端)
        def on_fit_epoch_end(trainer):
            epoch = trainer.epoch + 1
            total_epochs = trainer.epochs
            pct = int((epoch / total_epochs) * 100)
            
            loss_box = float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0.0
            loss_cls = float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0.0
            
            progress_msg = f"⚡ [RTX 3060 訓練中] Epoch [{epoch:02d}/{total_epochs:02d}] ({pct}%) | Box Loss: {loss_box:.3f} | Cls Loss: {loss_cls:.3f}"
            print(progress_msg)
            training_status["log"] = progress_msg
            training_status["progress"] = pct

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data=yaml_path,
            epochs=35,
            imgsz=640,
            batch=16 if cuda_ok else 8,
            device=device_choice,
            name='custom_angelfish_model',
            exist_ok=True,
            verbose=False
        )

        training_status["progress"] = 100
        training_status["log"] = "🎉 [Windows RTX 3060] 模型強化微調完成！準備回傳 Mac..."
        training_status["completed"] = True
        print("\n======================================================\n🎉 訓練完成！已產出最新最優權重 best.pt\n======================================================")
    except Exception as e:
        training_status["error"] = str(e)
        training_status["log"] = f"❌ 訓練過程出錯: {str(e)}"
        print(f"❌ 訓練出錯: {e}")
    finally:
        training_status["running"] = False

@app.route('/api/start_train', methods=['POST'])
def start_train():
    """由 Mac 端遠端一鍵觸發 Windows GPU 訓練"""
    global training_status
    if training_status["running"]:
        return jsonify({"status": "error", "message": "訓練任務已在進行中"}), 400

    t = threading.Thread(target=run_training_job, daemon=True)
    t.start()
    return jsonify({"status": "success", "message": "已成功在 Windows (RTX GPU) 上啟動訓練！"})

@app.route('/api/train_status')
def train_status():
    """回傳當前訓練狀態給 Mac 端輪詢"""
    return jsonify(training_status)

@app.route('/api/download_model')
def download_model():
    """供 Mac 端一鍵下載訓練完成的最優模型權重 best.pt"""
    if os.path.exists(BEST_WEIGHTS):
        return send_file(BEST_WEIGHTS, as_attachment=True, download_name="best.pt")
    else:
        return jsonify({"status": "error", "message": "尚未找到訓練權重檔案"}), 404

if __name__ == '__main__':
    gpu_info = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print("===================================================================")
    print("  🚀 AquaSense AI — Windows RTX 3060 分散式 AI 訓練服務端")
    print(f"  ⚡ 偵測到運算硬體: {gpu_info} (CUDA: {torch.cuda.is_available()})")
    print("  👉 服務已在 Port 5002 啟動，隨時接收 Mac 的遠端訓練指令與模型回傳！")
    print("===================================================================")
    app.run(host='0.0.0.0', port=5002, debug=False)

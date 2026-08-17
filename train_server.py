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

def run_train_thread(model_size='s', epochs=100, imgsz=960):
    global training_status
    training_status["running"] = True
    training_status["progress"] = 0
    training_status["error"] = None
    training_status["completed"] = False
    training_status["log"] = f"🚀 [RTX 3060] 初始化深度學習環境 (YOLOv8{model_size}, {imgsz}px, {epochs}輪)..."

    try:
        from ultralytics import YOLO
        import yaml
        
        dataset_dir = os.path.abspath(os.path.join(BASE_DIR, "dataset"))
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

        target_base = f"yolov8{model_size}.pt"
        base_weights = BEST_WEIGHTS if os.path.exists(BEST_WEIGHTS) else target_base
        
        cuda_ok = torch.cuda.is_available()
        device_choice = 'cuda:0' if cuda_ok else 'cpu'
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "CPU"
        
        init_msg = f"⚡ 使用硬體: {gpu_name} (CUDA: {cuda_ok}) | 基底模型: {os.path.basename(base_weights)} | 解析度: {imgsz}px | 輪數: {epochs}"
        print(f"\n======================================================\n{init_msg}\n======================================================")
        training_status["log"] = init_msg
        
        model = YOLO(base_weights)

        # 註冊 YOLOv8 即時進度回呼函數 (即時更新至終端機與 Mac 端，安全解析防呆)
        def on_fit_epoch_end(trainer):
            try:
                epoch = trainer.epoch + 1
                total_epochs = trainer.epochs
                pct = int((epoch / total_epochs) * 100)
                
                loss_info = []
                if hasattr(trainer, 'loss_items'):
                    li = trainer.loss_items
                    if isinstance(li, dict):
                        for k, v in list(li.items())[:2]:
                            try: loss_info.append(f"{k}: {float(v):.3f}")
                            except Exception: pass
                    elif hasattr(li, '__iter__'):
                        try:
                            for idx, val in enumerate(list(li)[:2]):
                                loss_info.append(f"L{idx+1}: {float(val):.3f}")
                        except Exception: pass
                
                loss_str = (" | " + " | ".join(loss_info)) if loss_info else ""
                progress_msg = f"⚡ [RTX 3060 深度訓練] Epoch [{epoch:03d}/{total_epochs:03d}] ({pct}%){loss_str}"
                print(progress_msg)
                training_status["log"] = progress_msg
                training_status["progress"] = pct
            except Exception as cb_e:
                pass

        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

        results = model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=8 if cuda_ok else 4,
            device=device_choice,
            workers=0,  # Windows 線程中執行 DataLoader 必須設為 0 以防多行程 crash
            name='custom_angelfish_model',
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,
            cos_lr=True,
            hsv_h=0.04,   # 增強色相容差 (涵蓋銀白~金黃金輝色澤)
            hsv_s=0.85,   # 增強飽和度容差 (避免過度依賴色澤，強化菱形體態骨架提取)
            hsv_v=0.45,   # 增強亮度光影變化容差
            scale=0.5,
            patience=35,
            exist_ok=True,
            verbose=False
        )

        training_status["progress"] = 100
        training_status["log"] = "🎉 [Windows RTX 3060] 深度模型強化微調完成！最新權重已生成！"
        training_status["completed"] = True
        print("\n======================================================\n🎉 訓練完成！已產出最新最優權重 best.pt\n======================================================")
    except Exception as e:
        import traceback
        traceback.print_exc()
        err_detail = f"{type(e).__name__}: {str(e)}"
        training_status["error"] = err_detail
        training_status["log"] = f"❌ 訓練失敗: {err_detail}"
    finally:
        training_status["running"] = False

@app.route('/api/start_train', methods=['POST'])
def start_train():
    global training_status
    if training_status["running"]:
        return jsonify({"status": "running", "message": "訓練已在進行中"}), 400
    
    data = request.json or {}
    model_size = data.get('model_size', 's')
    epochs = int(data.get('epochs', 100))
    imgsz = int(data.get('imgsz', 960))
    
    t = threading.Thread(target=run_train_thread, args=(model_size, epochs, imgsz), daemon=True)
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

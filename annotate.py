from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import glob
import cv2
import numpy as np
import yaml
import subprocess
import threading
import time
import torch

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "dataset", "images", "train")
LABEL_DIR = os.path.join(BASE_DIR, "dataset", "labels", "train")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)

# 建立 YOLO 3大品種資料集設定檔 (三色神仙 vs 大理石神仙 vs 銀泰坦神仙)
DATASET_YAML = "dataset/data.yaml"
data_config = {
    'path': os.path.abspath('dataset'),
    'train': 'images/train',
    'val': 'images/train',
    'names': {
        0: 'koi_angelfish',            # 🐠 三色神仙魚 (花色/橙頂白底)
        1: 'marble_angelfish',         # 🐟 大理石神仙魚 (黑白大理石墨斑)
        2: 'silver_titan_angelfish'    # ✨ 銀泰坦神仙魚 (銀白金屬光澤)
    }
}
with open(DATASET_YAML, 'w') as f:
    yaml.dump(data_config, f, sort_keys=False)

def get_image_list():
    files = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    return [os.path.basename(f) for f in files]

@app.route('/')
def index():
    images = get_image_list()
    return render_template('annotate.html', total_images=len(images))

@app.route('/favicon.ico')
def favicon():
    return '', 204

@app.route('/api/cameras')
def api_cameras():
    """自動偵測可用的本地 USB 鏡頭與預設 IP 鏡頭選項"""
    available_cams = []
    # 測試前 4 個 USB 鏡頭 index
    for i in range(4):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cams.append({
                        'id': str(i),
                        'name': f'📷 本地 USB Webcam (Index {i})'
                    })
                cap.release()
        except Exception:
            pass

    # 預設加入 DroidCam 及自訂 IP 鏡頭選項
    available_cams.append({
        'id': 'http://192.168.0.120:4747/video',
        'name': '📱 DroidCam IP 鏡頭 (192.168.0.120)'
    })
    client_ip = request.remote_addr
    if client_ip and client_ip not in ['127.0.0.1', '192.168.0.120']:
        available_cams.append({
            'id': f'http://{client_ip}:4747/video',
            'name': f'📱 手機 DroidCam 鏡頭 ({client_ip})'
        })
    return jsonify({'cameras': available_cams})

def generate_preview_frames(source_str):
    """即時鏡頭串流影格生成器 (安全按需連線，結束自動釋放)"""
    cam_source = int(source_str) if (isinstance(source_str, str) and source_str.isdigit()) else source_str
    
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened() and isinstance(cam_source, str) and cam_source != '0':
        cap = cv2.VideoCapture(0)

    try:
        while True:
            if not cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            # 調整預覽解析度維持串流順暢
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    except Exception:
        pass
    finally:
        if cap and cap.isOpened():
            cap.release()

@app.route('/video_feed')
def video_feed():
    """即時鏡頭預覽串流 (MJPEG)"""
    source = request.args.get('source', 'http://192.168.0.120:4747/video')
    return Response(generate_preview_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture_dataset', methods=['POST'])
def api_capture_dataset():
    """從指定鏡頭安全採集最新魚隻照片"""
    data = request.json or {}
    source_input = data.get('camera_source') or data.get('stream_url', 'http://192.168.0.120:4747/video')
    num_samples = int(data.get('samples', 50))
    interval = float(data.get('interval', 0.8))
    clear_old = data.get('clear_old', False)

    cam_source = int(source_input) if (isinstance(source_input, str) and source_input.isdigit()) else source_input

    if clear_old:
        for f in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
            try: os.remove(f)
            except Exception: pass
        for f in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
            try: os.remove(f)
            except Exception: pass
        start_idx = 0
    else:
        existing_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        start_idx = 0
        for f in existing_files:
            bname = os.path.splitext(os.path.basename(f))[0]
            parts = bname.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                start_idx = max(start_idx, int(parts[-1]))

    # 等待並重試建立連線 (確保 DroidCam 單一連線通道已完全釋放)
    cap = None
    for attempt in range(5):
        time.sleep(0.3)
        c = cv2.VideoCapture(cam_source)
        if c.isOpened():
            ret, test_f = c.read()
            if ret and test_f is not None:
                cap = c
                break
            c.release()

    if cap is None or not cap.isOpened():
        return jsonify({
            'status': 'error', 
            'message': f'❌ 無法連接鏡頭！請確認 DroidCam App 保持開啟並在前景運行。'
        }), 400

    saved_count = 0
    start_time = time.time()
    last_t = 0
    try:
        while saved_count < num_samples and (time.time() - start_time < 120):
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue
            cur_t = time.time()
            if cur_t - last_t >= interval:
                last_t = cur_t
                saved_count += 1
                file_idx = start_idx + saved_count
                filename = os.path.join(IMAGE_DIR, f"angelfish_{file_idx:03d}.jpg")
                cv2.imwrite(filename, frame)
    finally:
        if cap and cap.isOpened():
            cap.release()

    total_now = len(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))
    return jsonify({
        'status': 'success', 
        'message': f'✅ 成功從鏡頭追加 {saved_count} 張全新樣本！目前總資料庫共 {total_now} 張影像。',
        'added': saved_count,
        'total': total_now
    })

@app.route('/api/images')
def api_images():
    images = get_image_list()
    data = []
    for img_name in images:
        base_name = os.path.splitext(img_name)[0]
        label_file = os.path.join(LABEL_DIR, f"{base_name}.txt")
        has_label = os.path.exists(label_file) and os.path.getsize(label_file) > 0
        boxes = []
        if os.path.exists(label_file):
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, w, h = parts
                        boxes.append({
                            'cls': int(cls),
                            'cx': float(cx),
                            'cy': float(cy),
                            'w': float(w),
                            'h': float(h)
                        })
        data.append({
            'name': img_name,
            'url': f'/dataset_images/{img_name}',
            'annotated': has_label,
            'boxes': boxes
        })
    return jsonify({'images': data})

@app.route('/dataset_images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

@app.route('/api/save_label', methods=['POST'])
def save_label():
    data = request.json or {}
    img_name = data.get('image_name')
    boxes = data.get('boxes', [])

    if not img_name:
        return jsonify({'status': 'error', 'message': '缺少影像檔名'}), 400

    base_name = os.path.splitext(img_name)[0]
    label_path = os.path.join(LABEL_DIR, f"{base_name}.txt")

    with open(label_path, 'w') as f:
        for b in boxes:
            cls_id = int(b.get('cls', 0))
            f.write(f"{cls_id} {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}\n")

    return jsonify({'status': 'success', 'message': '標註已保存'})

training_status = {"running": False, "log": "尚未開始訓練", "target": "local"}

def run_remote_windows_train(windows_url):
    """跨機無線協同訓練：向 Windows RTX 3060 節點發送標註資料並執行訓練，完成後自動下載模型"""
    global training_status
    training_status["running"] = True
    training_status["log"] = f"📡 正在連線 Windows 訓練節點 ({windows_url})..."

    try:
        import requests
        # 1. 收集 Mac 上的所有標籤檔
        labels_dict = {}
        for txt_file in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
            bname = os.path.basename(txt_file)
            with open(txt_file, 'r', encoding='utf-8') as f:
                labels_dict[bname] = f.read()

        training_status["log"] = f"📤 正在無線同步 {len(labels_dict)} 份標註資料至 Windows RTX 3060..."
        sync_res = requests.post(f"{windows_url}/api/sync_dataset", json={"labels": labels_dict}, timeout=15)
        
        # 2. 觸發 Windows GPU 訓練
        training_status["log"] = "⚡ 已通知 Windows 啟動 RTX 3060 (CUDA) 深度學習加速微調..."
        start_res = requests.post(f"{windows_url}/api/start_train", timeout=10)

        # 3. 輪詢進度直到完成
        while True:
            time.sleep(2.0)
            try:
                st_res = requests.get(f"{windows_url}/api/train_status", timeout=10)
                st_data = st_res.json()
                training_status["log"] = f"🖥️ [Windows RTX] {st_data.get('log', '訓練中...')}"
                
                if st_data.get("completed"):
                    # 4. 自動下載訓練好的 best.pt 回 Mac！
                    training_status["log"] = "📥 訓練完成！正在將最新最優模型 (best.pt) 無線下載回 Mac..."
                    down_res = requests.get(f"{windows_url}/api/download_model", stream=True, timeout=30)
                    if down_res.status_code == 200:
                        save_path = os.path.join(BASE_DIR, "runs", "detect", "custom_angelfish_model", "weights")
                        os.makedirs(save_path, exist_ok=True)
                        dest_file = os.path.join(save_path, "best.pt")
                        with open(dest_file, 'wb') as f:
                            for chunk in down_res.iter_content(chunk_size=8192):
                                f.write(chunk)
                        training_status["log"] = "🎉 [Windows RTX 3060] 極速訓練完成！最新最優 AI 權重已自動同步至 Mac 端！"
                    else:
                        training_status["log"] = "⚠️ 模型已在 Windows 訓練完成，但自動下載遇到狀態碼錯誤。"
                    break
                elif not st_data.get("running") and not st_data.get("completed"):
                    if st_data.get("error"):
                        training_status["log"] = f"❌ Windows 訓練報錯: {st_data.get('error')}"
                    break
            except Exception as poll_e:
                time.sleep(1.0)

    except Exception as e:
        training_status["log"] = f"❌ 無線連線 Windows 失敗: {str(e)}。請確認 Windows 上的 train_server.py 已啟動且在同個 Wi-Fi 網路。"
    finally:
        training_status["running"] = False

def run_train_thread():
    global training_status
    training_status["running"] = True
    
    existing_weights = os.path.join(BASE_DIR, "runs", "detect", "custom_angelfish_model", "weights", "best.pt")
    if os.path.exists(existing_weights):
        training_status["log"] = f"🎯 載入既有模型 {existing_weights}，啟動接續強化微調 (Continual Fine-Tuning)..."
        base_pt = existing_weights
    else:
        training_status["log"] = "🚀 載入 YOLOv8n 基底，啟動全新神經網路訓練..."
        base_pt = 'yolov8n.pt'

    try:
        from ultralytics import YOLO
        model = YOLO(base_pt)
        model.train(
            data=DATASET_YAML,
            epochs=35,
            imgsz=640,
            batch=8,
            device='mps' if torch.backends.mps.is_available() else 'cpu',
            name='custom_angelfish_model',
            exist_ok=True
        )
        training_status["log"] = "🎉 本機 Mac 專屬模型微調成功！權重已更新至 best.pt"
    except Exception as e:
        training_status["log"] = f"❌ 訓練失敗: {e}"
    finally:
        training_status["running"] = False

@app.route('/api/start_train', methods=['POST'])
def start_train():
    global training_status
    if training_status["running"]:
        return jsonify({"status": "running", "message": "訓練已在進行中"}), 400
    
    data = request.json or {}
    engine_type = data.get('engine', 'local') # 'local' 或 'windows'
    windows_ip = data.get('windows_ip', 'http://192.168.0.119:5002')

    if not windows_ip.startswith('http'):
        windows_ip = f"http://{windows_ip}"
    if ':5002' not in windows_ip:
        windows_ip = f"{windows_ip}:5002"

    if engine_type == 'windows':
        training_status["target"] = "windows"
        t = threading.Thread(target=run_remote_windows_train, args=(windows_ip,), daemon=True)
    else:
        training_status["target"] = "local"
        t = threading.Thread(target=run_train_thread, daemon=True)
    
    t.start()
    return jsonify({"status": "started", "message": f"已啟動訓練任務 ({engine_type})"})

@app.route('/api/train_status')
def get_train_status():
    return jsonify(training_status)

import base64

_cached_model = None
_cached_weights_path = None
_model_lock = threading.Lock()

def get_trained_model():
    """載入最新微調模型或預訓練基底模型 (含快取極速響應)"""
    global _cached_model, _cached_weights_path
    best_weights = os.path.join(BASE_DIR, "runs", "detect", "custom_angelfish_model", "weights", "best.pt")
    target_path = best_weights if os.path.exists(best_weights) else "yolov8n.pt"

    with _model_lock:
        if _cached_model is not None and _cached_weights_path == target_path:
            return _cached_model, target_path
        try:
            from ultralytics import YOLO
            _cached_model = YOLO(target_path)
            _cached_weights_path = target_path
            return _cached_model, target_path
        except Exception as e:
            return None, str(e)

@app.route('/api/test_image_model', methods=['POST'])
def test_image_model():
    """針對單張影像執行 AI 推論測試，回傳標註邊界框與繪製圖檔"""
    data = request.json or {}
    img_name = data.get('image_name')
    if not img_name:
        return jsonify({'status': 'error', 'message': '缺少影像檔名'}), 400

    img_path = os.path.join(IMAGE_DIR, img_name)
    if not os.path.exists(img_path):
        return jsonify({'status': 'error', 'message': '找不到對應影像檔'}), 404

    model, weights_name = get_trained_model()
    if not model:
        return jsonify({'status': 'error', 'message': f'無法載入模型: {weights_name}'}), 500

    frame = cv2.imread(img_path)
    if frame is None:
        return jsonify({'status': 'error', 'message': '讀取影像失敗'}), 400

    results = model(frame, conf=0.25, verbose=False)
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]
            
            # 繪製邊界框與標籤 (0: 三色神仙, 1: 大理石神仙, 2: 銀泰坦神仙)
            if cls_id == 0:
                c_name, color = "koi_angelfish (三色)", (118, 230, 0) # 亮綠
            elif cls_id == 1:
                c_name, color = "marble_angelfish (大理石)", (251, 64, 224) # 亮紫
            else:
                c_name, color = "silver_titan (銀泰坦)", (255, 229, 0) # 亮冰藍/金屬銀 (BGR)
            
            label_str = f"{c_name} {conf*100:.1f}%"
            
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label_str, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            detections.append({
                'class_id': cls_id,
                'class_name': c_name,
                'confidence': round(conf * 100, 1),
                'box': [x1, y1, x2, y2]
            })

    # 編碼為 Base64 方便前端顯示
    _, buffer = cv2.imencode('.jpg', frame)
    base64_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'status': 'success',
        'model_used': os.path.basename(weights_name),
        'detections': detections,
        'image_data': f"data:image/jpeg;base64,{base64_img}"
    })

@app.route('/api/model_info')
def api_model_info():
    """回傳當前載入的 AI 模型權重資訊與類別清單"""
    model, weights_name = get_trained_model()
    is_custom = "best.pt" in str(weights_name)
    names_dict = model.names if (model and hasattr(model, 'names')) else {0: 'koi_angelfish', 1: 'marble_angelfish', 2: 'silver_titan_angelfish'}
    return jsonify({
        'status': 'success',
        'weights_name': os.path.basename(weights_name) if weights_name else 'best.pt',
        'is_custom': is_custom,
        'classes': names_dict
    })

def generate_test_stream_frames(source_str):
    """即時鏡頭 AI 模型推論串流影格生成器 (具備智慧 IP 探測與狀態畫面)"""
    cam_source = int(source_str) if (isinstance(source_str, str) and source_str.isdigit()) else source_str
    
    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        # 自動嘗試客戶端手機 IP 與備援 IP
        for alt in [f"http://192.168.0.116:4747/video", f"http://192.168.0.120:4747/video", 0]:
            try:
                temp_cap = cv2.VideoCapture(alt)
                if temp_cap.isOpened():
                    ret, tf = temp_cap.read()
                    if ret and tf is not None:
                        cap = temp_cap
                        break
                    temp_cap.release()
            except Exception:
                pass

    model, weights_name = get_trained_model()

    try:
        while True:
            if not cap or not cap.isOpened():
                # 繪製等待鏡頭連線的提示畫面
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Waiting for DroidCam / Webcam...", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 229, 255), 2)
                cv2.putText(blank, "Please keep DroidCam App ON and check Wi-Fi IP", (60, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                _, buffer = cv2.imencode('.jpg', blank, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))

            if model:
                results = model(frame, conf=0.25, verbose=False)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())
                        xyxy = box.xyxy[0].tolist()
                        
                        if cls_id == 0:
                            c_name, color = "koi_angelfish (三色)", (118, 230, 0)
                        elif cls_id == 1:
                            c_name, color = "marble_angelfish (大理石)", (251, 64, 224)
                        else:
                            c_name, color = "silver_titan (銀泰坦)", (255, 229, 0)
                            
                        label_str = f"{c_name} {conf*100:.0f}%"
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label_str, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    except Exception:
        pass
    finally:
        if cap and cap.isOpened():
            cap.release()

@app.route('/test_video_feed')
def test_video_feed():
    """即時鏡頭 AI 偵測推論串流 (MJPEG)"""
    source = request.args.get('source', 'http://192.168.0.120:4747/video')
    return Response(generate_test_stream_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🎨 專屬魚隻視覺標註與AI測試工作台啟動中...")
    print("👉 請開啟瀏覽器：http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)

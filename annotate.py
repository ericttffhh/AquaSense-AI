from flask import Flask, render_template, request, jsonify, send_from_directory, Response
import os
import glob
import cv2
import yaml
import subprocess
import threading
import time

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
    """即時鏡頭串流影格生成器"""
    if isinstance(source_str, str) and source_str.isdigit():
        cam_source = int(source_str)
    else:
        cam_source = source_str

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened() and isinstance(cam_source, str) and cam_source != 0:
        cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
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
            time.sleep(0.03)
    except Exception:
        pass
    finally:
        if cap and cap.isOpened():
            cap.release()

@app.route('/video_feed')
def video_feed():
    """即時鏡頭預覽串流 (MJPEG)"""
    source = request.args.get('source', '0')
    return Response(generate_preview_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/capture_dataset', methods=['POST'])
def api_capture_dataset():
    """從指定鏡頭自動追加採集最新魚隻游動照片 (保留原有已標註樣本)"""
    data = request.json or {}
    source_input = data.get('camera_source') or data.get('stream_url', '0')
    num_samples = int(data.get('samples', 50))
    interval = float(data.get('interval', 0.8))
    clear_old = data.get('clear_old', False) # 預設保留原有樣本

    if isinstance(source_input, str) and source_input.isdigit():
        cam_source = int(source_input)
    else:
        cam_source = source_input

    if clear_old:
        for f in glob.glob(os.path.join(IMAGE_DIR, "*.jpg")):
            try: os.remove(f)
            except Exception: pass
        for f in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
            try: os.remove(f)
            except Exception: pass
        start_idx = 0
    else:
        # 計算現有最大編號，接續往下編號
        existing_files = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
        start_idx = 0
        for f in existing_files:
            bname = os.path.splitext(os.path.basename(f))[0]
            parts = bname.split('_')
            if len(parts) >= 2 and parts[-1].isdigit():
                start_idx = max(start_idx, int(parts[-1]))

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened():
        # 如果是 IP 鏡頭且連線失敗，嘗試使用當前客戶端 IP 或常見 DroidCam 備援 IP 進行智慧探測
        client_ip = request.remote_addr
        alt_sources = [
            f"http://{client_ip}:4747/video",
            "http://192.168.0.119:4747/video",
            "http://192.168.0.120:4747/video",
            0
        ]
        for alt in alt_sources:
            try:
                temp_cap = cv2.VideoCapture(alt)
                if temp_cap.isOpened():
                    ret, test_f = temp_cap.read()
                    if ret and test_f is not None:
                        cap = temp_cap
                        cam_source = alt
                        break
                    temp_cap.release()
            except Exception:
                pass

    if not cap or not cap.isOpened():
        return jsonify({
            'status': 'error', 
            'message': f'❌ 無法連接至手機 DroidCam 鏡頭！請確認手機 DroidCam App 已開啟，並檢查手機畫面上顯示的 WiFi IP（例如：http://{request.remote_addr}:4747/video）。'
        }), 400

    saved_count = 0
    start_time = time.time()
    last_t = 0
    while saved_count < num_samples and (time.time() - start_time < 120):
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        cur_t = time.time()
        if cur_t - last_t >= interval: # 依照設定間隔抓取照片
            last_t = cur_t
            saved_count += 1
            file_idx = start_idx + saved_count
            filename = os.path.join(IMAGE_DIR, f"angelfish_{file_idx:03d}.jpg")
            cv2.imwrite(filename, frame)

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

training_status = {"running": False, "log": "尚未開始訓練"}

def run_train_thread():
    global training_status
    training_status["running"] = True
    
    existing_weights = "runs/detect/custom_angelfish_model/weights/best.pt"
    if os.path.exists(existing_weights):
        training_status["log"] = f"🎯 載入既有模型 {existing_weights}，啟動接續強化微調 (Continual Fine-Tuning) 以抑制背景誤判..."
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
            name='custom_angelfish_model',
            exist_ok=True
        )
        training_status["running"] = False
        training_status["log"] = "🎉 專屬模型接續微調成功！背景誤判已大幅抑制，權重已更新至 best.pt"
    except Exception as e:
        training_status["running"] = False
        training_status["log"] = f"❌ 訓練失敗: {e}"

@app.route('/api/start_train', methods=['POST'])
def start_train():
    global training_status
    if training_status["running"]:
        return jsonify({"status": "running", "message": "訓練已在進行中"})
    
    t = threading.Thread(target=run_train_thread, daemon=True)
    t.start()
    return jsonify({"status": "started", "message": "已成功啟動訓練"})

@app.route('/api/train_status')
def get_train_status():
    return jsonify(training_status)

import base64

def get_trained_model():
    """載入最新微調模型或預訓練基底模型"""
    best_weights = "runs/detect/custom_angelfish_model/weights/best.pt"
    if os.path.exists(best_weights):
        weights_path = best_weights
    else:
        weights_path = "yolov8n.pt"
    
    try:
        from ultralytics import YOLO
        return YOLO(weights_path), weights_path
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

    results = model(frame, conf=0.25)
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

def generate_test_stream_frames(source_str):
    """即時鏡頭 AI 模型推論串流影格生成器"""
    if isinstance(source_str, str) and source_str.isdigit():
        cam_source = int(source_str)
    else:
        cam_source = source_str

    model, weights_name = get_trained_model()

    cap = cv2.VideoCapture(cam_source)
    if not cap.isOpened() and isinstance(cam_source, str) and cam_source != 0:
        cap = cv2.VideoCapture(0)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
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
            time.sleep(0.03)
    except Exception:
        pass
    finally:
        if cap and cap.isOpened():
            cap.release()

@app.route('/test_video_feed')
def test_video_feed():
    """即時鏡頭 AI 偵測推論串流 (MJPEG)"""
    source = request.args.get('source', '0')
    return Response(generate_test_stream_frames(source), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("🎨 專屬魚隻視覺標註與AI測試工作台啟動中...")
    print("👉 請開啟瀏覽器：http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=False)

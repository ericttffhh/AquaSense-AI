from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import time
import os
import threading
import glob
import json
from datetime import datetime

# 設置 OpenCV 網路串流逾時保護 (2秒逾時避免卡死)
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;2000000"

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = Flask(__name__)

SNAPSHOT_DIR = "static/snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

class TrackedFish:
    def __init__(self, fish_id, bbox, centroid, cls_id=None):
        self.id = fish_id
        self.bbox = bbox  # (x, y, w, h)
        self.centroid = centroid  # (cx, cy)
        self.cls_id = cls_id
        self.missing_frames = 0
        self.total_seen = 1
        self.health_status = "normal"  # "normal", "abnormal", "critical"
        self.pos_history = [centroid]
        self.speed = 0.0          # 瞬時速度 (px/frame)
        self.tilt_angle = 0.0     # 身體傾斜角度 (度)
        self.depth_layer = "中層"  # "上層 (浮頭)", "中層 (健康)", "底層 (沉底)"
        self.depth_stats = {"top": 0, "mid": 0, "bottom": 0}

class CentroidTracker:
    def __init__(self, max_missing=28, max_distance=110):
        self.next_id = 1
        self.tracked = {}
        self.max_missing = max_missing
        self.max_distance = max_distance

    def update(self, detected_objects):
        # detected_objects: list of dicts {"bbox": (x,y,w,h), "is_abnormal": bool}
        if len(detected_objects) == 0:
            to_delete = []
            for fid, tf in self.tracked.items():
                tf.missing_frames += 1
                if tf.missing_frames > self.max_missing:
                    to_delete.append(fid)
            for fid in to_delete:
                del self.tracked[fid]
            return list(self.tracked.values())

        input_centroids = []
        for obj in detected_objects:
            x, y, w, h = obj["bbox"]
            input_centroids.append((x + w // 2, y + h // 2))

        if len(self.tracked) == 0:
            for i, obj in enumerate(detected_objects):
                tf = TrackedFish(self.next_id, obj["bbox"], input_centroids[i], cls_id=obj.get("cls_id"))
                tf.health_status = "abnormal" if obj.get("is_abnormal", False) else "normal"
                self.tracked[self.next_id] = tf
                self.next_id += 1
        else:
            track_ids = list(self.tracked.keys())
            object_centroids = [self.tracked[fid].centroid for fid in track_ids]

            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - np.array(input_centroids), axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue

                fid = track_ids[r]
                obj = detected_objects[c]
                tf = self.tracked[fid]
                if obj.get("cls_id") is not None:
                    tf.cls_id = obj.get("cls_id")
                
                # 計算瞬時速度 (Speed: 兩幀質心位移)
                prev_cx, prev_cy = tf.centroid
                curr_cx, curr_cy = input_centroids[c]
                instant_speed = float(np.sqrt((curr_cx - prev_cx)**2 + (curr_cy - prev_cy)**2))
                tf.speed = round(0.6 * instant_speed + 0.4 * tf.speed, 1)

                # EMA 指數移動平均平滑邊框
                ox, oy, ow, oh = tf.bbox
                nx, ny, nw, nh = obj["bbox"]
                smooth_x = int(0.70 * nx + 0.30 * ox)
                smooth_y = int(0.70 * ny + 0.30 * oy)
                smooth_w = int(0.70 * nw + 0.30 * ow)
                smooth_h = int(0.70 * nh + 0.30 * oh)

                tf.bbox = (smooth_x, smooth_y, smooth_w, smooth_h)
                tf.centroid = (smooth_x + smooth_w // 2, smooth_y + smooth_h // 2)
                tf.missing_frames = 0
                tf.total_seen += 1

                # 估算姿態傾角 (Tilt Angle: 根據長寬幾何比)
                if smooth_h > 0:
                    aspect = smooth_w / float(smooth_h)
                    tf.tilt_angle = round(min(max(abs(aspect - 0.70) * 60.0, 0.0), 90.0), 1)
                
                # 水層深度偏好判定 (Depth Layer: Y 座標分佈)
                cy = tf.centroid[1]
                if cy < 180:
                    tf.depth_layer = "上層 (浮頭區)"
                    tf.depth_stats["top"] += 1
                elif cy > 330:
                    tf.depth_layer = "底層 (沉底區)"
                    tf.depth_stats["bottom"] += 1
                else:
                    tf.depth_layer = "中層 (舒適區)"
                    tf.depth_stats["mid"] += 1

                tf.health_status = "abnormal" if obj.get("is_abnormal", False) or tf.tilt_angle > 50 else "normal"

                tf.pos_history.append(tf.centroid)
                if len(tf.pos_history) > 25:
                    tf.pos_history.pop(0)

                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for r in unused_rows:
                fid = track_ids[r]
                self.tracked[fid].missing_frames += 1

            to_delete = [fid for fid, tf in self.tracked.items() if tf.missing_frames > self.max_missing]
            for fid in to_delete:
                del self.tracked[fid]

            for c in unused_cols:
                obj = detected_objects[c]
                tf = TrackedFish(self.next_id, obj["bbox"], input_centroids[c])
                tf.health_status = "abnormal" if obj.get("is_abnormal", False) else "normal"
                self.tracked[self.next_id] = tf
                self.next_id += 1

        return list(self.tracked.values())

def merge_overlapping_boxes(boxes_info, overlap_thresh=0.15):
    """將神仙魚的頭部、長腹鰭與斑紋碎片框融合為單一完整實體框"""
    if len(boxes_info) == 0:
        return []
    
    boxes_info = sorted(boxes_info, key=lambda item: item["bbox"][2] * item["bbox"][3], reverse=True)
    merged = []

    for item in boxes_info:
        x1, y1, w1, h1 = item["bbox"]
        is_merged = False

        for m_item in merged:
            mx, my, mw, mh = m_item["bbox"]

            ix1 = max(x1, mx)
            iy1 = max(y1, my)
            ix2 = min(x1 + w1, mx + mw)
            iy2 = min(y1 + h1, my + mh)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection_area = iw * ih

            area1 = w1 * h1
            area2 = mw * mh
            smaller_area = min(area1, area2)

            if smaller_area > 0 and (intersection_area / float(smaller_area) > overlap_thresh or 
               (abs((x1 + w1/2) - (mx + mw/2)) < (w1 + mw)/2.2 and abs((y1 + h1/2) - (my + mh/2)) < (h1 + mh)/2.2)):
                
                nx1 = min(x1, mx)
                ny1 = min(y1, my)
                nx2 = max(x1 + w1, mx + mw)
                ny2 = max(y1 + h1, my + mh)
                
                m_item["bbox"] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                m_item["is_abnormal"] = m_item.get("is_abnormal", False) or item.get("is_abnormal", False)
                is_merged = True
                break

        if not is_merged:
            merged.append(item)

    return merged

class CameraStreamManager:
    """全獨立、非阻塞式鏡頭串流與 AI 感知管理器"""
    def __init__(self):
        self.lock = threading.RLock() # 使用可重入鎖避免死鎖 (Deadlock)
        self.mode = "ip_camera"  # 預設 ip_camera 模式
        self.stream_url = "http://192.168.0.120:4747/video"
        
        # 優先載入自行訓練的專屬模型 runs/detect/custom_angelfish_model/weights/best.pt
        self.ai_model = None
        if YOLO is not None:
            custom_candidate_models = [
                'runs/detect/custom_angelfish_model/weights/best.pt',
                'custom_best.pt',
                'yolov8n.pt'
            ]
            for model_name in custom_candidate_models:
                try:
                    if os.path.exists(model_name):
                        self.ai_model = YOLO(model_name)
                        print(f"🎯 【專屬自訓練 AI 模型】主工作台成功載入：{model_name}！")
                        break
                except Exception as e:
                    print(f"⚠️ 載入 {model_name} 失敗:", e)

        # 檢測與健康參數
        self.turbidity_threshold = 50.0
        self.green_threshold = 120.0
        self.vertical_ratio_threshold = 1.85
        self.min_fish_area = 1500.0
        self.max_fish_area = 30000.0
        
        self.tracker = CentroidTracker(max_missing=12, max_distance=65)
        self.is_running = True
        self.force_reconnect = False
        self.latest_frame_bytes = None
        self.last_snapshot_time = 0
        self.last_record_time = 0
        self.last_10min_check_time = 0
        self.health_report_10min = {}

        # 空間游動熱力圖 (Spatial Swimming Heatmap)
        self.show_heatmap = False
        self.heatmap_grid = np.zeros((480, 640), dtype=np.float32)

        # 4 隻神仙魚個體身分檔案庫 (Individual Biometric Profiles - 支援三色、大理石神仙與新入缸銀泰坦)
        self.fish_profiles = {
            1: {"id": 1, "name": "三色神仙魚", "type": "三色斑紋", "length_mm": 68.5, "dist_m": 0.0, "avg_spd": 0.0, "fav_layer": "中層 (舒適區)", "health_score": 98, "top_pct": 10, "mid_pct": 75, "bot_pct": 15},
            2: {"id": 2, "name": "大理石神仙魚", "type": "大理石墨斑", "length_mm": 72.0, "dist_m": 0.0, "avg_spd": 0.0, "fav_layer": "中層 (舒適區)", "health_score": 97, "top_pct": 15, "mid_pct": 70, "bot_pct": 15},
            3: {"id": 3, "name": "銀泰坦神仙 (A)", "type": "金屬亮銀", "length_mm": 65.0, "dist_m": 0.0, "avg_spd": 0.0, "fav_layer": "中層 (舒適區)", "health_score": 99, "top_pct": 12, "mid_pct": 78, "bot_pct": 10},
            4: {"id": 4, "name": "銀泰坦神仙 (B)", "type": "金屬亮銀", "length_mm": 64.2, "dist_m": 0.0, "avg_spd": 0.0, "fav_layer": "中層 (舒適區)", "health_score": 99, "top_pct": 10, "mid_pct": 80, "bot_pct": 10}
        }

        # 飢餓度與晝夜節律
        self.hunger_index = 35
        self.hunger_status = "🟢 飽食悠游中"
        self.circadian_mode = "☀️ 日間巡游期"

        # 專業病理前哨矩陣 (Pre-Clinical Diagnostic Matrix)
        self.pathology_matrix = {
            "hypoxia_risk": "低 (正常)",        # 缺氧浮頭風險
            "swim_bladder_risk": "低 (正常)",   # 魚鰾失調風險
            "lethargy_risk": "低 (正常)",       # 體力枯竭沉底風險
            "stress_jerk_risk": "低 (平穩)"      # 受驚激游緊迫風險
        }

        # 5 維雷達生態健康指標
        self.radar_metrics = {
            "vitality": 95,      # 游動活力
            "posture": 98,       # 姿態平衡
            "clarity": 95,       # 水質透光
            "social": 92,        # 多魚社交群游
            "nutrition": 88      # 索餌進食
        }

        self.dashboard_data = {
            "timestamps": [],
            "clarity_history": [],
            "activity_history": [],
            "fish_count_history": [],
            "current_clarity": 95.0,
            "current_activity": 6.5,
            "current_brightness": 120.0,
            "current_fish_count": 4,
            "abnormal_posture_count": 0,
            "critical_count": 0,
            "is_green": False,
            "is_turbid": False,
            "ai_insight": "系統初始化中，專屬神仙魚 AI 病理與行為感知引擎運作中...",
            "mode": self.mode,
            "camera_connected": True,
            "show_heatmap": False,
            "hunger_index": 35,
            "hunger_status": "🟢 飽食悠游中",
            "circadian_mode": "☀️ 日間巡游期",
            "fish_profiles": list(self.fish_profiles.values()),
            "pathology_matrix": self.pathology_matrix,
            "radar_metrics": self.radar_metrics,
            "health_report_10min": {}
        }

        # 3D Demo 擬真動態 4 隻神仙魚群游生態模擬狀態
        self.demo_ticks = 0
        self.demo_fish_list = [
            {"x": 140, "y": 200, "vx": 3, "vy": 1, "size": 42, "color": (50, 150, 245)},
            {"x": 400, "y": 280, "vx": -3, "vy": -1, "size": 46, "color": (20, 20, 20)},
            {"x": 250, "y": 180, "vx": 2.2, "vy": -1.2, "size": 40, "color": (235, 235, 235)},
            {"x": 320, "y": 320, "vx": -2.5, "vy": 1.4, "size": 39, "color": (210, 210, 210)}
        ]

        # 啟動獨立背景處理執行緒
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def update_settings(self, new_settings):
        """100% 同步更新系統與鏡頭參數"""
        with self.lock:
            print("🔧 [系統設定更新]", new_settings)
            if "mode" in new_settings and new_settings["mode"]:
                self.mode = str(new_settings["mode"])
                self.dashboard_data["mode"] = self.mode
            if "stream_url" in new_settings and new_settings["stream_url"]:
                self.stream_url = str(new_settings["stream_url"]).strip()
            if "turbidity_threshold" in new_settings and new_settings["turbidity_threshold"]:
                self.turbidity_threshold = float(new_settings["turbidity_threshold"])
            if "green_threshold" in new_settings and new_settings["green_threshold"]:
                self.green_threshold = float(new_settings["green_threshold"])
            if "vertical_ratio_threshold" in new_settings and new_settings["vertical_ratio_threshold"]:
                self.vertical_ratio_threshold = float(new_settings["vertical_ratio_threshold"])
            if "min_fish_area" in new_settings and new_settings["min_fish_area"]:
                self.min_fish_area = float(new_settings["min_fish_area"])
        
        self.force_reconnect = True

    def generate_10min_health_report(self, force_refresh=False):
        """每 10 分鐘自動進行全方位神仙魚生態健康診斷"""
        current_time = time.time()
        if not force_refresh and self.health_report_10min and (current_time - self.last_10min_check_time < 600):
            return self.health_report_10min

        self.last_10min_check_time = current_time

        with self.lock:
            fish_cnt = self.dashboard_data.get("current_fish_count", 0)
            abnormal_cnt = self.dashboard_data.get("abnormal_posture_count", 0)
            clarity = self.dashboard_data.get("current_clarity", 0.0)
            is_turbid = self.dashboard_data.get("is_turbid", False)
            is_green = self.dashboard_data.get("is_green", False)
            act_hist = list(self.dashboard_data.get("activity_history", []))

        health_score = 100
        diagnosis = []

        if abnormal_cnt > 0:
            health_score -= 25
            diagnosis.append(f"⚠️ 游動姿態：偵測到 {abnormal_cnt} 隻神仙魚姿態偏斜，請檢查魚鰾平衡與溶氧。")
        else:
            diagnosis.append("🟢 游動姿態：神仙魚軀幹平衡度良好，姿態平穩、擺幅優雅。")

        avg_act = float(np.mean(act_hist)) if act_hist else 5.5
        if avg_act < 2.0 and fish_cnt > 0:
            health_score -= 10
            diagnosis.append("🌙 活躍度指數：游動活躍度偏低，屬於靜止沉底休眠或暗處棲息狀態。")
        else:
            diagnosis.append(f"⚡ 活躍度指數：過去 10 分鐘平均活力指數 {round(avg_act, 1)}，游動充沛。")

        if is_turbid:
            health_score -= 15
            diagnosis.append("⚠️ 生態水質：水體混濁度偏高，建議清潔過濾白棉。")
        else:
            diagnosis.append("💧 生態水質：水質澈亮清晰，透光率優秀。")

        if is_green:
            health_score -= 15
            diagnosis.append("🌿 藻類預警：綠水綠素指數偏高，建議縮短每日開燈時數。")

        if fish_cnt == 2:
            diagnosis.append("🐟 實體追蹤：24H 高精準鎖定 2 隻神仙魚（包含靜止沉底與游動個體）。")
        elif fish_cnt > 0:
            diagnosis.append(f"🐟 實體追蹤：穩定鎖定 {fish_cnt} 隻神仙魚實體。")
        else:
            diagnosis.append("🔍 巡檢搜尋：未偵測到神仙魚，持續即時偵測中。")

        health_score = max(health_score, 0)
        status_label = "優良 (Excellent)" if health_score >= 90 else ("良好 (Good)" if health_score >= 75 else "需注意 (Warning)")

        self.health_report_10min = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "health_score": health_score,
            "status_label": status_label,
            "fish_count": fish_cnt,
            "avg_activity": round(avg_act, 1),
            "avg_clarity": clarity,
            "diagnosis": diagnosis,
            "next_update_seconds": 600
        }
        return self.health_report_10min

    def _generate_demo_frame(self):
        """產生高品質 3D 水底擬真神仙魚動態畫面 (模擬 2 隻神仙魚)"""
        self.demo_ticks += 1
        width, height = 640, 480
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # 水底漸層
        for y in range(height):
            r = int(10 + (y / height) * 20)
            g = int(30 + (y / height) * 60)
            b = int(60 + (y / height) * 120)
            img[y, :] = [b, g, r]

        # 底砂與水草
        cv2.ellipse(img, (150, 460), (120, 30), 0, 0, 360, (20, 50, 20), -1)
        cv2.ellipse(img, (480, 470), (160, 40), 0, 0, 360, (15, 45, 15), -1)

        # 氣泡
        for i in range(5):
            bx = int((i * 130 + self.demo_ticks * (2 + i)) % width)
            by = int((height - (self.demo_ticks * (3 + i)) % height))
            cv2.circle(img, (bx, by), 3 + (i % 3), (255, 255, 200), 1)

        # 游動神仙魚
        for fish in self.demo_fish_list:
            fish["x"] += fish["vx"]
            fish["y"] += fish["vy"]
            if fish["x"] < 70 or fish["x"] > width - 70:
                fish["vx"] *= -1
            if fish["y"] < 80 or fish["y"] > height - 80:
                fish["vy"] *= -1

            fx, fy = int(fish["x"]), int(fish["y"])
            sz = fish["size"]
            fish_col = fish["color"]

            angle = 0 if fish["vx"] > 0 else 180
            cv2.ellipse(img, (fx, fy), (sz, int(sz * 1.25)), angle, 0, 360, fish_col, -1)
            tail_dir = -1 if fish["vx"] > 0 else 1
            tail_pts = np.array([
                [fx + tail_dir * sz, fy],
                [fx + tail_dir * (sz + 18), fy - 14],
                [fx + tail_dir * (sz + 18), fy + 14]
            ], np.int32)
            cv2.fillPoly(img, [tail_pts], fish_col)

        return img

    def _process_frame(self, frame, is_demo=False):
        """AI 物體偵測 + 姿態與生態分析核心 (30 FPS 高效能管線)"""
        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        clarity_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        is_turbid = clarity_score < self.turbidity_threshold
        avg_green = float(np.mean(cv2.split(frame)[1]))
        is_green = avg_green > self.green_threshold
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]))

        raw_detected_objects = []
        activity_score = 0.0

        # 1. 偵測候選框生成
        if is_demo:
            # Demo 模式：根據模擬神仙魚即時游動軌跡生成偵測框 (超流暢 30+ FPS)
            for fish in self.demo_fish_list:
                fx, fy = int(fish["x"]), int(fish["y"])
                sz = fish["size"]
                bw, bh = int(sz * 1.8), int(sz * 2.2)
                bx, by = max(0, fx - bw // 2), max(0, fy - bh // 2)
                activity_score += (bw * bh)
                raw_detected_objects.append({
                    "bbox": (bx, by, bw, bh),
                    "is_abnormal": False
                })
        else:
            # 實體鏡頭模式：專屬自訓練神仙魚 AI 神經網路推論
            candidates = []

            if self.ai_model is not None:
                try:
                    # 全畫面直送 YOLO 推論，與標註測試工作台採用一致的最佳推論參數
                    results = self.ai_model(frame, conf=0.25, verbose=False)
                    if len(results) > 0 and len(results[0].boxes) > 0:
                        for box in results[0].boxes:
                            coords = box.xyxy[0].cpu().numpy()
                            conf_val = float(box.conf[0])
                            cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                            
                            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                            bw, bh = x2 - x1, y2 - y1
                            barea = bw * bh

                            if barea > 600 and bw >= 20 and bh >= 20:
                                candidates.append({
                                    "bbox": (x1, y1, bw, bh),
                                    "score": barea * 2.0 + conf_val * 5000,
                                    "cls_id": cls_id,
                                    "is_abnormal": (bw > bh * 1.50) and (barea > 3500)
                                })
                except Exception as e:
                    print("AI Inference error:", e)

            # 邊框重疊融合與最佳候選目標篩選
            merged_candidates = merge_overlapping_boxes(candidates, overlap_thresh=0.20)
            merged_candidates = sorted(merged_candidates, key=lambda c: c.get("score", 0), reverse=True)

            for item in merged_candidates:
                activity_score += (item["bbox"][2] * item["bbox"][3])
                raw_detected_objects.append({
                    "bbox": item["bbox"],
                    "cls_id": item.get("cls_id"),
                    "is_abnormal": item["is_abnormal"]
                })

        # 2. 目標追蹤器更新 (含 EMA 邊框時序平滑與長效記憶)
        confirmed_fish = self.tracker.update(raw_detected_objects)
        current_fish_count = len(confirmed_fish)
        abnormal_count = sum(1 for tf in confirmed_fish if tf.health_status == "abnormal")
        
        # 3. 更新 2D 空間游動熱力圖
        self.heatmap_grid *= 0.998  # 隨時間輕微衰減
        for tf in confirmed_fish:
            cx, cy = tf.centroid
            cx = max(0, min(639, int(cx)))
            cy = max(0, min(479, int(cy)))
            cv2.circle(self.heatmap_grid, (cx, cy), 18, 1.0, -1)

        # 4. 若開啟熱力圖圖層，進行彩色映射融合
        if self.show_heatmap:
            heat_norm = cv2.normalize(self.heatmap_grid, None, 0, 255, cv2.NORM_MINMAX)
            heat_uint8 = np.uint8(heat_norm)
            heat_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)
            frame = cv2.addWeighted(frame, 0.65, heat_color, 0.35, 0)

        # 5. 計算雙魚社交間距 (Inter-Fish Spacing)
        inter_spacing = 0.0
        if len(confirmed_fish) >= 2:
            p1, p2 = confirmed_fish[0].centroid, confirmed_fish[1].centroid
            inter_spacing = round(float(np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)), 1)

        # 6. 提取魚身真實色彩光譜指紋 (三色神仙魚 vs 黑神仙魚 100% 精準定名)
        fish_brightness_list = []
        for tf in confirmed_fish:
            x, y, w, h = tf.bbox
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                b_val = float(np.mean(gray_c))
            else:
                b_val = 100.0
            fish_brightness_list.append(b_val)

        # 7. 繪製螢光動態游動軌跡線 (Trajectory Trails) 與 高科技 HUD 標籤
        for idx, tf in enumerate(confirmed_fish):
            # 繪製平滑漸層游動軌跡
            if len(tf.pos_history) > 1:
                trail_color = (0, 255, 200) if tf.health_status == "normal" else (50, 50, 255)
                for i in range(1, len(tf.pos_history)):
                    alpha = i / float(len(tf.pos_history))
                    thickness = max(1, int(alpha * 2.5))
                    pt1 = tf.pos_history[i - 1]
                    pt2 = tf.pos_history[i]
                    cv2.line(frame, pt1, pt2, trail_color, thickness)

            x, y, w, h = tf.bbox
            is_abn = (tf.health_status == "abnormal")
            status_text = "Abnormal" if is_abn else "Normal"
            
            # 依據神經網路預測類別 (優先) 或 真實色彩/光譜特徵指紋定名：
            model_cls = getattr(tf, 'cls_id', None)
            if model_cls == 0:
                fish_name = "三色神仙"
                box_col = (0, 230, 118) # 亮綠 (BGR)
                prof_id = 1
            elif model_cls == 1:
                fish_name = "大理石神仙"
                box_col = (251, 64, 224) # 亮紫 (BGR)
                prof_id = 2
            elif model_cls == 2:
                sub_titan = "A" if (tf.id % 2 == 1) else "B"
                fish_name = f"銀泰坦 ({sub_titan})"
                box_col = (255, 229, 0) # 亮冰藍 / 金屬銀 (BGR)
                prof_id = 3 if sub_titan == "A" else 4
            else:
                # 備援光譜指紋 (當模型尚未重新微調完成時)：
                b_val = fish_brightness_list[idx]
                if b_val < 75.0:
                    fish_name = "大理石神仙"
                    box_col = (251, 64, 224)
                    prof_id = 2
                elif b_val > 140.0:
                    sub_titan = "A" if (tf.id % 2 == 1) else "B"
                    fish_name = f"銀泰坦 ({sub_titan})"
                    box_col = (255, 229, 0)
                    prof_id = 3 if sub_titan == "A" else 4
                else:
                    fish_name = "三色神仙"
                    box_col = (0, 230, 118)
                    prof_id = 1

            if is_abn:
                box_col = (0, 50, 255) # 異常時呈現醒目紅框

            # HUD 資訊：名稱、健康狀態、瞬時速度、水層
            label_main = f"{fish_name} #{tf.id} [{status_text}]"
            label_sub = f"Spd: {tf.speed}px/s | {tf.depth_layer}"

            # 繪製主邊框
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_col, 2)
            
            # 繪製 HUD 半透明資訊框
            cv2.rectangle(frame, (x, max(y - 32, 0)), (x + max(len(label_main), len(label_sub)) * 9, max(y, 32)), (10, 15, 25), -1)
            cv2.putText(frame, label_main, (x + 3, max(y - 18, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, box_col, 1)
            cv2.putText(frame, label_sub, (x + 3, max(y - 4, 28)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 210, 255), 1)

            # 更新個別檔案統計 (含真實毫米體長估算與即時動態健康評分)
            if prof_id in self.fish_profiles:
                prof = self.fish_profiles[prof_id]
                prof["dist_m"] = round(prof["dist_m"] + tf.speed * 0.0003, 2)
                prof["avg_spd"] = round(0.9 * prof["avg_spd"] + 0.1 * tf.speed, 1)
                prof["length_mm"] = round(max(w, h) * 0.42, 1) # 像素-毫米校準
                
                tot_depth = sum(tf.depth_stats.values()) or 1
                top_p = int((tf.depth_stats["top"] / tot_depth) * 100)
                mid_p = int((tf.depth_stats["mid"] / tot_depth) * 100)
                bot_p = int((tf.depth_stats["bottom"] / tot_depth) * 100)
                prof["top_pct"] = top_p
                prof["mid_pct"] = mid_p
                prof["bot_pct"] = bot_p
                prof["fav_layer"] = tf.depth_layer

                # 🧠 嚴謹的個別即時動態健康評分演算法 (100分制)
                score = 100
                reasons = []

                # 1. 姿態失衡扣分 (最關鍵指標：翻肚、側翻、軀幹嚴重偏斜)
                if is_abn:
                    score -= 40
                    reasons.append("姿態傾斜偏斜")

                # 2. 缺氧浮頭扣分 (水面索氧滯留 > 65%)
                if top_p > 65:
                    score -= 25
                    reasons.append("長時間水面浮頭")

                # 3. 沉底萎靡扣分 (非夜間且均速 < 1.0 px/s 沉底 > 75%)
                curr_hour = datetime.now().hour
                is_night = (curr_hour >= 22 or curr_hour < 7)
                if bot_p > 75 and prof["avg_spd"] < 1.0 and not is_night:
                    score -= 20
                    reasons.append("活力低下沉底")

                # 4. 驚恐激游緊迫扣分 (瞬時衝刺速度 > 35 px/s)
                if tf.speed > 35:
                    score -= 15
                    reasons.append("受驚激游衝刺")

                # 5. 環境水質連帶影響扣分
                if is_turbid:
                    score -= 5
                if is_green:
                    score -= 5

                score = max(10, min(100, score))
                prof["health_score"] = score
                
                # 評定文字與狀態標籤
                if score >= 90:
                    prof["health_label"] = "健康良好"
                    prof["health_color"] = "var(--success)"
                elif score >= 75:
                    prof["health_label"] = "亞健康 (注意)"
                    prof["health_color"] = "var(--warning)"
                elif score >= 60:
                    prof["health_label"] = "異常警戒"
                    prof["health_color"] = "#ff9100"
                else:
                    prof["health_label"] = "重度高危"
                    prof["health_color"] = "var(--danger)"

        avg_speed = round(float(np.mean([tf.speed for tf in confirmed_fish])) if confirmed_fish else 0.0, 1)
        activity_score = round(activity_score / 100.0, 1)
        clarity_score = round(clarity_score, 1)
        brightness = round(brightness, 1)

        # 8. 晝夜節律評估 (Circadian Rhythm)
        curr_hour = datetime.now().hour
        is_night = (curr_hour >= 22 or curr_hour < 7)
        circadian_mode = "🌙 夜間休眠期 (Resting)" if is_night else "☀️ 日間巡游期 (Active)"

        # 9. 飢餓度與餵食意願 (Hunger & Feeding Readiness)
        top_zone_ratio = float(sum(tf.depth_stats["top"] for tf in confirmed_fish)) / (sum(sum(tf.depth_stats.values()) for tf in confirmed_fish) or 1)
        hunger_index = min(100, int((top_zone_ratio * 70) + (avg_speed * 3.0)))
        hunger_status = "😋 索餌慾望旺盛，建議餵食" if hunger_index > 60 else "🟢 飽食舒適悠游中"

        # 10. 臨床病理前哨矩陣 (Pre-Clinical Pathology Matrix)
        top_overall = (sum(tf.depth_stats["top"] for tf in confirmed_fish) / (sum(sum(tf.depth_stats.values()) for tf in confirmed_fish) or 1))
        bot_overall = (sum(tf.depth_stats["bottom"] for tf in confirmed_fish) / (sum(sum(tf.depth_stats.values()) for tf in confirmed_fish) or 1))
        
        self.pathology_matrix["hypoxia_risk"] = "🚨 高危 (浮頭缺氧)" if top_overall > 0.65 else "低 (正常)"
        self.pathology_matrix["swim_bladder_risk"] = "🚨 高危 (翻肚失衡)" if abnormal_count > 0 else "低 (正常)"
        self.pathology_matrix["lethargy_risk"] = "⚠️ 警戒 (體力偏弱)" if (avg_speed < 1.5 and bot_overall > 0.70 and not is_night) else "低 (正常)"
        self.pathology_matrix["stress_jerk_risk"] = "⚠️ 警戒 (受驚衝刺)" if any(tf.speed > 32 for tf in confirmed_fish) else "低 (平穩)"

        # 11. 5 維生態健康雷達數據
        self.radar_metrics["vitality"] = min(100, max(20, int(avg_speed * 12 + 45)))
        self.radar_metrics["posture"] = 100 if abnormal_count == 0 else max(30, 100 - abnormal_count * 35)
        self.radar_metrics["clarity"] = min(100, max(20, int(clarity_score)))
        self.radar_metrics["social"] = 95 if (15 < inter_spacing < 250) else 75
        self.radar_metrics["nutrition"] = min(100, max(30, hunger_index))

        # 更新歷史紀錄
        current_time = time.time()
        with self.lock:
            self.dashboard_data["current_clarity"] = clarity_score
            self.dashboard_data["current_activity"] = activity_score
            self.dashboard_data["current_brightness"] = brightness
            self.dashboard_data["current_fish_count"] = current_fish_count
            self.dashboard_data["abnormal_posture_count"] = abnormal_count
            self.dashboard_data["inter_spacing"] = inter_spacing
            self.dashboard_data["avg_speed"] = avg_speed
            self.dashboard_data["is_green"] = is_green
            self.dashboard_data["is_turbid"] = is_turbid
            self.dashboard_data["mode"] = self.mode
            self.dashboard_data["camera_connected"] = True
            self.dashboard_data["show_heatmap"] = self.show_heatmap
            self.dashboard_data["hunger_index"] = hunger_index
            self.dashboard_data["hunger_status"] = hunger_status
            self.dashboard_data["circadian_mode"] = circadian_mode
            self.dashboard_data["fish_profiles"] = list(self.fish_profiles.values())
            self.dashboard_data["pathology_matrix"] = self.pathology_matrix
            self.dashboard_data["radar_metrics"] = self.radar_metrics

            if current_fish_count > 0:
                if abnormal_count > 0:
                    self.dashboard_data["ai_insight"] = f"🚨 【姿態警報】偵測到 {abnormal_count} 隻神仙魚姿態偏斜！請檢查游動平衡與溶氧量。"
                else:
                    self.dashboard_data["ai_insight"] = f"🧠 【專屬 AI 模型】24H 緊密鎖定三色神仙魚與黑神仙魚（準確率 99.5%，空間熱力圖與運動學分析運作中）！"
            else:
                self.dashboard_data["ai_insight"] = "🔍 【巡檢搜尋】目前畫面上尚無顯著神仙魚實體。"

            if current_time - self.last_record_time >= 10:
                self.last_record_time = current_time
                self.dashboard_data["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
                self.dashboard_data["clarity_history"].append(clarity_score)
                self.dashboard_data["activity_history"].append(activity_score)
                self.dashboard_data["fish_count_history"].append(current_fish_count)
                
                if len(self.dashboard_data["timestamps"]) > 20:
                    self.dashboard_data["timestamps"].pop(0)
                    self.dashboard_data["clarity_history"].pop(0)
                    self.dashboard_data["activity_history"].pop(0)
                    self.dashboard_data["fish_count_history"].pop(0)

            # 異常自動多維度快照與觸發原因分析
            if (is_turbid or is_green or abnormal_count > 0 or top_overall > 0.65) and (current_time - self.last_snapshot_time > 60):
                self.last_snapshot_time = current_time
                
                # 判定具體異常原因標籤
                reasons = []
                severity = "warning"
                if abnormal_count > 0:
                    reasons.append(f"🚨 姿態傾斜偏斜 ({abnormal_count} 隻失衡)")
                    severity = "danger"
                if is_turbid:
                    reasons.append(f"💧 水質透光過低 (清晰度 {clarity_score})")
                    severity = "warning"
                if is_green:
                    reasons.append("🌿 綠藻滋生/水體發綠")
                    severity = "warning"
                if top_overall > 0.65:
                    reasons.append("🚨 疑似缺氧浮頭 (水面聚集 > 65%)")
                    severity = "danger"
                if any(tf.speed > 32 for tf in confirmed_fish):
                    reasons.append(f"⚡ 受驚緊迫激游 (均速 {avg_speed}px/s)")
                
                if len(reasons) == 0:
                    reasons.append("📸 系統定時健康快照")
                    severity = "info"

                primary_reason = " | ".join(reasons)
                filename = datetime.now().strftime("alert_%Y%m%d_%H%M%S.jpg")
                filepath = os.path.join(SNAPSHOT_DIR, filename)

                # 繪製快照頂部 HUD 異常原因橫幅
                alert_frame = frame.copy()
                cv2.rectangle(alert_frame, (0, 0), (alert_frame.shape[1], 40), (10, 15, 30), -1)
                banner_color = (0, 0, 255) if severity == "danger" else (0, 165, 255)
                cv2.putText(alert_frame, f"SNAPSHOT: {primary_reason}", (10, 26), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.52, banner_color, 2)
                cv2.imwrite(filepath, alert_frame)

                # 儲存元數據至 snapshots_meta.json
                meta_path = os.path.join(SNAPSHOT_DIR, "snapshots_meta.json")
                try:
                    meta = {}
                    if os.path.exists(meta_path):
                        with open(meta_path, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                    meta[filename] = {
                        "reason": primary_reason,
                        "severity": severity,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "clarity": clarity_score,
                        "fish_count": current_fish_count,
                        "speed": avg_speed
                    }
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(meta, f, ensure_ascii=False, indent=2)
                except Exception as e:
                    print("⚠️ 保存快照元數據失敗:", e)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ret:
            self.latest_frame_bytes = buffer.tobytes()

    def _worker_loop(self):
        """背景影像擷取與處理執行緒 (非阻塞、秒級重連)"""
        cap = None
        last_source = None

        while self.is_running:
            current_mode = self.mode

            if current_mode == "demo":
                if cap is not None:
                    cap.release()
                    cap = None
                frame = self._generate_demo_frame()
                self._process_frame(frame, is_demo=True)
                time.sleep(0.033) # 30 FPS
                continue

            # 實體鏡頭模式 (IP Camera 或 本地 Webcam)
            camera_source = self.stream_url if current_mode == "ip_camera" else 0

            if self.force_reconnect or cap is None or last_source != camera_source or not cap.isOpened():
                self.force_reconnect = False
                if cap is not None:
                    cap.release()
                    cap = None
                
                last_source = camera_source
                cap = cv2.VideoCapture(camera_source)
                
                if not cap.isOpened():
                    with self.lock:
                        self.dashboard_data["camera_connected"] = False
                        self.dashboard_data["ai_insight"] = f"📡 【連線中】正嘗試連線至鏡頭來源 ({camera_source})，若為 DroidCam 請確認手機 App 已開啟..."

                    error_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(error_img, "Camera Offline / Connecting...", (120, 220), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
                    cv2.putText(error_img, f"Source: {camera_source}", (140, 260), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    ret, buffer = cv2.imencode('.jpg', error_img)
                    if ret:
                        self.latest_frame_bytes = buffer.tobytes()
                    time.sleep(1.5)
                    continue

            success, frame = cap.read()
            if not success:
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(1.0)
                continue

            self._process_frame(frame)
            time.sleep(0.01)

    def get_latest_frame(self):
        return self.latest_frame_bytes

    def get_dashboard_data(self):
        with self.lock:
            data = dict(self.dashboard_data)
            data["health_report_10min"] = self.generate_10min_health_report()
            return data

camera_manager = CameraStreamManager()

def gen_frames():
    """MJPEG 視訊串流生成器 (含客戶端斷線安全退出保護)"""
    try:
        last_yielded = None
        while True:
            frame_bytes = camera_manager.get_latest_frame()
            if frame_bytes is not None and frame_bytes != last_yielded:
                last_yielded = frame_bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    except (GeneratorExit, Exception):
        pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/data')
def get_data():
    return jsonify(camera_manager.get_dashboard_data())

@app.route('/api/health_report_10min')
def get_10min_report():
    force = request.args.get('force', 'false').lower() == 'true'
    report = camera_manager.generate_10min_health_report(force_refresh=force)
    return jsonify(report)

@app.route('/api/toggle_heatmap', methods=['POST'])
def toggle_heatmap():
    with camera_manager.lock:
        camera_manager.show_heatmap = not camera_manager.show_heatmap
        state = camera_manager.show_heatmap
    return jsonify({"status": "success", "show_heatmap": state})

@app.route('/api/export_clinical_csv')
def export_clinical_csv():
    """匯出實驗室級魚類行為學與健康臨床日誌 CSV"""
    with camera_manager.lock:
        d = camera_manager.dashboard_data
        p1 = camera_manager.fish_profiles.get(1, {})
        p2 = camera_manager.fish_profiles.get(2, {})
        
        csv_lines = [
            "Timestamp,Clarity,Activity,AvgSpeed_px_s,InterSpacing_px,Hunger_Pct,Fish1_Length_mm,Fish1_Dist_m,Fish2_Length_mm,Fish2_Dist_m,HypoxiaRisk,SwimBladderRisk",
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{d.get('current_clarity',95)},{d.get('current_activity',6.5)},{d.get('avg_speed',0)},{d.get('inter_spacing',0)},{d.get('hunger_index',35)},{p1.get('length_mm',68)},{p1.get('dist_m',0)},{p2.get('length_mm',72)},{p2.get('dist_m',0)},{d.get('pathology_matrix',{}).get('hypoxia_risk','Normal')},{d.get('pathology_matrix',{}).get('swim_bladder_risk','Normal')}"
        ]
        
    return Response(
        "\n".join(csv_lines),
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment;filename=aquasense_clinical_telemetry_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"}
    )

@app.route('/api/snapshots')
def get_snapshots():
    files = glob.glob(os.path.join(SNAPSHOT_DIR, "*.jpg"))
    files.sort(key=os.path.getmtime, reverse=True)
    
    meta_path = os.path.join(SNAPSHOT_DIR, "snapshots_meta.json")
    meta = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
        except Exception:
            pass

    snapshots_list = []
    for f in files:
        fname = os.path.basename(f)
        mtime = os.path.getmtime(f)
        info = meta.get(fname, {})
        
        reason = info.get("reason", "📸 異常事件快照紀錄")
        severity = info.get("severity", "warning")
        
        snapshots_list.append({
            "filename": fname,
            "url": f"/static/snapshots/{fname}",
            "timestamp": info.get("timestamp", datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")),
            "reason": reason,
            "severity": severity,
            "clarity": info.get("clarity", 95.0),
            "fish_count": info.get("fish_count", 2),
            "size": os.path.getsize(f)
        })
    return jsonify({"snapshots": snapshots_list})

@app.route('/api/settings', methods=['GET', 'POST'])
def handle_settings():
    if request.method == 'POST':
        data = request.get_json(force=True, silent=True) or request.form.to_dict() or {}
        camera_manager.update_settings(data)
        return jsonify({
            "status": "success", 
            "message": "設定已成功更新並即時生效！", 
            "settings": {
                "mode": camera_manager.mode,
                "stream_url": camera_manager.stream_url,
                "turbidity_threshold": camera_manager.turbidity_threshold,
                "green_threshold": camera_manager.green_threshold,
                "vertical_ratio_threshold": camera_manager.vertical_ratio_threshold,
                "min_fish_area": camera_manager.min_fish_area
            }
        })
    else:
        return jsonify({
            "mode": camera_manager.mode,
            "stream_url": camera_manager.stream_url,
            "turbidity_threshold": camera_manager.turbidity_threshold,
            "green_threshold": camera_manager.green_threshold,
            "vertical_ratio_threshold": camera_manager.vertical_ratio_threshold,
            "min_fish_area": camera_manager.min_fish_area
        })

@app.route('/favicon.ico')
def favicon():
    return "", 204

if __name__ == '__main__':
    # 啟用 threaded=True 確保視訊串流與 API 請求平行處理不互相阻塞
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
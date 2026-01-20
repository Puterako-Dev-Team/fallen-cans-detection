import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import cv2
from ultralytics import YOLO
from datetime import datetime
import json
import os
from PIL import Image, ImageTk
import threading
import torch
import queue
from collections import deque
import time
import numpy as np

class ModelManager:
    """Singleton untuk manage shared YOLO model"""
    _instance = None
    _model = None
    _lock = threading.Lock()
    _device = None
    
    @classmethod
    def get_model(cls, model_path):
        if cls._model is None:
            with cls._lock:
                if cls._model is None:  # Double-check
                    print(f"🔄 Loading model: {model_path}")
                    cls._model = YOLO(model_path)
                    
                    # Tentukan device: CUDA jika tersedia, CPU jika tidak
                    if torch.cuda.is_available():
                        cls._device = 'cuda'
                        cls._model.to('cuda')
                        print(f"✅ Model loaded to CUDA (GPU: {torch.cuda.get_device_name(0)})")
                        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                    else:
                        cls._device = 'cpu'
                        print("⚠️ CUDA tidak tersedia, menggunakan CPU (performa lebih rendah)")
        return cls._model
    
    @classmethod
    def get_device(cls):
        """Dapatkan device yang sedang digunakan"""
        return cls._device if cls._device else ('cuda' if torch.cuda.is_available() else 'cpu')
    
    @classmethod
    def predict_batch(cls, frames, conf_threshold):
        """Batch prediction untuk multiple frames"""
        with cls._lock:
            if cls._model is None:
                return []
            # Process semua frames sekaligus (batch processing)
            # Gunakan half precision hanya untuk CUDA
            use_half = cls._device == 'cuda'
            results = cls._model(frames, conf=conf_threshold, verbose=False, stream=False, half=use_half)
            return results

    @classmethod
    def clear_cache(cls):
        """Clear GPU/CPU cache setelah selesai"""
        if cls._device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()


class CameraPanel:
    app_config = {}
    log_callback = None
    
    def __init__(self, parent, camera_id, on_remove_callback):
        self.camera_id = camera_id
        self.on_remove_callback = on_remove_callback
        self.is_running = False
        self.cap = None
        self.thread = None
        
        # Detection variables
        self.frame_count = 0
        self.fps = 0
        self.detected_cans = 0
        self.total_detections = 0

        # Tracking kaleng individual dengan ID
        self.tracked_cans = {}  
        self.next_can_id = 1
        self.max_distance = 50
        
        # Log throttling untuk mencegah spam
        self.last_detection_count = 0
        self.last_log_frame = 0
        self.log_interval = CameraPanel.app_config.get('log_interval_frames', 30)  # Dari config atau default 30 frame
        self.last_alert_state = False
        
        # Frame queue untuk optimization
        self.frame_queue = deque(maxlen=2)  # Buffer kecil
        
        # Frame
        self.frame = ttk.LabelFrame(parent, text=f"Camera {camera_id}", padding=10)
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Video display
        self.video_label = tk.Label(self.frame, bg='black', width=400, height=300)
        self.video_label.grid(row=0, column=0, columnspan=3, pady=5, sticky='ew')
        self.update_placeholder()
        
        # Source input
        src_frame = ttk.Frame(self.frame)
        src_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=5)
        ttk.Label(src_frame, text="Src:").pack(side='left', padx=5)
        self.source_var = tk.StringVar(value="")
        self.source_entry = ttk.Entry(src_frame, textvariable=self.source_var)
        self.source_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # Control buttons
        btn_frame = ttk.Frame(self.frame)
        btn_frame.grid(row=2, column=0, columnspan=3, pady=5)
        
        self.start_btn = ttk.Button(btn_frame, text="▶ Start", command=self.start_camera, width=12)
        self.start_btn.pack(side='left', padx=2)
        
        self.stop_btn = ttk.Button(btn_frame, text="⬛ Stop", command=self.stop_camera, width=12, state='disabled')
        self.stop_btn.pack(side='left', padx=2)
        
        # Status labels
        self.status_label = ttk.Label(self.frame, text="Inactive", foreground='gray')
        self.status_label.grid(row=3, column=0, sticky='w', pady=2)
        
        self.fps_label = ttk.Label(self.frame, text="FPS: 0")
        self.fps_label.grid(row=3, column=1, pady=2)
        
        self.count_label = ttk.Label(self.frame, text="Detected: 0", foreground='green')
        self.count_label.grid(row=3, column=2, sticky='e', pady=2)
        
        # GPU info label
        self.gpu_label = ttk.Label(self.frame, text="", foreground='blue', font=('Arial', 8))
        self.gpu_label.grid(row=4, column=0, columnspan=3, pady=2)
    
    def log(self, message, level="INFO"):
        if CameraPanel.log_callback:
            CameraPanel.log_callback(message, level, self.camera_id)
    
    def update_placeholder(self):
        placeholder = Image.new('RGB', (320, 240), color='black')
        photo = ImageTk.PhotoImage(placeholder)
        self.video_label.configure(image=photo)
        self.video_label.image = photo
        self.video_label.configure(text=f"Camera {self.camera_id}\nNot Active", 
                                   compound='center', fg='white', font=('Arial', 12))
    
    def start_camera(self):
        source = self.source_var.get().strip()
        if not source:
            messagebox.showwarning("Warning", "Masukkan source video/RTSP URL!")
            return
        
        self.is_running = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.source_entry.config(state='disabled')
        self.status_label.config(text="Active", foreground='green')
        
        self.log(f"Camera started with source: {source}", "INFO")
        
        # Start detection thread
        self.thread = threading.Thread(target=self.detection_loop, daemon=True)
        self.thread.start()
    
    def stop_camera(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.source_entry.config(state='normal')
        self.status_label.config(text="Inactive", foreground='gray')
        self.update_placeholder()
        self.log("Camera stopped", "INFO")
    
    def detection_loop(self):
        config = CameraPanel.app_config
        source = self.source_var.get().strip()
        
        try:
            source = int(source)
        except ValueError:
            pass

        # Normalisasi nama kelas dari config
        base_class_name = config.get('class_name', 'cans').strip()
        fallen_class_name = config.get('fallen_class_name', '').strip() or None
        base_class_name_l = base_class_name.lower()
        fallen_class_name_l = fallen_class_name.lower() if fallen_class_name else None

        # Load shared model
        try:
            model = ModelManager.get_model(config['model_path'])
            device = ModelManager.get_device()
            if device == 'cuda':
                self.log("Using shared CUDA model", "SUCCESS")
            else:
                self.log("Using CPU model (slower)", "WARNING")
        except Exception as e:
            self.log(f"Failed to load model: {str(e)}", "ERROR")
            messagebox.showerror("Error", f"Gagal load model: {str(e)}")
            self.stop_camera()
            return

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            self.log(f"Failed to open video source: {source}", "ERROR")
            messagebox.showerror("Error", f"Gagal membuka video source: {source}")
            self.stop_camera()
            return

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.fps = 30

        last_time = time.time()
        frame_counter = 0
        fps_process = 0
        device = ModelManager.get_device()

        last_display_update = time.time()
        DISPLAY_UPDATE_INTERVAL = 0.05
        
        GRID_SIZE = 100
        frame_skip_counter = 0

        # ============================================
        # CONVEYOR BELT TRACKING SYSTEM
        # ============================================
        # Get frame dimensions untuk setup ROI
        ret, sample_frame = self.cap.read()
        if not ret:
            self.log("Failed to read sample frame", "ERROR")
            self.stop_camera()
            return
        
        frame_height, frame_width = sample_frame.shape[:2]
        
        # ROI Configuration (bisa di-adjust via config)
        roi_config = config.get('roi', {
            'entry_line_y': 0.05,      # 15% dari atas (entry point)
            'exit_line_y': 0.85,       # 85% dari atas (exit point)
            'active_zone_top': 0.10,   # 20% dari atas (mulai tracking)
            'active_zone_bottom': 0.84 # 80% dari atas (stop tracking)
        })
        
        # Convert percentage to pixels
        ENTRY_LINE_Y = int(frame_height * roi_config['entry_line_y'])
        EXIT_LINE_Y = int(frame_height * roi_config['exit_line_y'])
        ACTIVE_ZONE_TOP = int(frame_height * roi_config['active_zone_top'])
        ACTIVE_ZONE_BOTTOM = int(frame_height * roi_config['active_zone_bottom'])
        
        # Tracking states
        STATE_ENTERING = 'entering'    # Belum melewati entry line
        STATE_TRACKING = 'tracking'    # Sedang di-track (di zona aktif)
        STATE_EXITING = 'exiting'      # Sudah melewati exit line (normal)
        STATE_DROPPED = 'dropped'      # Hilang di tengah (ALERT!)
        
        # Tracking data dengan state
        tracked_cans_stateful = {}  # {can_id: {cx, cy, state, last_seen_frame, ...}}
        next_can_id = 1
        
        # Statistics
        total_entered = 0
        total_exited = 0
        total_dropped = 0

        # ✅ Fallen cans simple tracker
        fallen_tracks = {}           # {fallen_id: {cx, cy, last_seen_frame, stable_frames, alerted}}
        next_fallen_id = 1
        FALLEN_STABLE_FRAMES = config.get('fallen_stable_frames', 1)
        FALLEN_MAX_DISTANCE = config.get('dist_threshold', 60)
        fallen_class_name = config.get('fallen_class_name', None)
        FALLEN_CONF_THRESHOLD = config.get('fallen_conf_threshold', config.get('count_conf_threshold', 0.6))
        
        self.log(f"ROI Setup: Entry={ENTRY_LINE_Y}px, Exit={EXIT_LINE_Y}px, Active={ACTIVE_ZONE_TOP}-{ACTIVE_ZONE_BOTTOM}px", "INFO")

        RECONNECT_DELAY = 5  # detik

        while self.is_running:
            # Cek koneksi, kalau cap belum ada atau sudah mati, lakukan reconnect
            if self.cap is None or not self.cap.isOpened():
                self.log("RTSP connection lost, trying to reconnect...", "WARNING")
                try:
                    if self.cap:
                        self.cap.release()
                except:
                    pass

                while self.is_running:
                    self.cap = cv2.VideoCapture(source)
                    if self.cap.isOpened():
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.log("RTSP reconnected successfully", "SUCCESS")
                        # Cek sample frame untuk pastikan benar-benar dapat frame
                        ret, sample_frame = self.cap.read()
                        if ret:
                            break
                        else:
                            self.cap.release()
                            self.cap = None
                    self.log("Reconnect failed, retrying...", "WARNING")
                    time.sleep(RECONNECT_DELAY)
                if not self.is_running:
                    break
                # Update frame size, dst, jika perlu

            # Baca frame
            ret, frame = self.cap.read()
            if not ret:
                self.log("Failed to read frame, will try to reconnect...", "WARNING")
                try:
                    self.cap.release()
                except:
                    pass
                self.cap = None
                time.sleep(1)
                continue

            self.frame_count += 1
            alert_messages = []

            # Inference
            use_half = device == 'cuda'
            results = model(
                frame,
                conf=config['conf_threshold'],
                # ini untuk deteksi jika pakai RTSP
                # iou=0.5,
                # imgsz=320,
                # ini untuk deteksi jika pakai Video Mp4
                iou=0.5,
                imgsz=640,
                verbose=False,
                half=use_half,
                device=device,
                max_det=300
            )

            # Extract detections
            all_detections = []
            counted_centroids_list = []
            fallen_detections = []
            
            for result in results:
                boxes = result.boxes
                if len(boxes) == 0:
                    continue
                
                cls_ids = boxes.cls.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                
                valid_names = {config['class_name']}
                if fallen_class_name:
                    valid_names.add(fallen_class_name)
                valid_mask = np.array([result.names[cls_id] in valid_names for cls_id in cls_ids])
                
                cls_ids = cls_ids[valid_mask]
                confs = confs[valid_mask]
                xyxy = xyxy[valid_mask]
                
                if len(xyxy) == 0:
                    continue
                
                centroids = ((xyxy[:, 0] + xyxy[:, 2]) // 2, (xyxy[:, 1] + xyxy[:, 3]) // 2)
                cx_arr = centroids[0]
                cy_arr = centroids[1]
                
                count_mask = confs >= config['count_conf_threshold']
                
                for i in range(len(xyxy)):
                    class_name = result.names[cls_ids[i]]
                    is_cans = (class_name == config['class_name'])
                    is_fallen = (fallen_class_name is not None and class_name == fallen_class_name)
                    
                    # ✅ Hitung ukuran bounding box
                    x1, y1, x2, y2 = xyxy[i]
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    
                    # ✅ Filter: Skip jika bbox terlalu besar (kertas besar/background)
                    max_width = config.get('max_bbox_width', 200)
                    max_height = config.get('max_bbox_height', 200)
                    max_area = config.get('max_bbox_area', 30000)
                    min_area = config.get('min_bbox_area', 100)
                    
                    if bbox_width > max_width or bbox_height > max_height or bbox_area > max_area:
                        continue  # Skip deteksi yang terlalu besar
                    
                    if bbox_area < min_area:
                        continue  # Skip deteksi yang terlalu kecil (noise)

                    det = {
                        'bbox': tuple(xyxy[i]),
                        'cx': int(cx_arr[i]),
                        'cy': int(cy_arr[i]),
                        'conf': float(confs[i]),
                        'is_green': bool(is_cans and count_mask[i]),
                        'label': class_name,
                        'width': int(bbox_width),   # ✅ Simpan ukuran untuk debugging
                        'height': int(bbox_height),
                        'area': int(bbox_area),
                    }
                    all_detections.append(det)
                    
                    if is_cans and count_mask[i]:
                        counted_centroids_list.append((det['cx'], det['cy']))
                    
                    if is_fallen and det['conf'] >= FALLEN_CONF_THRESHOLD:
                        fallen_detections.append(det)

            for det in fallen_detections:
                fx, fy = det['cx'], det['cy']
                best_id = None
                best_dist = FALLEN_MAX_DISTANCE
                
                for fid, fdata in fallen_tracks.items():
                    dist = ((fx - fdata['cx'])**2 + (fy - fdata['cy'])**2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_id = fid
                
                if best_id is not None:
                    track = fallen_tracks[best_id]
                    track['cx'] = fx
                    track['cy'] = fy
                    track['last_seen_frame'] = self.frame_count
                    track['stable_frames'] += 1
                else:
                    fallen_tracks[next_fallen_id] = {
                        'cx': fx,
                        'cy': fy,
                        'last_seen_frame': self.frame_count,
                        'stable_frames': 1,
                        'alerted': False,
                    }
                    next_fallen_id += 1

            # Evaluasi fallen_tracks → kirim alert kalau stabil di active zone
            for fid in list(fallen_tracks.keys()):
                fdata = fallen_tracks[fid]
                frames_missing = self.frame_count - fdata['last_seen_frame']
                
                # hapus track yang sudah lama hilang
                if frames_missing > FALLEN_STABLE_FRAMES * 2:
                    del fallen_tracks[fid]
                    continue

                if (not fdata['alerted']
                    and fdata['stable_frames'] >= FALLEN_STABLE_FRAMES
                    and ACTIVE_ZONE_TOP <= fdata['cy'] <= ACTIVE_ZONE_BOTTOM):
                    
                    video_time = f"{int(self.frame_count/self.fps//60):02d}:{int(self.frame_count/self.fps%60):02d}"
                    alert_messages.append(
                        f"🚨 ALERT: Trash Paper detected at Y={fdata['cy']}px | "
                        f"Frame: {self.frame_count}"
                    )
                    fdata['alerted'] = True
                    total_dropped += 1  # hitung sebagai problem juga


            # ============================================
            # CONVEYOR BELT STATE MACHINE TRACKING
            # ============================================
            MAX_MISSING_FRAMES = config.get('debounce_frames', 10)
            STABLE_FRAMES_REQUIRED = config.get('lock_frames', 3)
            MAX_DISTANCE = config.get('dist_threshold', 60)
            
            num_detections = len(counted_centroids_list)
            num_tracked = len(tracked_cans_stateful)
            
            if num_tracked > 0 and num_detections > 0:
                # Build spatial grid
                grid = {}
                for can_id, can_data in tracked_cans_stateful.items():
                    # Skip yang sudah exit/dropped
                    if can_data['state'] in [STATE_EXITING, STATE_DROPPED]:
                        continue
                        
                    gx = can_data['cx'] // GRID_SIZE
                    gy = can_data['cy'] // GRID_SIZE
                    grid_key = (gx, gy)
                    
                    if grid_key not in grid:
                        grid[grid_key] = []
                    grid[grid_key].append((can_id, can_data['cx'], can_data['cy']))
                
                # Match detections
                matched_tracked_ids = set()
                matched_detection_indices = set()
                
                for d_idx, (det_cx, det_cy) in enumerate(counted_centroids_list):
                    det_gx = det_cx // GRID_SIZE
                    det_gy = det_cy // GRID_SIZE
                    
                    best_match_id = None
                    best_distance = MAX_DISTANCE
                    
                    for gx in range(det_gx - 1, det_gx + 2):
                        for gy in range(det_gy - 1, det_gy + 2):
                            grid_key = (gx, gy)
                            if grid_key not in grid:
                                continue
                            
                            for can_id, can_cx, can_cy in grid[grid_key]:
                                if can_id in matched_tracked_ids:
                                    continue
                                
                                # Skip cans yang sudah dropped/exited
                                if tracked_cans_stateful[can_id]['state'] in [STATE_DROPPED, STATE_EXITING]:
                                    continue
                                
                                dist = ((det_cx - can_cx)**2 + (det_cy - can_cy)**2)**0.5
                                if dist < best_distance:
                                    best_distance = dist
                                    best_match_id = can_id
                    
                    if best_match_id:
                        # Update tracked can
                        old_state = tracked_cans_stateful[best_match_id]['state']
                        old_cy = tracked_cans_stateful[best_match_id]['cy']
                        
                        tracked_cans_stateful[best_match_id].update({
                                'cx': det_cx,
                                'cy': det_cy,
                                'last_seen_frame': self.frame_count,
                                'stable_frames': tracked_cans_stateful[best_match_id]['stable_frames'] + 1,
                                'missing_count': 0,  # RESET: Kembali terdeteksi
                                'was_missing': False
                            })
                        matched_tracked_ids.add(best_match_id)
                        matched_detection_indices.add(d_idx)
                
                # Add new cans (unmatched detections in active zone)
                for d_idx, (cx, cy) in enumerate(counted_centroids_list):
                    if d_idx not in matched_detection_indices:
                        # Determine initial state based on position
                        if cy < ENTRY_LINE_Y:
                            initial_state = STATE_ENTERING
                        elif cy > EXIT_LINE_Y:
                            initial_state = STATE_EXITING  # Skip cans yang muncul di bawah
                        else:
                            initial_state = STATE_TRACKING
                        
                        tracked_cans_stateful[next_can_id] = {
                            'cx': cx,
                            'cy': cy,
                            'last_seen_frame': self.frame_count,
                           'first_seen_frame': self.frame_count,  # penting untuk MIN_TRACKING_DURATION
                            'stable_frames': 1,
                            'state': initial_state,
                            'alerted': False,
                            'missing_count': 0,
                            'was_missing': False 
                        }
                        next_can_id += 1
            else:
                # Bootstrap: add all detections
                for cx, cy in counted_centroids_list:
                    if cy < ENTRY_LINE_Y:
                        initial_state = STATE_ENTERING
                    elif cy > EXIT_LINE_Y:
                        initial_state = STATE_EXITING
                    else:
                        initial_state = STATE_TRACKING
                    
                    tracked_cans_stateful[next_can_id] = {
                        'cx': cx,
                        'cy': cy,
                        'last_seen_frame': self.frame_count,
                        'first_seen_frame': self.frame_count,
                        'stable_frames': 1,
                        'state': initial_state,
                        'alerted': False,
                        'missing_count': 0, 
                        'was_missing': False
                    }
                    next_can_id += 1
            
            # ============================================
            # DETECT DROPPED CANS (Missing in active zone)
            # ============================================
            GRACE_PERIOD = config.get('grace_period_frames', 25)  # ~1.5 detik toleransi
            MIN_TRACKING_DURATION = config.get('min_tracking_duration', 10)  # Minimal 10 frame tracking
            
            to_remove = []

            use_missing_drop_alert = not bool(fallen_class_name)

            for can_id, can_data in tracked_cans_stateful.items():
                frames_missing = self.frame_count - can_data['last_seen_frame']
                tracking_duration = can_data['last_seen_frame'] - can_data.get('first_seen_frame', can_data['last_seen_frame'])
                
                # Skip yang sudah exiting (normal case)
                if can_data['state'] == STATE_EXITING:
                    if frames_missing > MAX_MISSING_FRAMES:
                        to_remove.append(can_id)
                    continue
                
                # Update missing count untuk can yang sedang tracking
                if can_data['state'] == STATE_TRACKING and frames_missing > 0:
                    if not can_data.get('was_missing', False):
                        can_data['missing_count'] = can_data.get('missing_count', 0) + 1
                        can_data['was_missing'] = True
                
                # Reset alerted jika can kembali terdeteksi setelah dropped
                if can_data['state'] == STATE_DROPPED and frames_missing == 0:
                    # Can kembali muncul setelah dropped, reset state untuk tracking ulang
                    can_data['state'] = STATE_TRACKING
                    can_data['alerted'] = False  # RESET FLAG
                    can_data['first_seen_frame'] = self.frame_count  # Reset tracking duration
                    self.log(f"Can #{can_id} recovered after drop (Y={can_data['cy']}px)", "WARNING")
                
                # ALERT: Can dropped dengan grace period
                if can_data['state'] == STATE_TRACKING:
                    if (use_missing_drop_alert and
                        can_data['stable_frames'] >= STABLE_FRAMES_REQUIRED and
                        tracking_duration >= MIN_TRACKING_DURATION and
                        frames_missing > GRACE_PERIOD and
                        not can_data['alerted']):
                        
                        video_time = f"{int(self.frame_count/self.fps//60):02d}:{int(self.frame_count/self.fps%60):02d}"
                        alert_messages.append(
                            f"🚨 ALERT: Can #{can_id} DROPPED at Y={can_data['cy']}px | "
                            f"Missing: {frames_missing} frames ({frames_missing/self.fps:.1f}s) | "
                            f"Frame: {self.frame_count} | Time: {video_time}"
                        )
                        tracked_cans_stateful[can_id]['state'] = STATE_DROPPED
                        tracked_cans_stateful[can_id]['alerted'] = True
                        total_dropped += 1
                    
                    # Cleanup setelah grace period + buffer
                    if frames_missing > GRACE_PERIOD + MAX_MISSING_FRAMES:
                        to_remove.append(can_id)
                
                # Clean up entering cans yang hilang (probably false positive)
                elif can_data['state'] == STATE_ENTERING and frames_missing > MAX_MISSING_FRAMES:
                    to_remove.append(can_id)
                
                # Clean up dropped cans yang tidak kembali dalam waktu lama
                elif can_data['state'] == STATE_DROPPED and frames_missing > GRACE_PERIOD * 2:
                    to_remove.append(can_id)
            
            # Batch delete
            for can_id in to_remove:
                del tracked_cans_stateful[can_id]
            
            # Batch logging
            if alert_messages:
                for msg in alert_messages:
                    self.log(msg, "ALERT")
            
            # Count active cans (entering + tracking only)
            active_cans = sum(1 for can in tracked_cans_stateful.values() 
                            if can['state'] in [STATE_ENTERING, STATE_TRACKING] 
                            and can['stable_frames'] >= STABLE_FRAMES_REQUIRED)

            # ============================================
            # VISUALIZATION with ROI lines
            # ============================================
            current_time = time.time()
            should_update_display = (current_time - last_display_update) >= DISPLAY_UPDATE_INTERVAL
            
            if should_update_display:
                annotated = frame.copy()
                
                # Draw ROI lines
                cv2.line(annotated, (0, ENTRY_LINE_Y), (frame_width, ENTRY_LINE_Y), (0, 255, 255), 2)  # Yellow: Entry
                cv2.line(annotated, (0, EXIT_LINE_Y), (frame_width, EXIT_LINE_Y), (255, 0, 255), 2)    # Magenta: Exit
                cv2.rectangle(annotated, (0, ACTIVE_ZONE_TOP), (frame_width, ACTIVE_ZONE_BOTTOM), (0, 255, 0), 2)  # Green: Active zone
                
                # Add ROI labels
                cv2.putText(annotated, "ENTRY LINE", (10, ENTRY_LINE_Y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(annotated, "EXIT LINE", (10, EXIT_LINE_Y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                cv2.putText(annotated, "ACTIVE ZONE", (frame_width - 150, ACTIVE_ZONE_TOP + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Draw detections with state colors (tanpa teks ID)
                for det in all_detections:
                    x1, y1, x2, y2 = det['bbox']
                    cx, cy = det['cx'], det['cy']
                    class_name = det.get('label')
                    
                    # Find if this detection is tracked
                    color = (0, 165, 255)  # Default: Orange (untracked)
                    is_tracking = False

                    # Fallen cans: selalu merah, tidak ikut state conveyor
                    if fallen_class_name and class_name == fallen_class_name:
                        color = (0, 0, 255)
                    else:
                        # Find if this detection is tracked sebagai cans biasa
                        for can_id, can_data in tracked_cans_stateful.items():
                            if abs(can_data['cx'] - cx) < 20 and abs(can_data['cy'] - cy) < 20:
                                if can_data['state'] == STATE_ENTERING:
                                    color = (255, 255, 0)  # Cyan: Entering
                                elif can_data['state'] == STATE_TRACKING:
                                    color = (0, 255, 0)  # Green: Tracking
                                    is_tracking = True
                                elif can_data['state'] == STATE_EXITING:
                                    color = (255, 0, 255)  # Magenta: Exiting
                                break
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
                    
                    # Gambar titik kecil di tengah kaleng yang sedang di-track
                    if is_tracking:
                        cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)  # Titik hijau solid
                        cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)  # Border putih

                # Info overlay (hanya statistik total)
                video_time = f"{int(self.frame_count/self.fps//60):02d}:{int(self.frame_count/self.fps%60):02d}"
                info_texts = [
                    f'Frame: {self.frame_count} | Time: {video_time}',
                    f'Active Cans: {active_cans}',
                ]
                
                y_pos = 25
                for text in info_texts:
                    # Background untuk readability
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    cv2.rectangle(annotated, (5, y_pos - 18), (tw + 15, y_pos + 5), (0, 0, 0), -1)
                    cv2.putText(annotated, text, (10, y_pos), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    y_pos += 30

                # Display
                display_size = 400
                h_in, w_in = annotated.shape[:2]
                ratio = display_size / max(h_in, w_in)
                target_w = int(w_in * ratio)
                target_h = int(h_in * ratio)
                
                resized = cv2.resize(annotated, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                
                x_pad = (display_size - target_w) // 2
                y_pad = (display_size - target_h) // 2
                
                display_frame = np.zeros((display_size, display_size, 3), dtype=np.uint8)
                display_frame[y_pad:y_pad + target_h, x_pad:x_pad + target_w] = resized
                
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                photo = ImageTk.PhotoImage(image=img)
                
                self.video_label.configure(image=photo, text='')
                self.video_label.image = photo
                
                last_display_update = current_time
            
            self.detected_cans = active_cans
            self.total_detections = len(all_detections)
            
            # Status update
            frame_counter += 1
            if frame_counter >= 30:
                now = time.time()
                fps_process = frame_counter / (now - last_time)
                last_time = now
                frame_counter = 0
                
                if device == 'cuda' and torch.cuda.is_available():
                    mem_used = torch.cuda.memory_allocated(0) / 1024**3
                    self.gpu_label.config(text=f"GPU: {mem_used:.2f}GB | Dropped: {total_dropped}")
                else:
                    self.gpu_label.config(text=f"CPU | Dropped: {total_dropped}")
                
                self.fps_label.config(text=f"FPS: {fps_process:.1f}")
                self.count_label.config(text=f"Active: {active_cans} | Drop: {total_dropped}")

        if self.cap:
            self.cap.release()
        self.is_running = False
        ModelManager.clear_cache()
        self.log(f"Detection stopped | Final stats: Entered={total_entered}, Exited={total_exited}, Dropped={total_dropped}", "INFO")

    def get_config(self):
        return {
            'source': self.source_var.get(),
            'camera_id': self.camera_id
        }
    
    def set_config(self, conf):
        self.source_var.set(conf.get('source', ''))


class ConfigDialog:
    def __init__(self, parent, current_config, callback):
        self.callback = callback
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Konfigurasi Sistem")
        self.dialog.geometry("400x500")
        self.dialog.resizable(False, False)
        
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        main_frame = ttk.Frame(self.dialog, padding=20)
        main_frame.pack(fill='both', expand=True)
        
        configs = [
            ('Model Path:', 'model_path', current_config['model_path']),
            ('Class Name:', 'class_name', current_config['class_name']),
            ('Fallen Class Name:', 'fallen_class_name', current_config.get('fallen_class_name', 'fallen_cans')),  
            ('Conf Threshold:', 'conf_threshold', current_config['conf_threshold']),
            ('Count Conf Threshold:', 'count_conf_threshold', current_config['count_conf_threshold']),
            ('Fallen Conf Threshold:', 'fallen_conf_threshold', current_config.get('fallen_conf_threshold', 0.75)),
            ('Distance Threshold:', 'dist_threshold', current_config['dist_threshold']),
            ('Alert Cooldown (frames):', 'alert_cooldown_frames', current_config['alert_cooldown_frames']),
            ('Display Scale:', 'display_scale', current_config['display_scale']),
            ('Debounce (frames):', 'debounce_frames', current_config.get('debounce_frames', 5)),  
            ('Lock Frames:', 'lock_frames', current_config.get('lock_frames', 2)), 
            ('Log Interval (frames):', 'log_interval_frames', current_config.get('log_interval_frames', 30)), 
            ('Grace Period (frames):', 'grace_period_frames', current_config.get('grace_period_frames', 25)), 
            ('Min Tracking Duration:', 'min_tracking_duration', current_config.get('min_tracking_duration', 10)), 
            ('Fallen Stable Frames:', 'fallen_stable_frames', current_config.get('fallen_stable_frames', 1)), 

            ('Max BBox Width (px):', 'max_bbox_width', current_config.get('max_bbox_width', 200)),
            ('Max BBox Height (px):', 'max_bbox_height', current_config.get('max_bbox_height', 200)),
            ('Max BBox Area (px²):', 'max_bbox_area', current_config.get('max_bbox_area', 30000)),
            ('Min BBox Area (px²):', 'min_bbox_area', current_config.get('min_bbox_area', 100)),
        ]
        
        self.vars = {}
        for i, (label, key, value) in enumerate(configs):
            ttk.Label(main_frame, text=label).grid(row=i, column=0, sticky='w', pady=5)
            var = tk.StringVar(value=str(value))
            self.vars[key] = var
            ttk.Entry(main_frame, textvariable=var, width=30).grid(row=i, column=1, pady=5)
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=len(configs), column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Simpan", command=self.save).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Batal", command=self.dialog.destroy).pack(side='left', padx=5)
    
    def save(self):
        try:
            new_config = {
                'model_path': self.vars['model_path'].get(),
                'class_name': self.vars['class_name'].get(),
                'fallen_class_name': self.vars['fallen_class_name'].get(),                   
                'conf_threshold': float(self.vars['conf_threshold'].get()),
                'count_conf_threshold': float(self.vars['count_conf_threshold'].get()),
                'fallen_conf_threshold': float(self.vars['fallen_conf_threshold'].get()),
                'dist_threshold': int(self.vars['dist_threshold'].get()),
                'alert_cooldown_frames': int(self.vars['alert_cooldown_frames'].get()),
                'display_scale': float(self.vars['display_scale'].get()),
                'debounce_frames': int(self.vars['debounce_frames'].get()),  
                'lock_frames': int(self.vars['lock_frames'].get()),  
                'log_interval_frames': int(self.vars['log_interval_frames'].get()),  
                'grace_period_frames': int(self.vars['grace_period_frames'].get()),
                'min_tracking_duration': int(self.vars['min_tracking_duration'].get()),
                'fallen_stable_frames': int(self.vars['fallen_stable_frames'].get()),   

                'max_bbox_width': int(self.vars['max_bbox_width'].get()),
                'max_bbox_height': int(self.vars['max_bbox_height'].get()),
                'max_bbox_area': int(self.vars['max_bbox_area'].get()),
                'min_bbox_area': int(self.vars['min_bbox_area'].get()),    
            }
            self.callback(new_config)
            self.dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Error", f"Invalid input: {str(e)}")


class MainApp:
    config = {
        'model_path': 'model-pav-yolo11.pt',
        'class_name': 'full_paper',
        'fallen_class_name': 'paper',  
        'conf_threshold': 0.4,
        'count_conf_threshold': 0.6,
        'dist_threshold': 120,
        'alert_cooldown_frames': 30,
        'display_scale': 0.3,
        'debounce_frames': 15,
        'lock_frames': 5,
        'log_interval_frames': 30,  # Log interval untuk menghindari spam (30 frames = ~1 detik)
        'grace_period_frames': 25,  # Toleransi sebelum alert (45 frames = ~1.5 detik)
        'min_tracking_duration': 10, 
        'fallen_stable_frames': 1,   

        'max_bbox_width': 10000,      # Maksimal lebar bbox (pixels)
        'max_bbox_height': 400,     # Maksimal tinggi bbox (pixels)
        'max_bbox_area': 50000,     # Maksimal area bbox (pixels²)
        'min_bbox_area': 100,
    }
    
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera Can Detection System - Optimized CUDA")
        self.root.geometry("1400x800")
        
        self.cameras = []
        self.camera_panels = []
        
        self.load_config()
        
        CameraPanel.app_config = MainApp.config
        CameraPanel.log_callback = self.add_log
        
        # Header dengan GPU info
        header = ttk.Frame(root, padding=10)
        header.pack(fill='x')
        
        ttk.Label(header, text="Nestle Can Detection System", 
                 font=('Arial', 16, 'bold')).pack(side='left')
        
        # Device info di header
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            self.gpu_info_label = ttk.Label(header, 
                text=f"🎮 GPU: {gpu_name} ({gpu_mem:.1f}GB)", 
                foreground='green', font=('Arial', 9))
            self.gpu_info_label.pack(side='left', padx=20)
        else:
            self.gpu_info_label = ttk.Label(header, 
                text=f"⚙️ CPU Mode (No GPU detected)", 
                foreground='orange', font=('Arial', 9))
            self.gpu_info_label.pack(side='left', padx=20)
        
        ttk.Button(header, text="⚙ Configure", command=self.open_config).pack(side='right', padx=5)
        ttk.Button(header, text="🗑️ Clear Logs", command=self.clear_logs).pack(side='right', padx=5)
        # Main content
        main_content = ttk.PanedWindow(root, orient='horizontal')
        main_content.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Camera panels
        left_frame = ttk.Frame(main_content)
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill='both', expand=True)
        
        self.canvas = tk.Canvas(canvas_frame)
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient='vertical', command=self.canvas.yview)
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient='horizontal', command=self.canvas.xview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self.canvas.pack(side='top', fill='both', expand=True)
        h_scrollbar.pack(side='bottom', fill='x')
        v_scrollbar.pack(side='right', fill='y')
        
        self.camera_grid = ttk.Frame(self.scrollable_frame)
        self.camera_grid.pack(fill='both', expand=True, padx=10, pady=10)
        
        for i in range(2):
            self.camera_grid.grid_columnconfigure(i, minsize=450, weight=0)
        
        # Log panel
        right_frame = ttk.LabelFrame(main_content, text="📋 Activity Log", padding=10)
        main_content.add(left_frame, weight=10)
        main_content.add(right_frame, weight=3)
        
        self.log_text = scrolledtext.ScrolledText(right_frame, wrap=tk.WORD, 
                                                   width=45, height=30,
                                                   font=('Consolas', 9))
        self.log_text.pack(fill='both', expand=True)
        
        self.log_text.tag_config('INFO', foreground='blue')
        self.log_text.tag_config('SUCCESS', foreground='green')
        self.log_text.tag_config('WARNING', foreground='orange')
        self.log_text.tag_config('ERROR', foreground='red')
        self.log_text.tag_config('ALERT', foreground='red', font=('Consolas', 9, 'bold'))
        
        # Control buttons
        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill='x')
        
        tk.Button(control_frame, text="➕ Add Camera", 
                  command=self.add_camera, bg='#4CAF50', fg='white', 
                  font=('Arial', 10, 'bold'), padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="➖ Remove Last Camera", 
                  command=self.remove_last_camera, bg='#f44336', fg='white', 
                  font=('Arial', 10, 'bold'), padx=20, pady=10).pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_label = tk.Label(root, text="Status: 0 camera(s) added", 
                                     background='#1a1a1a', foreground='#4CAF50', 
                                     relief='sunken', anchor='w')
        self.status_label.pack(fill='x', side='bottom')
        
        self.add_log("System initialized with CUDA optimization", "SUCCESS")
        
        self.load_cameras()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    def add_log(self, message, level="INFO", camera_id=None):
        timestamp = datetime.now().strftime("%H:%M:%S")
        cam_prefix = f"[Cam {camera_id}] " if camera_id else ""
        log_entry = f"[{timestamp}] {cam_prefix}{message}\n"
        
        self.log_text.insert(tk.END, log_entry, level)
        self.log_text.see(tk.END)
        
        line_count = int(self.log_text.index('end-1c').split('.')[0])
        if line_count > 1000:
            self.log_text.delete('1.0', '2.0')
    
    def clear_logs(self):
        self.log_text.delete('1.0', tk.END)
        self.add_log("Logs cleared", "INFO")
    
    def add_camera(self):
        camera_id = len(self.camera_panels) + 1
        panel = CameraPanel(self.camera_grid, camera_id, self.remove_camera)
        
        row = (camera_id - 1) // 2
        col = (camera_id - 1) % 2
        panel.frame.grid(row=row, column=col, padx=10, pady=10, sticky='nsew')
        
        self.camera_panels.append(panel)
        self.update_status()
        self.save_cameras()
        self.add_log(f"Camera {camera_id} added", "INFO")
    
    def remove_last_camera(self):
        if self.camera_panels:
            panel = self.camera_panels.pop()
            camera_id = panel.camera_id
            if panel.is_running:
                panel.stop_camera()
            panel.frame.destroy()
            self.update_status()
            self.save_cameras()
            self.add_log(f"Camera {camera_id} removed", "INFO")
    
    def remove_camera(self, panel):
        if panel in self.camera_panels:
            camera_id = panel.camera_id
            self.camera_panels.remove(panel)
            if panel.is_running:
                panel.stop_camera()
            panel.frame.destroy()
            self.update_status()
            self.save_cameras()
            self.add_log(f"Camera {camera_id} removed", "INFO")
    
    def open_config(self):
        ConfigDialog(self.root, MainApp.config, self.save_config)
    
    def save_config(self, new_config):
        MainApp.config = new_config
        CameraPanel.app_config = new_config
        with open('can_detection_config.json', 'w') as f:
            json.dump(new_config, f, indent=2)
        messagebox.showinfo("Success", "Konfigurasi berhasil disimpan!")
        self.add_log("Configuration updated", "SUCCESS")
    
    def load_config(self):
        if os.path.exists('can_detection_config.json'):
            try:
                with open('can_detection_config.json', 'r') as f:
                    MainApp.config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
    
    def save_cameras(self):
        cameras_config = [panel.get_config() for panel in self.camera_panels]
        with open('can_detection_cameras.json', 'w') as f:
            json.dump(cameras_config, f, indent=2)
    
    def load_cameras(self):
        if os.path.exists('can_detection_cameras.json'):
            try:
                with open('can_detection_cameras.json', 'r') as f:
                    cameras_config = json.load(f)
                    for cam_conf in cameras_config:
                        self.add_camera()
                        self.camera_panels[-1].set_config(cam_conf)
            except Exception as e:
                print(f"Error loading cameras: {e}")
    
    def update_status(self):
        active = sum(1 for p in self.camera_panels if p.is_running)
        total = len(self.camera_panels)
        self.status_label.config(text=f"Status: {active} camera(s) active | Total: {total}")
    
    def on_closing(self):
        for panel in self.camera_panels:
            if panel.is_running:
                panel.stop_camera()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()
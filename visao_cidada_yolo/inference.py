"""Funções centrais para detecção, contagem, privacidade e registro de pessoas com YOLO."""

import csv
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Generator

import cv2
import numpy as np
from ultralytics import YOLO

from config import (
    CONFIDENCE, DEVICE, IMAGE_SIZE, IOU_THRESHOLD,
    PERSON_CLASS_ID, RECORDS_FILE, ROOT_DIR, SNAPSHOTS_DIR,
    ensure_project_directories,
)

_MODEL_CACHE: dict[str, YOLO] = {}
_FACE_CASCADE: cv2.CascadeClassifier | None = None

def get_face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if Path(cascade_path).exists():
            _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE

def load_model(weights: str | Path) -> YOLO:
    weights_str = str(weights)
    resolved_path = ROOT_DIR / weights_str if not Path(weights_str).is_absolute() else Path(weights_str)
    key = str(resolved_path if resolved_path.exists() else weights_str)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(key)
    return _MODEL_CACHE[key]

def _unwrap_result(result: Any) -> Any:
    if isinstance(result, tuple) and len(result) >= 1:
        return result[0]
    return result

def apply_preprocessing(frame, enhance=True):
    if not enhance:
        return frame
    
    # Apply CLAHE to the L channel in LAB color space for better detection in low-light
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    
    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_frame

def detect_people(model, frame, confidence=CONFIDENCE, iou=IOU_THRESHOLD, device=DEVICE, augment: bool = False, max_det: int = 300):
    start_time = time.perf_counter()
    results = model.predict(
        source=frame, conf=confidence, iou=iou,
        classes=[PERSON_CLASS_ID], imgsz=IMAGE_SIZE,
        device=device, verbose=False, augment=augment, max_det=max_det
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    res = results[0]
    res.inference_latency_ms = latency_ms
    return res

def detect_people_enhanced(model, frame, confidence=CONFIDENCE, iou=IOU_THRESHOLD, device=DEVICE, augment: bool = False, max_det: int = 300, use_tta: bool = False, multi_scale: bool = False):
    augment = augment or use_tta
    if not multi_scale:
        return detect_people(model, frame, confidence, iou, device, augment, max_det)
    
    start_time = time.perf_counter()
    scales = [640, 800, 1024]
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for scale in scales:
        results = model.predict(
            source=frame, conf=confidence, iou=iou,
            classes=[PERSON_CLASS_ID], imgsz=scale,
            device=device, verbose=False, augment=augment, max_det=max_det
        )
        res = results[0]
        if res.boxes:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            for b, s, c in zip(boxes, scores, classes):
                all_boxes.append(b.tolist())
                all_scores.append(float(s))
                all_classes.append(int(c))
            
    # Base result to mimic YOLO result
    base_res = detect_people(model, frame, confidence, iou, device, augment, max_det)
    
    if all_boxes:
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, confidence, iou)
        if len(indices) > 0:
            final_boxes = [all_boxes[i] for i in indices.flatten()]
            final_scores = [all_scores[i] for i in indices.flatten()]
            final_classes = [all_classes[i] for i in indices.flatten()]
            
            import torch
            device_tensor = base_res.boxes.xyxy.device
            base_res.boxes.xyxy = torch.tensor(final_boxes, device=device_tensor)
            base_res.boxes.conf = torch.tensor(final_scores, device=device_tensor)
            base_res.boxes.cls = torch.tensor(final_classes, device=device_tensor)
        else:
            base_res.boxes = []
    else:
        base_res.boxes = []
        
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    base_res.inference_latency_ms = latency_ms
    return base_res

def count_people(result):
    res = _unwrap_result(result)
    if res is None or getattr(res, "boxes", None) is None:
        return 0
    return len(res.boxes)

def extract_detections_info(result):
    res = _unwrap_result(result)
    if res is None or getattr(res, "boxes", None) is None:
        return []
    detections = []
    boxes = res.boxes
    if len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        conf = float(confs[i])
        detections.append({
            "id": i + 1,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "confidence": round(conf, 4),
            "class_name": "pessoa",
        })
    return detections

def blur_detected_faces(frame, blur_kernel_size=31, boxes=None):
    cascade = get_face_cascade()
    if cascade is None or cascade.empty():
        return frame
    output = frame.copy()
    height, width = output.shape[:2]
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    ksize = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
    
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            roi_gray = gray[y1:y2, x1:x2]
            if roi_gray.size == 0:
                continue
            faces = cascade.detectMultiScale(roi_gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
            for fx, fy, fw, fh in faces:
                fx += x1
                fy += y1
                fx1 = max(0, fx)
                fy1 = max(0, fy)
                fx2 = min(width, fx + fw)
                fy2 = min(height, fy + fh)
                face_roi = output[fy1:fy2, fx1:fx2]
                if face_roi.size > 0:
                    output[fy1:fy2, fx1:fx2] = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
    else:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(20, 20))
        for fx, fy, fw, fh in faces:
            x1 = max(0, fx)
            y1 = max(0, fy)
            x2 = min(width, fx + fw)
            y2 = min(height, fy + fh)
            face_roi = output[y1:y2, x1:x2]
            if face_roi.size > 0:
                output[y1:y2, x1:x2] = cv2.GaussianBlur(face_roi, (ksize, ksize), 0)
    return output

def annotate_frame(result, blur_faces=False, show_labels=True, show_conf=True, thickness=2):
    res = _unwrap_result(result)
    
    if hasattr(res, 'orig_img') and res.orig_img is not None:
        annotated = res.orig_img.copy()
    else:
        # Fallback if orig_img is missing
        annotated = np.zeros((720, 1280, 3), dtype=np.uint8)
        
    if getattr(res, "boxes", None) is None or len(res.boxes) == 0:
        if blur_faces:
            annotated = blur_detected_faces(annotated)
        count = 0
    else:
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)
        count = len(boxes)
        
        for i in range(count):
            x1, y1, x2, y2 = boxes[i].astype(int)
            conf = float(confs[i])
            
            # Color code by confidence
            if conf > 0.7:
                color = (0, 255, 0) # Green
            elif conf >= 0.4:
                color = (0, 255, 255) # Yellow
            else:
                color = (0, 0, 255) # Red
                
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            if show_labels or show_conf:
                label = "pessoa" if show_labels else ""
                if show_conf:
                    label += f" {conf*100:.1f}%"
                label = label.strip()
                
                if label:
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
                    cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    
        if blur_faces:
            annotated = blur_detected_faces(annotated, boxes=boxes)
            
    # Draw small counter overlay in top-left corner
    overlay = annotated.copy()
    cv2.rectangle(overlay, (5, 5), (150, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, annotated, 0.4, 0, annotated)
    cv2.putText(annotated, f"Total: {count}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
    return annotated

def register_detection(source, input_type, count, confidence, device=DEVICE):
    ensure_project_directories()
    new_file = not RECORDS_FILE.exists()
    now = datetime.now()
    with RECORDS_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if new_file:
            writer.writerow(["data", "horario", "origem", "tipo", "quantidade", "confianca_minima", "dispositivo"])
        writer.writerow([
            now.date().isoformat(), now.strftime("%H:%M:%S"),
            Path(source).name, input_type, count,
            round(confidence, 2), device,
        ])

def save_snapshot(frame, prefix="webcam"):
    ensure_project_directories()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{prefix}_{now_str}.jpg"
    target = SNAPSHOTS_DIR / filename
    cv2.imwrite(str(target), frame)
    return target

def process_video(model, video_path, confidence=CONFIDENCE, iou=IOU_THRESHOLD,
                  blur_faces=False, source_name=None, device=DEVICE, frame_step=1):
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    maximum = 0
    frame_idx = 0
    
    fps_ema = 0.0
    alpha = 0.1
    
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_idx += 1
            if frame_step > 1 and (frame_idx % frame_step != 0):
                continue
                
            start_time = time.time()
            result = detect_people(model, frame, confidence=confidence, iou=iou, device=device)
            count = count_people(result)
            maximum = max(maximum, count)
            annotated = annotate_frame(result, blur_faces=blur_faces)
            
            process_time = time.time() - start_time
            current_fps = 1.0 / (process_time + 1e-6)
            
            if fps_ema == 0.0:
                fps_ema = current_fps
            else:
                fps_ema = alpha * current_fps + (1 - alpha) * fps_ema
                
            # Render smoothed FPS to screen
            cv2.putText(annotated, f"FPS: {fps_ema:.1f}", (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
            yield annotated, count, frame_idx, total_frames, video_fps
    finally:
        capture.release()
    register_detection(source_name or video_path, "video", maximum, confidence, device=device)

def load_records_df():
    import pandas as pd
    ensure_project_directories()
    expected_cols = ["data", "horario", "origem", "tipo", "quantidade", "confianca_minima", "dispositivo"]
    if not RECORDS_FILE.exists() or RECORDS_FILE.stat().st_size == 0:
        return pd.DataFrame(columns=expected_cols)
    try:
        df = pd.read_csv(RECORDS_FILE, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(columns=expected_cols)
    if "dispositivo" not in df.columns:
        df["dispositivo"] = "cpu"
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "N/A"
    return df

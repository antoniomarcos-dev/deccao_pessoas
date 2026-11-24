"""
Motor Central de Inferência, Detecção Multi-Escala/Tiling (Alta Densidade),
Anonimização LGPD de Alta Performance, Rastreamento e Persistência com YOLO.
Otimizado para notebooks Intel Core i7 (12ª Geração) + Intel UHD Graphics (OpenVINO FP16).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import threading
import time
from typing import Any, Generator

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from config import (
    CONFIDENCE,
    DEFAULT_MODEL_NAME,
    DEVICE,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS_DEFAULT,
    OPENVINO_IMAGE_SIZE,
    PERSON_CLASS_ID,
    ROOT_DIR,
    SNAPSHOTS_DIR,
    VIDEOS_EXPORT_DIR,
    ensure_project_directories,
)
from utils.logger import get_logger
from utils.storage import classify_scenario_label, get_repository
from utils.tracker import HeatmapAccumulator, PersonTracker, TrackedPerson, render_instant_heatmap

logger = get_logger("inference")

_MODEL_CACHE: dict[str, YOLO] = {}
_WARMED_MODELS: set[str] = set()
_MODEL_LOCK = threading.RLock()
_FAILED_DEVICES: set[str] = set()
_FACE_CASCADE: cv2.CascadeClassifier | None = None


def get_face_cascade() -> cv2.CascadeClassifier | None:
    """Carrega e mantém em cache o classificador Haar Cascade para rostos."""
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if Path(cascade_path).exists():
            _FACE_CASCADE = cv2.CascadeClassifier(cascade_path)
    return _FACE_CASCADE


def _is_openvino_model(model: Any) -> bool:
    model_str = (
        str(getattr(model, "model_name", ""))
        + str(getattr(model, "ckpt_path", ""))
        + str(getattr(model, "overrides", {}).get("model", ""))
    )
    return "openvino" in model_str.lower()


def load_model(weights: str | Path = DEFAULT_MODEL_NAME, device: str | None = None, force_pytorch: bool = False) -> YOLO:
    """
    Carrega e mantém o modelo YOLO em cache thread-safe com aquecimento automático (warmup).
    Quando o dispositivo é `intel:*` e os pesos são `best.pt`, utiliza `best_openvino_model` (FP16)
    para máxima taxa de quadros na GPU integrada Intel UHD ou CPU Intel Core i7.
    """
    weights_str = str(weights)
    orig_weights_str = weights_str

    if (
        not force_pytorch
        and device
        and str(device).startswith("intel:")
        and Path(weights_str).name in {"best.pt", "best_openvino_model"}
    ):
        ov_path = ROOT_DIR / "best_openvino_model"
        if ov_path.exists():
            weights_str = "best_openvino_model"

    resolved_path = ROOT_DIR / weights_str if not Path(weights_str).is_absolute() else Path(weights_str)
    key = str(resolved_path if resolved_path.exists() else weights_str)
    warmup_key = f"{key}::{device or 'default'}"

    with _MODEL_LOCK:
        if key not in _MODEL_CACHE:
            logger.info("Inicializando modelo YOLO a partir de: %s", key)
            loaded = YOLO(key, task="detect")
            setattr(loaded, "_orig_weights_name", orig_weights_str)
            _MODEL_CACHE[key] = loaded

        model = _MODEL_CACHE[key]

        # Executa warmup único na compilação inicial para evitar pico de 600ms no 1º frame da webcam
        if warmup_key not in _WARMED_MODELS:
            _WARMED_MODELS.add(warmup_key)
            try:
                dummy = np.zeros((320, 320, 3), dtype=np.uint8)
                _run_raw_predict(
                    model=model,
                    frame=dummy,
                    confidence=0.5,
                    iou=0.5,
                    device=device or DEVICE,
                    imgsz=OPENVINO_IMAGE_SIZE,
                    augment=False,
                    max_det=10,
                )
            except Exception:
                pass

        return model


def _unwrap_result(result: Any) -> Any:
    if isinstance(result, (tuple, list)) and len(result) >= 1:
        return result[0]
    return result


def classify_scenario(count: int) -> tuple[str, str, tuple[int, int, int], str, str]:
    """
    Classifica o cenário com base na quantidade de pessoas detectadas.
    Retorna: (rotulo_curto, rotulo_completo, cor_bgr, cor_hex, nivel_alerta)
    """
    if count <= 0:
        return "Ambiente Livre", "AMBIENTE LIVRE (0 pessoas)", (160, 160, 160), "#94A3B8", "baixo"
    if count == 1:
        return "Individual", "INDIVIDUAL (1 pessoa)", (80, 215, 50), "#22C55E", "baixo"
    if count <= 4:
        return "Em Grupo", f"EM GRUPO ({count} pessoas)", (0, 200, 255), "#EAB308", "moderado"
    if count <= 12:
        return "Grupo Grande", f"GRUPO GRANDE ({count} pessoas)", (0, 140, 255), "#F97316", "alto"
    return "Aglomeração", f"AGLOMERACAO ({count} pessoas)", (50, 50, 255), "#EF4444", "critico"


def apply_preprocessing(frame: np.ndarray, enhance: bool = True) -> np.ndarray:
    """
    Aplica realce adaptativo de contraste (CLAHE no espaço LAB)
    para melhorar detecção em ambientes escuros, contra-luz ou neblina.
    """
    if not enhance or frame is None or frame.size == 0:
        return frame

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    mean_l = float(np.mean(l_channel))
    clip_limit = 2.8 if mean_l < 85.0 else 2.0

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    limg = cv2.merge((cl, a, b))
    enhanced_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_frame


def _run_raw_predict(
    model: YOLO,
    frame: np.ndarray,
    confidence: float,
    iou: float,
    device: str,
    imgsz: int,
    augment: bool,
    max_det: int,
) -> Any:
    """
    Executa `model.predict` com proteção automática contra erros de driver/kernel
    e incompatibilidades entre dispositivos OpenVINO e PyTorch.
    """
    is_openvino = _is_openvino_model(model)
    predict_device = str(device) if device else "cpu"

    if predict_device in _FAILED_DEVICES:
        predict_device = "intel:cpu" if is_openvino else "cpu"

    if not is_openvino and predict_device.startswith("intel:"):
        predict_device = "cpu"
    elif is_openvino and not predict_device.startswith("intel:"):
        predict_device = "intel:cpu"

    effective_imgsz = OPENVINO_IMAGE_SIZE if is_openvino else int(imgsz)
    effective_augment = False if is_openvino else bool(augment)

    try:
        with _MODEL_LOCK:
            results = model.predict(
                source=frame,
                conf=float(confidence),
                iou=float(iou),
                classes=[PERSON_CLASS_ID],
                imgsz=effective_imgsz,
                device=predict_device,
                verbose=False,
                augment=effective_augment,
                agnostic_nms=True,
                max_det=int(max_det),
            )
        return results[0]
    except Exception as exc:
        logger.warning(
            "Falha na inferência com dispositivo '%s' (%s). Aplicando fallback seguro.",
            predict_device,
            exc,
        )
        _FAILED_DEVICES.add(predict_device)
        fallback_dev = "intel:cpu" if is_openvino and predict_device != "intel:cpu" else "cpu"
        if is_openvino and fallback_dev == "cpu":
            pt_model = load_model(getattr(model, "_orig_weights_name", DEFAULT_MODEL_NAME), device="cpu", force_pytorch=True)
            with _MODEL_LOCK:
                results = pt_model.predict(
                    source=frame,
                    conf=float(confidence),
                    iou=float(iou),
                    classes=[PERSON_CLASS_ID],
                    imgsz=int(imgsz),
                    device="cpu",
                    verbose=False,
                    augment=bool(augment),
                    agnostic_nms=True,
                    max_det=int(max_det),
                )
            return results[0]

        with _MODEL_LOCK:
            results = model.predict(
                source=frame,
                conf=float(confidence),
                iou=float(iou),
                classes=[PERSON_CLASS_ID],
                imgsz=effective_imgsz,
                device=fallback_dev,
                verbose=False,
                augment=effective_augment,
                agnostic_nms=True,
                max_det=int(max_det),
            )
        return results[0]


def merge_boxes_wbf_nms(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.78,
    max_det: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combina caixas de múltiplas escalas/tiles usando Supressão Não-Máxima (NMS) com Fusão Ponderada (WBF)
    e Supressão de Contenção (IoA) para eliminar fragmentos de bordas de tiles em grupos/multidões.
    """
    if len(boxes_xyxy) == 0:
        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

    boxes = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    confs = np.asarray(scores, dtype=np.float32).reshape(-1)

    order = np.argsort(-confs)
    boxes = boxes[order]
    confs = confs[order]

    areas = np.maximum(1.0, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    kept_boxes: list[np.ndarray] = []
    kept_scores: list[float] = []
    suppressed = np.zeros(len(boxes), dtype=bool)

    for i in range(len(boxes)):
        if suppressed[i]:
            continue

        b_i = boxes[i]
        c_i = confs[i]
        a_i = areas[i]

        rem_idx = np.where(~suppressed[i + 1 :])[0] + (i + 1)
        if len(rem_idx) > 0:
            b_rem = boxes[rem_idx]
            a_rem = areas[rem_idx]

            xx1 = np.maximum(b_i[0], b_rem[:, 0])
            yy1 = np.maximum(b_i[1], b_rem[:, 1])
            xx2 = np.minimum(b_i[2], b_rem[:, 2])
            yy2 = np.minimum(b_i[3], b_rem[:, 3])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            union = a_i + a_rem - inter
            iou = inter / np.maximum(union, 1e-6)
            min_area = np.minimum(a_i, a_rem)
            ioa = inter / np.maximum(min_area, 1e-6)

            same_person_mask = iou >= iou_threshold
            if np.any(same_person_mask):
                cluster_indices = rem_idx[same_person_mask]
                cluster_boxes = np.vstack([b_i[None, :], boxes[cluster_indices]])
                cluster_weights = np.concatenate([[c_i], confs[cluster_indices]])
                fused_box = np.average(cluster_boxes, axis=0, weights=cluster_weights)
                boost = min(0.06, 0.02 * len(cluster_indices))
                c_i = min(0.99, float(c_i + boost))
                b_i = fused_box.astype(np.float32)

            suppress_mask = same_person_mask | (ioa >= containment_threshold)
            suppressed[rem_idx[suppress_mask]] = True

        kept_boxes.append(b_i)
        kept_scores.append(float(c_i))
        if len(kept_boxes) >= max_det:
            break

    return np.array(kept_boxes, dtype=np.float32), np.array(kept_scores, dtype=np.float32)


def detect_people(
    model: YOLO,
    frame: np.ndarray,
    confidence: float = CONFIDENCE,
    iou: float = IOU_THRESHOLD,
    device: str = DEVICE,
    augment: bool = False,
    max_det: int = MAX_DETECTIONS_DEFAULT,
    high_density: bool = False,
    fast_stream_mode: bool = False,
) -> Any:
    """
    Detecta pessoas no frame.
    Se `augment=True` ou `high_density=True`, aciona o motor aprimorado de alta densidade
    (Tiling + Fusão WBF/NMS). Quando `fast_stream_mode=True` (webcam/vídeo em tempo real),
    usa Tiling Horizontal 1x2 otimizado para manter alto FPS no notebook.
    """
    if augment or high_density:
        return detect_people_enhanced(
            model=model,
            frame=frame,
            confidence=confidence,
            iou=iou,
            device=device,
            augment=augment,
            max_det=max_det,
            use_tta=augment,
            multi_scale=True,
            fast_stream_mode=fast_stream_mode,
        )

    start_time = time.perf_counter()
    res = _run_raw_predict(
        model=model,
        frame=frame,
        confidence=confidence,
        iou=iou,
        device=device,
        imgsz=IMAGE_SIZE,
        augment=False,
        max_det=max_det,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000.0
    res.inference_latency_ms = latency_ms
    return res


def detect_people_enhanced(
    model: YOLO,
    frame: np.ndarray,
    confidence: float = CONFIDENCE,
    iou: float = IOU_THRESHOLD,
    device: str = DEVICE,
    augment: bool = False,
    max_det: int = MAX_DETECTIONS_DEFAULT,
    use_tta: bool = False,
    multi_scale: bool = True,
    fast_stream_mode: bool = False,
) -> Any:
    """
    Motor de Detecção de Alta Densidade (Multi-Scale + Sliced Tiling SAHI-like + Fusão WBF/NMS).
    - Em fotos estáticas (`fast_stream_mode=False`): executa 1 passe global + 4 tiles 2x2 sobrepostos.
    - Em streams de webcam/vídeo (`fast_stream_mode=True`): executa 1 passe global + 2 tiles horizontais
      focados na faixa central/fundo (40% mais leve para rodar fluido na iGPU Intel UHD).
    """
    augment = bool(augment or use_tta)
    if not multi_scale and not augment:
        return detect_people(model, frame, confidence, iou, device, augment=False, max_det=max_det)

    start_time = time.perf_counter()
    h, w = frame.shape[:2]
    is_openvino = _is_openvino_model(model)

    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []

    # 1. Inferência no Frame Completo
    base_res = _run_raw_predict(
        model=model,
        frame=frame,
        confidence=confidence,
        iou=min(0.60, iou + 0.05),
        device=device,
        imgsz=960 if not is_openvino else OPENVINO_IMAGE_SIZE,
        augment=augment and not is_openvino,
        max_det=max_det,
    )
    if base_res.boxes is not None and len(base_res.boxes) > 0:
        all_boxes.append(base_res.boxes.xyxy.cpu().numpy())
        all_scores.append(base_res.boxes.conf.cpu().numpy())

    # 2. Inferência em Tiles Sobrepostos
    if w >= 480 and h >= 360:
        overlap = 0.22
        tile_w = int(w * (0.5 + overlap * 0.5))
        x_offsets = [0, max(0, w - tile_w)]

        if fast_stream_mode:
            # Em tempo real, foca 2 tiles horizontais cobrindo a faixa de interesse (10% a 90% da altura)
            y0 = int(h * 0.08)
            tile_h = int(h * 0.84)
            y_offsets = [y0]
        else:
            tile_h = int(h * (0.5 + overlap * 0.5))
            y_offsets = [0, max(0, h - tile_h)]

        tile_conf = max(0.12, confidence * 0.88)

        for y0 in y_offsets:
            for x0 in x_offsets:
                tile = frame[y0 : y0 + tile_h, x0 : x0 + tile_w]
                if tile.size == 0:
                    continue
                t_res = _run_raw_predict(
                    model=model,
                    frame=tile,
                    confidence=tile_conf,
                    iou=min(0.60, iou + 0.05),
                    device=device,
                    imgsz=IMAGE_SIZE,
                    augment=False,
                    max_det=max_det,
                )
                if t_res.boxes is not None and len(t_res.boxes) > 0:
                    t_boxes = t_res.boxes.xyxy.cpu().numpy().copy()
                    t_confs = t_res.boxes.conf.cpu().numpy().copy()
                    t_boxes[:, [0, 2]] += float(x0)
                    t_boxes[:, [1, 3]] += float(y0)
                    all_boxes.append(t_boxes)
                    all_scores.append(t_confs)

    # 3. Fusão WBF + NMS + Supressão de Contenção
    device_tensor = (
        base_res.boxes.data.device
        if (base_res.boxes is not None and hasattr(base_res.boxes, "data"))
        else torch.device("cpu")
    )

    if all_boxes:
        cat_boxes = np.vstack(all_boxes)
        cat_scores = np.concatenate(all_scores)
        merged_boxes, merged_scores = merge_boxes_wbf_nms(
            cat_boxes,
            cat_scores,
            iou_threshold=iou,
            containment_threshold=0.78,
            max_det=max_det,
        )
        keep = merged_scores >= confidence
        merged_boxes = merged_boxes[keep]
        merged_scores = merged_scores[keep]

        if len(merged_boxes) > 0:
            cls_col = np.full((len(merged_boxes), 1), float(PERSON_CLASS_ID), dtype=np.float32)
            det_data = np.hstack([merged_boxes, merged_scores[:, None], cls_col])
            base_res.update(boxes=torch.tensor(det_data, dtype=torch.float32, device=device_tensor))
        else:
            base_res.update(boxes=torch.empty((0, 6), dtype=torch.float32, device=device_tensor))
    else:
        base_res.update(boxes=torch.empty((0, 6), dtype=torch.float32, device=device_tensor))

    latency_ms = (time.perf_counter() - start_time) * 1000.0
    base_res.inference_latency_ms = latency_ms
    return base_res


def count_people(result: Any) -> int:
    """Retorna o total de pessoas detectadas no resultado."""
    res = _unwrap_result(result)
    if res is None or getattr(res, "boxes", None) is None:
        return 0
    return len(res.boxes)


def extract_detections_info(result: Any) -> list[dict[str, Any]]:
    """Extrai lista estruturada de dicionários com coordenadas, área e confiança de cada pessoa."""
    res = _unwrap_result(result)
    if res is None or getattr(res, "boxes", None) is None:
        return []
    detections: list[dict[str, Any]] = []
    boxes = res.boxes
    if len(boxes) == 0:
        return detections
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i].astype(int)
        conf = float(confs[i])
        w_box = max(0, int(x2 - x1))
        h_box = max(0, int(y2 - y1))
        detections.append({
            "id": i + 1,
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
            "largura_px": w_box,
            "altura_px": h_box,
            "confidence": round(conf, 4),
            "class_name": "pessoa",
        })
    return detections


def compute_crowd_metrics(result: Any) -> dict[str, Any]:
    """Calcula métricas analíticas completas da cena (ocupação, cenário, densidade espacial, confiança)."""
    res = _unwrap_result(result)
    count = count_people(res)
    short_lbl, full_lbl, bgr_color, hex_color, risk = classify_scenario(count)
    latency_ms = float(getattr(res, "inference_latency_ms", 0.0))

    if count == 0 or getattr(res, "boxes", None) is None:
        return {
            "count": 0,
            "avg_confidence": 0.0,
            "min_confidence": 0.0,
            "max_confidence": 0.0,
            "area_coverage_pct": 0.0,
            "scenario_short": short_lbl,
            "scenario_full": full_lbl,
            "color_bgr": bgr_color,
            "color_hex": hex_color,
            "risk_level": risk,
            "latency_ms": round(latency_ms, 1),
        }

    boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
    confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)

    area_pct = 0.0
    if hasattr(res, "orig_img") and res.orig_img is not None:
        h, w = res.orig_img.shape[:2]
        frame_area = float(max(1, h * w))
        mask = np.zeros((h, w), dtype=np.uint8)
        for b in boxes.astype(int):
            x1, y1 = max(0, b[0]), max(0, b[1])
            x2, y2 = min(w, b[2]), min(h, b[3])
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = 1
        area_pct = round((float(np.count_nonzero(mask)) / frame_area) * 100.0, 1)

    return {
        "count": count,
        "avg_confidence": round(float(np.mean(confs)), 4),
        "min_confidence": round(float(np.min(confs)), 4),
        "max_confidence": round(float(np.max(confs)), 4),
        "area_coverage_pct": area_pct,
        "scenario_short": short_lbl,
        "scenario_full": full_lbl,
        "color_bgr": bgr_color,
        "color_hex": hex_color,
        "risk_level": risk,
        "latency_ms": round(latency_ms, 1),
    }


def blur_detected_faces(
    frame: np.ndarray,
    blur_kernel_size: int = 31,
    boxes: np.ndarray | list[Any] | None = None,
    strict_lgpd: bool = True,
) -> np.ndarray:
    """
    Anonimiza rostos no frame em conformidade com a LGPD, calibrado tanto para
    Close-Ups/Retratos de Webcam (onde o rosto ocupa o centro da caixa) quanto para
    Pessoas de Corpo Inteiro/Distantes:
      1. Redimensiona dinamicamente a ROI para no máximo 240px de largura antes do Haar Cascade,
         executando a detecção facial em < 1.5ms e encontrando rostos grandes de webcam com precisão.
      2. Se o Haar Cascade encontrar o rosto, expande suavemente a margem (15%) para cobrir da testa ao queixo.
      3. Se `strict_lgpd=True` e o rosto estiver inclinado/perfil, calcula a posição anatômica do rosto
         diferenciando automaticamente entre Close-Up de Webcam (busto/retrato) e Corpo Inteiro.
    """
    if frame is None or frame.size == 0:
        return frame

    cascade = get_face_cascade()
    output = frame.copy()
    height, width = output.shape[:2]
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) if cascade is not None and not cascade.empty() else None

    def _apply_adaptive_blur(x1: int, y1: int, x2: int, y2: int) -> None:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 <= x1 or y2 <= y1:
            return
        roi = output[y1:y2, x1:x2]
        if roi.size == 0:
            return
        reg_dim = max(x2 - x1, y2 - y1)
        k = max(21, min(111, (reg_dim // 2) | 1))
        if blur_kernel_size > 31:
            k = max(k, blur_kernel_size | 1)
        output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            bx1, by1, bx2, by2 = [int(v) for v in box[:4]]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(width, bx2), min(height, by2)
            bw = bx2 - bx1
            bh = by2 - by1
            if bw < 8 or bh < 12:
                continue

            # Detecta se a caixa é um Close-Up/Busto de Webcam (apenas cabeça e ombros) ou Corpo Inteiro
            aspect_ratio = float(bw) / float(max(1, bh))
            is_closeup_bust = (aspect_ratio >= 0.56) or (bh >= height * 0.48 and by2 >= height - 25)

            # Em close-up de webcam o rosto vai até ~85% da altura da caixa; em corpo inteiro fica nos 50% superiores
            search_ratio = 0.88 if is_closeup_bust else 0.52
            head_y2 = min(by2, by1 + max(20, int(bh * search_ratio)))
            face_found = False

            if gray is not None and (head_y2 - by1) >= 20 and bw >= 20:
                roi_gray = gray[by1:head_y2, bx1:bx2]
                rh, rw = roi_gray.shape[:2]

                # Downscale rápido para máx 220px de largura: acelera o Haar Cascade em 5x e melhora detecção de rostos grandes
                scale = min(1.0, 220.0 / float(max(1, rw)))
                if scale < 1.0:
                    small_w = max(24, int(rw * scale))
                    small_h = max(24, int(rh * scale))
                    proc_gray = cv2.resize(roi_gray, (small_w, small_h), interpolation=cv2.INTER_AREA)
                else:
                    proc_gray = roi_gray

                min_face_dim = max(18, int(min(proc_gray.shape[:2]) * 0.22))
                faces = cascade.detectMultiScale(
                    proc_gray,
                    scaleFactor=1.10,
                    minNeighbors=3,
                    minSize=(min_face_dim, min_face_dim),
                )
                inv_scale = 1.0 / scale
                for fx, fy, fw, fh in faces:
                    real_fx = int(fx * inv_scale)
                    real_fy = int(fy * inv_scale)
                    real_fw = int(fw * inv_scale)
                    real_fh = int(fh * inv_scale)
                    # Expande levemente (12% X, 18% Y) para cobrir todo o rosto (testa ao queixo)
                    pad_x = int(real_fw * 0.12)
                    pad_y = int(real_fh * 0.18)
                    _apply_adaptive_blur(
                        bx1 + real_fx - pad_x,
                        by1 + real_fy - pad_y,
                        bx1 + real_fx + real_fw + pad_x,
                        by1 + real_fy + real_fh + pad_y,
                    )
                    face_found = True

            # Fallback anatômico LGPD calibrado conforme o enquadramento (Close-Up vs Corpo Inteiro)
            if not face_found and strict_lgpd:
                cx = (bx1 + bx2) // 2
                if is_closeup_bust:
                    # Em close-up de webcam (cabeça e ombros), o rosto fica exatamente no centro da caixa (15% a 78% da altura)
                    est_w = max(16, int(bw * 0.56))
                    hy1 = by1 + int(bh * 0.14)
                    hy2 = by1 + int(bh * 0.78)
                else:
                    # Em corpo inteiro / média distância, a cabeça fica no topo (3% a 28% da altura)
                    est_w = max(10, int(bw * 0.54))
                    hy1 = by1 + int(bh * 0.03)
                    hy2 = by1 + int(bh * 0.28)

                hx1 = cx - est_w // 2
                hx2 = cx + est_w // 2
                _apply_adaptive_blur(hx1, hy1, hx2, min(by2, hy2))
    elif gray is not None:
        faces = cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(24, 24))
        for fx, fy, fw, fh in faces:
            _apply_adaptive_blur(fx, fy, fx + fw, fy + fh)

    return output


def annotate_frame(
    result: Any,
    blur_faces: bool = False,
    show_labels: bool = True,
    show_conf: bool = True,
    thickness: int = 2,
    tracks: list[TrackedPerson] | None = None,
    heatmap: HeatmapAccumulator | bool | None = None,
    show_trails: bool = False,
    color_by_scenario: bool = False,
) -> np.ndarray:
    """
    Renderiza anotações visuais limpas e profissionais sobre o frame:
    - Desfoque de privacidade LGPD (aplicado antes das caixas e textos)
    - Mapa de calor espacial (opcional)
    - Caixas delimitadoras com cantos HUD codificadas por confiança ou por cenário (Modo Feira)
    - Sem linhas/pontos sobre o rosto dos usuários
    """
    res = _unwrap_result(result)

    if hasattr(res, "orig_img") and res.orig_img is not None:
        annotated = res.orig_img.copy()
    else:
        annotated = np.zeros((720, 1280, 3), dtype=np.uint8)

    has_boxes = getattr(res, "boxes", None) is not None and len(res.boxes) > 0
    if has_boxes:
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes.xyxy, "cpu") else np.array(res.boxes.xyxy)
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes.conf, "cpu") else np.array(res.boxes.conf)
    else:
        boxes = np.empty((0, 4), dtype=np.float32)
        confs = np.empty((0,), dtype=np.float32)

    count = len(boxes)
    short_scen, full_scen, scen_color, _, _ = classify_scenario(count)

    # 1. Aplica desfoque facial ANTES de desenhar caixas e textos
    if blur_faces:
        annotated = blur_detected_faces(annotated, boxes=boxes if count > 0 else None)

    # 2. Aplica Mapa de Calor (se ativado)
    if isinstance(heatmap, HeatmapAccumulator):
        annotated = heatmap.render_overlay(annotated)
    elif heatmap is True and count > 0:
        annotated = render_instant_heatmap(annotated, boxes)

    h_img, w_img = annotated.shape[:2]

    # 3. Associa IDs de rastreamento às caixas (sem desenhar linhas azuis sobre o rosto)
    track_id_by_box_idx: dict[int, int] = {}
    if tracks and count > 0:
        for trk in tracks:
            t_box = trk.bbox
            dists = np.linalg.norm(boxes - t_box[None, :], axis=1)
            best_i = int(np.argmin(dists))
            if dists[best_i] < max(w_img, h_img) * 0.15:
                track_id_by_box_idx[best_i] = trk.track_id

    # 4. Desenha Caixas e Rótulos
    for i in range(count):
        x1, y1, x2, y2 = boxes[i].astype(int)
        conf = float(confs[i])

        x1 = max(0, min(w_img - 1, x1))
        y1 = max(0, min(h_img - 1, y1))
        x2 = max(0, min(w_img - 1, x2))
        y2 = max(0, min(h_img - 1, y2))

        if color_by_scenario:
            color = scen_color
        elif conf >= 0.65:
            color = (60, 235, 90)
        elif conf >= 0.40:
            color = (0, 215, 255)
        else:
            color = (60, 90, 255)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

        corner_len = max(6, min(16, (x2 - x1) // 5, (y2 - y1) // 5))
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thickness + 1)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thickness + 1)
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thickness + 1)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thickness + 1)

        if show_labels or show_conf:
            tid = track_id_by_box_idx.get(i)
            parts = []
            if show_labels:
                parts.append(f"pessoa #{tid}" if tid is not None else "pessoa")
            elif tid is not None:
                parts.append(f"#{tid}")
            if show_conf:
                parts.append(f"{conf * 100:.0f}%")
            label = " ".join(parts).strip()

            if label:
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y1 = max(0, y1 - th - 8)
                label_y2 = y1
                if y1 - th - 8 < 0:
                    label_y1 = y1
                    label_y2 = min(h_img, y1 + th + 8)
                cv2.rectangle(annotated, (x1, label_y1), (min(w_img, x1 + tw + 8), label_y2), color, -1)
                text_y = max(12, label_y2 - 4 if y1 - th - 8 < 0 else y1 - 4)
                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (15, 20, 25),
                    1,
                    cv2.LINE_AA,
                )

    # 5. Badge de Status e Cenário no canto superior esquerdo
    badge_text = f"Total: {count} | {full_scen if color_by_scenario else short_scen}"
    (bw_txt, _), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    badge_w = max(175, bw_txt + 28)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (min(w_img - 8, 8 + badge_w), 44), (15, 23, 42), -1)
    cv2.addWeighted(overlay, 0.72, annotated, 0.28, 0, annotated)
    cv2.rectangle(annotated, (8, 8), (13, 44), scen_color, -1)
    cv2.putText(
        annotated,
        badge_text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (248, 250, 252),
        2,
        cv2.LINE_AA,
    )

    return annotated


def register_detection(
    source: str | Path,
    input_type: str,
    count: int,
    confidence: float,
    device: str = DEVICE,
    unique_people: int | None = None,
    avg_confidence: float = 0.0,
    latency_ms: float = 0.0,
    model_name: str = DEFAULT_MODEL_NAME,
    cooldown_seconds: float = 8.0,
) -> bool:
    """Registra a análise de forma thread-safe no banco SQLite e no arquivo CSV sincronizado."""
    repo = get_repository()
    return repo.register(
        source=source,
        input_type=input_type,
        count=count,
        confidence=confidence,
        device=device,
        unique_people=unique_people,
        avg_confidence=avg_confidence,
        latency_ms=latency_ms,
        model_name=model_name,
        cooldown_seconds=cooldown_seconds,
    )


def clear_detection_records() -> None:
    """Limpa todos os registros do histórico."""
    get_repository().clear_all()


def save_snapshot(frame: np.ndarray, prefix: str = "webcam") -> Path:
    """Salva um frame anotado em disco no diretório `logs/snapshots/`."""
    ensure_project_directories()
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
    filename = f"{prefix}_{now_str}.jpg"
    target = SNAPSHOTS_DIR / filename
    cv2.imwrite(str(target), frame)
    return target


def process_video(
    model: YOLO,
    video_path: str | Path,
    confidence: float = CONFIDENCE,
    iou: float = IOU_THRESHOLD,
    blur_faces: bool = False,
    source_name: str | None = None,
    device: str = DEVICE,
    frame_step: int = 1,
    low_light_prep: bool = False,
    high_density: bool = False,
    enable_tracking: bool = True,
    enable_heatmap: bool = False,
    color_by_scenario: bool = False,
    max_det: int = MAX_DETECTIONS_DEFAULT,
    export_video_path: str | Path | None = None,
) -> Generator[tuple[np.ndarray, int, int, int, float, dict[str, Any]], None, None]:
    """
    Processa um arquivo de vídeo frame a frame com rastreamento de pessoas únicas,
    mapa de calor opcional, realce Low-Light e exportação `.mp4`.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

    maximum = 0
    frame_idx = 0
    fps_ema = 0.0
    alpha = 0.15
    conf_accumulator: list[float] = []
    latency_accumulator: list[float] = []

    tracker = PersonTracker(min_hits_to_confirm=2 if frame_step == 1 else 1) if enable_tracking else None
    heatmap_acc = HeatmapAccumulator() if enable_heatmap else None
    video_writer: cv2.VideoWriter | None = None

    if export_video_path is not None:
        ensure_project_directories()
        out_fps = max(1.0, video_fps / max(1, frame_step))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(export_video_path), fourcc, out_fps, (width, height))

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if frame_step > 1 and (frame_idx % frame_step != 0):
                continue

            start_time = time.perf_counter()
            proc_frame = apply_preprocessing(frame, enhance=True) if low_light_prep else frame

            result = detect_people(
                model=model,
                frame=proc_frame,
                confidence=confidence,
                iou=iou,
                device=device,
                high_density=high_density,
                max_det=max_det,
                fast_stream_mode=True,
            )
            result.orig_img = frame

            count = count_people(result)
            maximum = max(maximum, count)
            lat_ms = float(getattr(result, "inference_latency_ms", 0.0))
            latency_accumulator.append(lat_ms)

            has_boxes = getattr(result, "boxes", None) is not None and len(result.boxes) > 0
            boxes_np = result.boxes.xyxy.cpu().numpy() if has_boxes else np.empty((0, 4), dtype=np.float32)
            confs_np = result.boxes.conf.cpu().numpy() if has_boxes else np.empty((0,), dtype=np.float32)

            if len(confs_np) > 0:
                conf_accumulator.append(float(np.mean(confs_np)))

            active_tracks = None
            if tracker is not None:
                video_timestamp = float(frame_idx) / max(1.0, video_fps)
                active_tracks = tracker.update(boxes_np, confs_np, timestamp=video_timestamp)

            if heatmap_acc is not None:
                heatmap_acc.update(frame.shape, boxes_np)

            annotated = annotate_frame(
                result,
                blur_faces=blur_faces,
                tracks=active_tracks,
                heatmap=heatmap_acc,
                color_by_scenario=color_by_scenario,
            )

            process_time = time.perf_counter() - start_time
            current_fps = 1.0 / max(process_time, 1e-6)
            fps_ema = current_fps if fps_ema == 0.0 else (alpha * current_fps + (1.0 - alpha) * fps_ema)

            unique_total = tracker.unique_people_count if tracker is not None else maximum
            hud_extra = f"FPS IA: {fps_ema:.1f} | Pico: {maximum} | Unicas: {unique_total}"
            cv2.putText(
                annotated,
                hud_extra,
                (10, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (56, 189, 248),
                2,
                cv2.LINE_AA,
            )

            if video_writer is not None:
                video_writer.write(annotated)

            extra_stats = {
                "unique_people": unique_total,
                "stabilized_count": tracker.stabilized_count if tracker is not None else count,
                "avg_dwell_time_s": tracker.avg_dwell_time if tracker is not None else 0.0,
                "processing_fps": round(fps_ema, 1),
                "latency_ms": round(lat_ms, 1),
                "scenario": classify_scenario_label(count),
            }

            yield annotated, count, frame_idx, total_frames, video_fps, extra_stats
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()

    final_unique = tracker.unique_people_count if tracker is not None else maximum
    avg_conf_video = float(np.mean(conf_accumulator)) if conf_accumulator else 0.0
    avg_lat_video = float(np.mean(latency_accumulator)) if latency_accumulator else 0.0
    register_detection(
        source=source_name or video_path,
        input_type="video",
        count=maximum,
        confidence=confidence,
        device=device,
        unique_people=final_unique,
        avg_confidence=avg_conf_video,
        latency_ms=avg_lat_video,
        cooldown_seconds=2.0,
    )


def load_records_df() -> pd.DataFrame:
    """Carrega todos os registros de detecção armazenados em um DataFrame Pandas."""
    return get_repository().load_dataframe()

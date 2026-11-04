"""
Módulo de Rastreamento Multi-Objeto (MOT), Estabilização Temporal e Mapa de Calor.
Projetado para contagem de pessoas únicas, cálculo de tempo de permanência (dwell time),
suavização de contagem (anti-flicker) e análise espacial de densidade.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import time
from typing import Any

import cv2
import numpy as np


@dataclass
class TrackedPerson:
    """Representa uma pessoa rastreada ao longo do tempo."""

    track_id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    first_seen: float = field(default_factory=time.perf_counter)
    last_seen: float = field(default_factory=time.perf_counter)
    hits: int = 1
    age: int = 0
    missed_frames: int = 0
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    trajectory: deque = field(default_factory=lambda: deque(maxlen=35))

    def __post_init__(self) -> None:
        cx, cy = self.center
        self.trajectory.append((int(cx), int(cy)))

    @property
    def center(self) -> tuple[float, float]:
        return (
            float(self.bbox[0] + self.bbox[2]) * 0.5,
            float(self.bbox[1] + self.bbox[3]) * 0.5,
        )

    @property
    def foot_point(self) -> tuple[int, int]:
        """Ponto inferior central (pés da pessoa), ideal para mapas de calor e zonas."""
        return (
            int((self.bbox[0] + self.bbox[2]) * 0.5),
            int(self.bbox[3]),
        )

    @property
    def dwell_time_seconds(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def predict_next_bbox(self) -> np.ndarray:
        """Prediz a posição da caixa no próximo frame usando velocidade linear amortecida."""
        if self.missed_frames == 0:
            return self.bbox.copy()
        dx, dy = self.velocity * float(self.missed_frames)
        pred = self.bbox.copy().astype(np.float32)
        pred[0] += dx
        pred[2] += dx
        pred[1] += dy
        pred[3] += dy
        return pred

    def update(self, new_bbox: np.ndarray, confidence: float, now: float) -> None:
        old_cx, old_cy = self.center
        new_cx = float(new_bbox[0] + new_bbox[2]) * 0.5
        new_cy = float(new_bbox[1] + new_bbox[3]) * 0.5

        inst_vel = np.array([new_cx - old_cx, new_cy - old_cy], dtype=np.float32)
        self.velocity = 0.6 * self.velocity + 0.4 * inst_vel

        # Suavização espacial da bounding box para reduzir trepidação visual (jitter)
        self.bbox = 0.25 * self.bbox + 0.75 * new_bbox.astype(np.float32)
        self.confidence = float(confidence)
        self.last_seen = now
        self.hits += 1
        self.missed_frames = 0
        self.trajectory.append((int(new_cx), int(new_cy)))


def compute_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Calcula a matriz de IoU (Intersection over Union) entre dois conjuntos de caixas [N, 4] e [M, 4]."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    area_a = np.maximum(1e-6, (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    area_b = np.maximum(1e-6, (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))

    union_area = area_a[:, None] + area_b[None, :] - inter_area
    return inter_area / np.maximum(union_area, 1e-6)


class PersonTracker:
    """
    Rastreador leve e robusto baseado em IoU + Distância de Centroide com predição de velocidade.
    Funciona independente do backend do modelo (PyTorch, OpenVINO, ONNX) e suporta pulo de frames.
    """

    def __init__(
        self,
        iou_threshold: float = 0.25,
        max_missed_frames: int = 18,
        min_hits_to_confirm: int = 2,
        ema_alpha: float = 0.25,
    ) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_hits_to_confirm = min_hits_to_confirm
        self.ema_alpha = ema_alpha

        self.tracks: dict[int, TrackedPerson] = {}
        self.next_id: int = 1
        self.confirmed_unique_ids: set[int] = set()
        self.completed_dwell_times: list[float] = []
        self.peak_count: int = 0
        self.smoothed_count_val: float = 0.0
        self._recent_counts: deque[int] = deque(maxlen=7)

    def reset(self) -> None:
        """Reinicia todo o estado do rastreador."""
        self.tracks.clear()
        self.next_id = 1
        self.confirmed_unique_ids.clear()
        self.completed_dwell_times.clear()
        self.peak_count = 0
        self.smoothed_count_val = 0.0
        self._recent_counts.clear()

    def update(
        self,
        boxes: np.ndarray,
        confidences: np.ndarray,
        timestamp: float | None = None,
    ) -> list[TrackedPerson]:
        """
        Atualiza o rastreador com as detecções do frame atual.
        Retorna a lista de objetos TrackedPerson ativos no frame.
        """
        now = timestamp if timestamp is not None else time.perf_counter()
        raw_count = len(boxes)

        self._recent_counts.append(raw_count)
        if self.smoothed_count_val == 0.0 and raw_count > 0:
            self.smoothed_count_val = float(raw_count)
        else:
            self.smoothed_count_val = (
                self.ema_alpha * float(raw_count) + (1.0 - self.ema_alpha) * self.smoothed_count_val
            )
        self.peak_count = max(self.peak_count, raw_count)

        if len(boxes) == 0:
            to_delete = []
            for tid, trk in self.tracks.items():
                trk.missed_frames += 1
                trk.age += 1
                if trk.missed_frames > self.max_missed_frames:
                    if trk.track_id in self.confirmed_unique_ids and trk.dwell_time_seconds >= 0.5:
                        self.completed_dwell_times.append(trk.dwell_time_seconds)
                    to_delete.append(tid)
            for tid in to_delete:
                del self.tracks[tid]
            return []

        boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        confs_arr = np.asarray(confidences, dtype=np.float32).reshape(-1)

        active_ids = list(self.tracks.keys())
        if not active_ids:
            for box, conf in zip(boxes_arr, confs_arr):
                self._create_track(box, float(conf), now)
            return self.get_visible_tracks()

        pred_boxes = np.array([self.tracks[tid].predict_next_bbox() for tid in active_ids], dtype=np.float32)
        iou_mat = compute_iou_matrix(pred_boxes, boxes_arr)

        # Combina IoU com proximidade de centroide normalizada para recuperar movimentos rápidos
        pred_centers = np.column_stack([
            (pred_boxes[:, 0] + pred_boxes[:, 2]) * 0.5,
            (pred_boxes[:, 1] + pred_boxes[:, 3]) * 0.5,
        ])
        det_centers = np.column_stack([
            (boxes_arr[:, 0] + boxes_arr[:, 2]) * 0.5,
            (boxes_arr[:, 1] + boxes_arr[:, 3]) * 0.5,
        ])
        pred_diags = np.maximum(
            20.0,
            np.hypot(pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]),
        )
        dist_mat = np.linalg.norm(pred_centers[:, None, :] - det_centers[None, :, :], axis=2)
        norm_dist = dist_mat / pred_diags[:, None]

        # Score combinado: IoU + bônus de proximidade quando dentro de 1.2x a diagonal da caixa
        proximity_bonus = np.where(norm_dist < 1.2, (1.2 - norm_dist) * 0.25, 0.0)
        score_mat = iou_mat + proximity_bonus

        matched_track_indices: set[int] = set()
        matched_det_indices: set[int] = set()

        # Associação gulosa ordenada pelo maior score de compatibilidade
        flat_indices = np.argsort(-score_mat, axis=None)
        num_tracks, num_dets = score_mat.shape

        for flat_idx in flat_indices:
            t_idx = int(flat_idx // num_dets)
            d_idx = int(flat_idx % num_dets)
            if t_idx in matched_track_indices or d_idx in matched_det_indices:
                continue
            if iou_mat[t_idx, d_idx] < self.iou_threshold and norm_dist[t_idx, d_idx] > 0.75:
                continue

            tid = active_ids[t_idx]
            trk = self.tracks[tid]
            trk.update(boxes_arr[d_idx], float(confs_arr[d_idx]), now)
            trk.age += 1
            if trk.hits >= self.min_hits_to_confirm:
                self.confirmed_unique_ids.add(tid)

            matched_track_indices.add(t_idx)
            matched_det_indices.add(d_idx)

        # Trata tracks não associados neste frame
        to_delete = []
        for t_idx, tid in enumerate(active_ids):
            if t_idx not in matched_track_indices:
                trk = self.tracks[tid]
                trk.missed_frames += 1
                trk.age += 1
                if trk.missed_frames > self.max_missed_frames:
                    if trk.track_id in self.confirmed_unique_ids and trk.dwell_time_seconds >= 0.5:
                        self.completed_dwell_times.append(trk.dwell_time_seconds)
                    to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]

        # Cria novos tracks para detecções não associadas
        for d_idx in range(num_dets):
            if d_idx not in matched_det_indices:
                self._create_track(boxes_arr[d_idx], float(confs_arr[d_idx]), now)

        return self.get_visible_tracks()

    def _create_track(self, box: np.ndarray, confidence: float, now: float) -> TrackedPerson:
        tid = self.next_id
        self.next_id += 1
        trk = TrackedPerson(
            track_id=tid,
            bbox=box.astype(np.float32),
            confidence=confidence,
            first_seen=now,
            last_seen=now,
        )
        if self.min_hits_to_confirm <= 1:
            self.confirmed_unique_ids.add(tid)
        self.tracks[tid] = trk
        return trk

    def get_visible_tracks(self) -> list[TrackedPerson]:
        """Retorna as pessoas detectadas no frame atual (missed_frames == 0)."""
        return [t for t in self.tracks.values() if t.missed_frames == 0]

    @property
    def unique_people_count(self) -> int:
        """Total acumulado de pessoas únicas confirmadas na sessão."""
        # Se a sessão teve apenas 1 frame ou poucos frames ainda não confirmados, garante ao menos o pico
        return max(len(self.confirmed_unique_ids), self.peak_count if len(self.tracks) > 0 and not self.confirmed_unique_ids else len(self.confirmed_unique_ids))

    @property
    def stabilized_count(self) -> int:
        """Contagem estabilizada por mediana de janela curta para evitar oscilação de 1 frame."""
        if not self._recent_counts:
            return 0
        return int(round(float(np.median(self._recent_counts))))

    @property
    def avg_dwell_time(self) -> float:
        """Tempo médio de permanência (em segundos) das pessoas rastreadas."""
        active_dwells = [
            t.dwell_time_seconds
            for t in self.tracks.values()
            if t.track_id in self.confirmed_unique_ids and t.dwell_time_seconds >= 0.5
        ]
        all_dwells = self.completed_dwell_times + active_dwells
        if not all_dwells:
            return 0.0
        return float(np.mean(all_dwells))

    def get_metrics_summary(self) -> dict[str, Any]:
        return {
            "unique_people": self.unique_people_count,
            "stabilized_count": self.stabilized_count,
            "smoothed_ema": round(self.smoothed_count_val, 2),
            "peak_count": self.peak_count,
            "avg_dwell_time_s": round(self.avg_dwell_time, 1),
            "active_tracks": len(self.get_visible_tracks()),
        }


class HeatmapAccumulator:
    """
    Acumulador de Mapa de Calor Espacial (Densidade de Ocupação).
    Gera visualização térmica de onde as pessoas permanecem ou transitam com maior frequência.
    """

    def __init__(self, decay: float = 0.97, radius_ratio: float = 0.06) -> None:
        self.decay = decay
        self.radius_ratio = radius_ratio
        self._grid: np.ndarray | None = None
        self._shape: tuple[int, int] | None = None

    def reset(self) -> None:
        self._grid = None
        self._shape = None

    def update(self, frame_shape: tuple[int, ...], boxes: np.ndarray) -> None:
        h, w = frame_shape[:2]
        # Usa resolução interna reduzida (1/4) para máxima performance em tempo real
        gh, gw = max(32, h // 4), max(32, w // 4)

        if self._grid is None or self._shape != (gh, gw):
            self._grid = np.zeros((gh, gw), dtype=np.float32)
            self._shape = (gh, gw)
        else:
            self._grid *= self.decay

        if boxes is None or len(boxes) == 0:
            return

        scale_x = gw / float(w)
        scale_y = gh / float(h)
        base_radius = max(4, int(min(gh, gw) * self.radius_ratio))

        stamp = np.zeros((gh, gw), dtype=np.float32)
        for box in boxes:
            x1, y1, x2, y2 = box[:4]
            cx = int(((x1 + x2) * 0.5) * scale_x)
            # Foca no centro-inferior da pessoa (tronco/pés) para representar ocupação no piso
            cy = int((y1 * 0.35 + y2 * 0.65) * scale_y)
            cx = max(0, min(gw - 1, cx))
            cy = max(0, min(gh - 1, cy))

            box_r = max(base_radius, int(((x2 - x1) * scale_x) * 0.45))
            cv2.circle(stamp, (cx, cy), box_r, 1.0, -1)

        ksize = (base_radius * 2) | 1
        stamp = cv2.GaussianBlur(stamp, (ksize, ksize), 0)
        self._grid = np.clip(self._grid + stamp * 0.35, 0.0, 5.0)

    def render_overlay(self, frame: np.ndarray, alpha: float = 0.42) -> np.ndarray:
        """Aplica o mapa de calor sobre o frame original com máscara de transparência suave."""
        if self._grid is None or float(self._grid.max()) < 1e-3:
            return frame

        h, w = frame.shape[:2]
        max_val = max(1.0, float(np.percentile(self._grid, 99.0)), float(self._grid.max()))
        norm = np.clip(self._grid / max_val, 0.0, 1.0)
        norm_u8 = (norm * 255.0).astype(np.uint8)
        norm_resized = cv2.resize(norm_u8, (w, h), interpolation=cv2.INTER_LINEAR)

        colormap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        heat_color = cv2.applyColorMap(norm_resized, colormap)

        # Aplica apenas nas regiões onde há calor significativo (> 12/255)
        mask = norm_resized > 12
        if not np.any(mask):
            return frame

        output = frame.copy()
        blended = cv2.addWeighted(frame, 1.0 - alpha, heat_color, alpha, 0)
        output[mask] = blended[mask]
        return output


def render_instant_heatmap(frame: np.ndarray, boxes: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Gera um mapa de calor instantâneo para uma única imagem estática."""
    acc = HeatmapAccumulator(decay=1.0, radius_ratio=0.07)
    # Acumula com intensidade cheia para fotografia única
    acc.update(frame.shape, boxes)
    if acc._grid is not None:
        acc._grid *= 3.0
    return acc.render_overlay(frame, alpha=alpha)


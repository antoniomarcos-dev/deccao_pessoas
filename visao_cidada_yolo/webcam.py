"""
Módulo Profissional de Detecção, Rastreamento e Contagem em Tempo Real via Webcam.
Executa com captura assíncrona de baixa latência (ThreadedCamera), Rastreamento Multi-Objeto (MOT),
Mapa de Calor Térmico e overlay HUD (Head-Up Display) de alta tecnologia.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from config import CONFIDENCE, DEFAULT_MODEL_NAME, DEVICE, IOU_THRESHOLD
from inference import (
    annotate_frame,
    apply_preprocessing,
    classify_scenario,
    count_people,
    detect_people,
    load_model,
    register_detection,
    save_snapshot,
)
from utils.camera import ThreadedCamera
from utils.logger import get_logger
from utils.tracker import HeatmapAccumulator, PersonTracker


def draw_hud(
    frame: np.ndarray,
    count: int,
    unique_count: int,
    peak_count: int,
    fps: float,
    latency_ms: float,
    confidence: float,
    blur_faces: bool,
    high_density: bool,
    tracking_on: bool,
    heatmap_on: bool,
    low_light_on: bool,
    is_paused: bool,
) -> np.ndarray:
    """Desenha painel HUD profissional com métricas em tempo real e status dos motores sobre o frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Barra superior e barra inferior translúcidas
    cv2.rectangle(overlay, (0, 0), (w, 58), (15, 20, 28), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (12, 15, 22), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)

    short_scen, _, scen_color, _, _ = classify_scenario(count)

    # 1. Pessoas Agora + Cenário
    count_color = (0, 235, 120) if count > 0 else (180, 180, 180)
    cv2.putText(frame, f"AGORA: {count}", (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.80, count_color, 2, cv2.LINE_AA)

    # 2. Pico & Pessoas Únicas
    cv2.putText(
        frame,
        f"PICO: {peak_count} | UNICAS: {unique_count}",
        (185, 36),
        cv2.FONT_HERSHEY_DUPLEX,
        0.65,
        (255, 205, 85),
        2,
        cv2.LINE_AA,
    )

    # 3. Cenário Classificado
    cv2.putText(
        frame,
        f"[{short_scen.upper()}]",
        (500, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        scen_color,
        2,
        cv2.LINE_AA,
    )

    # 4. FPS, Latência e Confiança
    fps_color = (0, 220, 255) if fps >= 15 else (0, 140, 255)
    perf_str = f"{fps:.1f} FPS | {latency_ms:.0f}ms | Conf:{confidence:.2f}"
    cv2.putText(frame, perf_str, (705, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fps_color, 2, cv2.LINE_AA)

    # 5. Badges de Modo no canto direito
    modes = []
    if blur_faces:
        modes.append("LGPD:ON")
    if high_density:
        modes.append("DENSIDADE+")
    if heatmap_on:
        modes.append("CALOR")
    if low_light_on:
        modes.append("LOW-LIGHT")
    mode_str = " | ".join(modes) if modes else "PADRAO"
    mode_color = (120, 255, 140) if blur_faces else (100, 120, 255)
    (mw, _), _ = cv2.getTextSize(mode_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
    cv2.putText(frame, mode_str, (max(980, w - mw - 16), 35), cv2.FONT_HERSHEY_SIMPLEX, 0.50, mode_color, 1, cv2.LINE_AA)

    # 6. Indicador de Pausa
    if is_paused:
        cv2.rectangle(frame, (w // 2 - 140, h // 2 - 32), (w // 2 + 140, h // 2 + 32), (0, 0, 180), -1)
        cv2.putText(
            frame,
            "PAUSADO [ESPACO]",
            (w // 2 - 115, h // 2 + 8),
            cv2.FONT_HERSHEY_DUPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 7. Barra inferior: Atalhos
    shortcuts = (
        "[S] Foto | [B] LGPD | [D] Alta Densidade | [T] Rastreio | [H] Calor | [L] Luz | [+/-] Conf | [C] Reset | [ESPACO] Pausa | [Q] Sair"
    )
    cv2.putText(frame, shortcuts, (14, h - 13), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 220, 230), 1, cv2.LINE_AA)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_MODEL_NAME, help="Pesos do modelo YOLO")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE, help="Confiança mínima (0 a 1)")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Threshold IoU")
    parser.add_argument("--device", default=DEVICE, help="Dispositivo de execução (intel:gpu, intel:cpu, cpu, 0)")
    parser.add_argument("--no-blur", action="store_true", help="Desativa o desfoque facial de privacidade")
    parser.add_argument("--high-density", action="store_true", help="Inicia com Modo Alta Densidade ativo")
    args = parser.parse_args()

    if not 0 < args.confidence <= 1:
        parser.error("--confidence deve estar entre 0 e 1")

    logger = get_logger("webcam")
    logger.info("Carregando modelo %s (Dispositivo: %s)...", args.weights, args.device)
    model = load_model(args.weights, device=args.device)

    logger.info("Abrindo câmera assíncrona no índice %d...", args.camera)
    camera = ThreadedCamera(source=args.camera, width=1280, height=720).start()
    if not camera.isOpened():
        logger.error("Não foi possível acessar a câmera no índice %d.", args.camera)
        raise RuntimeError(f"Câmera {args.camera} indisponível.")

    window_name = "Visao Cidada YOLO - Monitoramento Inteligente em Tempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    blur_faces = not args.no_blur
    high_density = bool(args.high_density)
    tracking_on = True
    heatmap_on = False
    low_light_on = False
    conf_thresh = float(args.confidence)

    tracker = PersonTracker(min_hits_to_confirm=2)
    heatmap_acc = HeatmapAccumulator()

    peak_count = 0
    is_paused = False
    prev_time = time.perf_counter()
    fps_ema = 0.0
    last_frame_annotated: np.ndarray | None = None
    last_count = 0
    last_latency = 0.0
    conf_history: list[float] = []
    lat_history: list[float] = []

    logger.info("Webcam iniciada com sucesso. Pressione Q na janela para sair.")

    try:
        while True:
            if not is_paused:
                ok, raw_frame = camera.read()
                if not ok or raw_frame is None:
                    time.sleep(0.01)
                    continue

                curr_time = time.perf_counter()
                time_diff = max(curr_time - prev_time, 1e-5)
                prev_time = curr_time
                inst_fps = 1.0 / time_diff
                fps_ema = inst_fps if fps_ema == 0.0 else (0.15 * inst_fps + 0.85 * fps_ema)

                proc_frame = apply_preprocessing(raw_frame, enhance=True) if low_light_on else raw_frame
                result = detect_people(
                    model,
                    proc_frame,
                    confidence=conf_thresh,
                    iou=args.iou,
                    device=args.device,
                    high_density=high_density,
                )
                result.orig_img = raw_frame

                latency_ms = float(getattr(result, "inference_latency_ms", 0.0))
                lat_history.append(latency_ms)
                count = count_people(result)
                peak_count = max(peak_count, count)

                has_boxes = getattr(result, "boxes", None) is not None and len(result.boxes) > 0
                boxes_np = result.boxes.xyxy.cpu().numpy() if has_boxes else np.empty((0, 4), dtype=np.float32)
                confs_np = result.boxes.conf.cpu().numpy() if has_boxes else np.empty((0,), dtype=np.float32)
                if len(confs_np) > 0:
                    conf_history.append(float(np.mean(confs_np)))

                active_tracks = tracker.update(boxes_np, confs_np, timestamp=curr_time) if tracking_on else None
                heatmap_acc.update(raw_frame.shape, boxes_np)

                annotated = annotate_frame(
                    result,
                    blur_faces=blur_faces,
                    tracks=active_tracks,
                    heatmap=heatmap_acc if heatmap_on else None,
                    show_trails=tracking_on,
                )

                unique_val = max(peak_count, tracker.unique_people_count) if tracking_on else peak_count
                display_frame = draw_hud(
                    annotated,
                    count=count,
                    unique_count=unique_val,
                    peak_count=peak_count,
                    fps=fps_ema,
                    latency_ms=latency_ms,
                    confidence=conf_thresh,
                    blur_faces=blur_faces,
                    high_density=high_density,
                    tracking_on=tracking_on,
                    heatmap_on=heatmap_on,
                    low_light_on=low_light_on,
                    is_paused=False,
                )
                last_frame_annotated = annotated
                last_count = count
                last_latency = latency_ms
            else:
                base_frame = (
                    last_frame_annotated.copy()
                    if last_frame_annotated is not None
                    else np.zeros((720, 1280, 3), dtype=np.uint8)
                )
                unique_val = max(peak_count, tracker.unique_people_count) if tracking_on else peak_count
                display_frame = draw_hud(
                    base_frame,
                    count=last_count,
                    unique_count=unique_val,
                    peak_count=peak_count,
                    fps=0.0,
                    latency_ms=last_latency,
                    confidence=conf_thresh,
                    blur_faces=blur_faces,
                    high_density=high_density,
                    tracking_on=tracking_on,
                    heatmap_on=heatmap_on,
                    low_light_on=low_light_on,
                    is_paused=True,
                )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("b"), ord("B")):
                blur_faces = not blur_faces
                logger.info("Desfoque facial LGPD: %s", "Ligado" if blur_faces else "Desligado")
            elif key in (ord("d"), ord("D")):
                high_density = not high_density
                logger.info("Modo Alta Densidade (Tiling/Multi-Escala): %s", "Ligado" if high_density else "Desligado")
            elif key in (ord("t"), ord("T")):
                tracking_on = not tracking_on
                logger.info("Rastreamento de Pessoas Únicas: %s", "Ligado" if tracking_on else "Desligado")
            elif key in (ord("h"), ord("H")):
                heatmap_on = not heatmap_on
                logger.info("Mapa de Calor Espacial: %s", "Ligado" if heatmap_on else "Desligado")
            elif key in (ord("l"), ord("L")):
                low_light_on = not low_light_on
                logger.info("Realce Low-Light (CLAHE): %s", "Ligado" if low_light_on else "Desligado")
            elif key in (ord("+"), ord("=")):
                conf_thresh = max(0.10, round(conf_thresh - 0.05, 2))
                logger.info("Sensibilidade aumentada (Confiança mínima: %.2f)", conf_thresh)
            elif key in (ord("-"), ord("_")):
                conf_thresh = min(0.90, round(conf_thresh + 0.05, 2))
                logger.info("Sensibilidade reduzida (Confiança mínima: %.2f)", conf_thresh)
            elif key in (ord("c"), ord("C")):
                peak_count = last_count
                tracker.reset()
                heatmap_acc.reset()
                logger.info("Contadores de pico, únicos e mapa de calor resetados.")
            elif key == ord(" "):
                is_paused = not is_paused
                logger.info("Pausa alternada para: %s", is_paused)
            elif key in (ord("s"), ord("S")):
                saved_path = save_snapshot(display_frame, prefix="snapshot_manual")
                logger.info("Snapshot salvo em: %s", saved_path)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if peak_count > 0:
            unique_final = max(peak_count, tracker.unique_people_count)
            avg_c = float(np.mean(conf_history)) if conf_history else 0.0
            avg_l = float(np.mean(lat_history)) if lat_history else 0.0
            register_detection(
                source=f"Webcam (Cam {args.camera})",
                input_type="webcam_aovivo",
                count=peak_count,
                confidence=conf_thresh,
                device=args.device,
                unique_people=unique_final,
                avg_confidence=avg_c,
                latency_ms=avg_l,
                model_name=args.weights,
            )
        logger.info("Sessão da webcam finalizada.")


if __name__ == "__main__":
    main()

"""
Módulo Profissional de Detecção e Contagem em Tempo Real via Webcam.
Executa com OpenCV e overlay HUD (Head-Up Display) de alta tecnologia.
"""

import argparse
from datetime import datetime
import time

import cv2
import numpy as np

from config import CONFIDENCE, DEFAULT_MODEL_NAME, DEVICE, IOU_THRESHOLD
from inference import (
    annotate_frame,
    count_people,
    detect_people,
    load_model,
    register_detection,
    save_snapshot,
)
from utils.logger import get_logger


def draw_hud(
    frame: np.ndarray,
    count: int,
    peak_count: int,
    fps: float,
    latency_ms: float,
    blur_faces: bool,
    is_paused: bool,
    device_name: str,
) -> np.ndarray:
    """Desenha painel HUD profissional com métricas em tempo real sobre o frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Barra superior com gradiente/escurecimento
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 24, 30), -1)
    # Barra inferior de atalhos
    cv2.rectangle(overlay, (0, h - 35), (w, h), (15, 18, 22), -1)

    # Mescla suave para efeito de vidro translúcido
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # 1. Indicador de Pessoas
    count_color = (0, 235, 120) if count > 0 else (180, 180, 180)
    cv2.putText(frame, f"PESSOAS: {count}", (16, 36), cv2.FONT_HERSHEY_DUPLEX, 0.85, count_color, 2, cv2.LINE_AA)

    # 2. Pico Máximo
    cv2.putText(frame, f"PICO: {peak_count}", (220, 36), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 200, 80), 2, cv2.LINE_AA)

    # 3. FPS & Latência
    fps_color = (0, 220, 255) if fps >= 15 else (0, 140, 255)
    cv2.putText(frame, f"{fps:.1f} FPS | {latency_ms:.0f}ms", (370, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2, cv2.LINE_AA)

    # 4. Status de Privacidade
    blur_text = "PRIVACIDADE: ON" if blur_faces else "PRIVACIDADE: OFF"
    blur_color = (100, 255, 100) if blur_faces else (100, 100, 255)
    cv2.putText(frame, blur_text, (w - 230, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.55, blur_color, 1, cv2.LINE_AA)

    # 5. Indicador de Pausa
    if is_paused:
        cv2.rectangle(frame, (w // 2 - 120, h // 2 - 30), (w // 2 + 120, h // 2 + 30), (0, 0, 180), -1)
        cv2.putText(frame, "PAUSADO [ESPACO]", (w // 2 - 100, h // 2 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    # 6. Barra inferior: Atalhos
    shortcuts = "[S] Snapshot  |  [B] Privacidade  |  [C] Reset Pico  |  [ESPACO] Pausar  |  [Q] Sair"
    cv2.putText(frame, shortcuts, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_MODEL_NAME, help="Pesos do modelo YOLO")
    parser.add_argument("--camera", type=int, default=0, help="Índice da webcam (padrão: 0)")
    parser.add_argument("--confidence", type=float, default=CONFIDENCE, help="Confiança mínima (0 a 1)")
    parser.add_argument("--iou", type=float, default=IOU_THRESHOLD, help="Threshold IoU")
    parser.add_argument("--device", default=DEVICE, help="Dispositivo de execução (intel:gpu, intel:cpu, cpu, 0)")
    parser.add_argument("--no-blur", action="store_true", help="Desativa o desfoque facial de privacidade")
    args = parser.parse_args()

    if not 0 < args.confidence <= 1:
        parser.error("--confidence deve estar entre 0 e 1")

    logger = get_logger("webcam")
    logger.info("Carregando modelo %s (Dispositivo: %s)...", args.weights, args.device)
    model = load_model(args.weights, device=args.device)

    logger.info("Abrindo câmera índice %d...", args.camera)
    capture = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if cv2.CAP_DSHOW else 0)
    if not capture.isOpened():
        # Fallback para backend padrão
        capture = cv2.VideoCapture(args.camera)

    if not capture.isOpened():
        logger.error("Não foi possível acessar a câmera no índice %d.", args.camera)
        raise RuntimeError(f"Câmera {args.camera} indisponível.")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "Visao Cidada YOLO - Monitoramento em Tempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    blur_faces = not args.no_blur
    peak_count = 0
    is_paused = False
    prev_time = time.perf_counter()
    fps = 0.0
    last_frame_annotated = None
    last_count = 0
    last_latency = 0.0

    logger.info("Webcam iniciada com sucesso. Pressione Q na janela para sair.")

    try:
        while True:
            if not is_paused:
                ok, raw_frame = capture.read()
                if not ok:
                    logger.warning("Falha ao capturar frame da webcam.")
                    break

                curr_time = time.perf_counter()
                time_diff = curr_time - prev_time
                prev_time = curr_time
                fps = 1.0 / max(time_diff, 1e-5)

                result = detect_people(
                    model, raw_frame, confidence=args.confidence, iou=args.iou, device=args.device
                )
                latency_ms = getattr(result, "inference_latency_ms", 0.0)
                count = count_people(result)
                peak_count = max(peak_count, count)

                annotated = annotate_frame(result, blur_faces=blur_faces)
                display_frame = draw_hud(
                    annotated,
                    count=count,
                    peak_count=peak_count,
                    fps=fps,
                    latency_ms=latency_ms,
                    blur_faces=blur_faces,
                    is_paused=is_paused,
                    device_name=args.device,
                )
                last_frame_annotated = display_frame
                last_count = count
                last_latency = latency_ms
            else:
                base_frame = last_frame_annotated.copy() if last_frame_annotated is not None else np.zeros((720, 1280, 3), dtype=np.uint8)
                display_frame = draw_hud(
                    base_frame,
                    count=last_count,
                    peak_count=peak_count,
                    fps=0.0,
                    latency_ms=last_latency,
                    blur_faces=blur_faces,
                    is_paused=True,
                    device_name=args.device,
                )

            cv2.imshow(window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # Q ou ESC
                break
            elif key == ord("b"):
                blur_faces = not blur_faces
                logger.info("Desfoque facial alternado para: %s", "Ligado" if blur_faces else "Desligado")
            elif key == ord("c"):
                peak_count = last_count
                logger.info("Contador de pico resetado.")
            elif key == ord(" "):
                is_paused = not is_paused
                logger.info("Pausa alternada para: %s", is_paused)
            elif key == ord("s"):
                saved_path = save_snapshot(display_frame, prefix="snapshot_manual")
                logger.info("Snapshot salvo em: %s", saved_path)
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if peak_count > 0:
            register_detection(f"Webcam (Cam {args.camera})", "webcam_aovivo", peak_count, args.confidence, device=args.device)
        logger.info("Sessão da webcam finalizada.")


if __name__ == "__main__":
    main()

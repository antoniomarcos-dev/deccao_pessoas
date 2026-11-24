"""
Visão Cidadã YOLO - Aplicativo Unificado (Interface Web Streamlit + Modo Demonstração Feira Desktop).
Conecta a interface web e o modo de demonstração em tela cheia ao motor Back-End de Alta Performance
(OpenVINO/PyTorch, Tiling Multi-Escala WBF, Rastreamento Multi-Objeto, Mapa de Calor e SQLite/CSV).
Otimizado para eliminar falhas de reconciliação DOM (React removeChild) e rodar com fluidez em notebooks Intel Core i7 + UHD Graphics.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from config import (
    CONFIDENCE,
    DEFAULT_MODEL_NAME,
    DEVICE,
    IMAGE_SIZE,
    IOU_THRESHOLD,
    MAX_DETECTIONS_DEFAULT,
    ROOT_DIR,
    VIDEOS_EXPORT_DIR,
    ensure_project_directories,
    get_available_devices,
    get_available_models,
    get_system_hardware_info,
)
from inference import (
    annotate_frame,
    apply_preprocessing,
    blur_detected_faces,
    classify_scenario,
    clear_detection_records,
    compute_crowd_metrics,
    count_people,
    detect_people,
    extract_detections_info,
    load_model,
    load_records_df,
    process_video,
    register_detection,
    save_snapshot,
)
from utils.camera import ThreadedCamera
from utils.tracker import HeatmapAccumulator, PersonTracker


def classificar_cenario(qtd: int) -> tuple[str, tuple[int, int, int]]:
    """Retorna rótulo completo de cenário e cor BGR correspondente (compatível com Modo Feira)."""
    _, full_lbl, bgr_color, _, _ = classify_scenario(qtd)
    return full_lbl, bgr_color


def run_feira_desktop_mode(
    model_name: str = DEFAULT_MODEL_NAME,
    source: str = "0",
    conf: float = 0.25,
    iou: float = 0.55,
    device: str = DEVICE,
    alta_densidade_init: bool = False,
    blur_init: bool = False,
) -> None:
    """
    Executa o Modo Demonstração Feira em janela nativa OpenCV de alta taxa de quadros.
    """
    model = load_model(model_name, device=device)
    src = int(source) if str(source).isdigit() else source

    is_live_cam = isinstance(src, int)
    if is_live_cam:
        cap = ThreadedCamera(source=src, width=1280, height=720).start()
    else:
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir a câmera/fonte: {source}")

    alta_densidade = bool(alta_densidade_init)
    blur_lgpd = bool(blur_init)
    heatmap_on = False
    tracker = PersonTracker(min_hits_to_confirm=2)
    heatmap_acc = HeatmapAccumulator()
    peak_seen = 0

    print("=================================================================")
    print("🚀 Visão Cidadã (Modo Feira Unificado) iniciado!")
    print("   [D] -> Ativar/Desativar Modo Alta Densidade (Tiling + Fusão WBF)")
    print("   [B] -> Ativar/Desativar Privacidade Facial (LGPD)")
    print("   [H] -> Ativar/Desativar Mapa de Calor de Ocupação")
    print("   [S] -> Salvar Foto/Snapshot")
    print("   [+]/[-] -> Aumentar/Diminuir sensibilidade (confiança)")
    print("   [Q] ou [ESC] -> Encerrar demonstração")
    print("=================================================================")

    window_title = "Visao Cidada - Demonstracao Feira (Tempo Real)"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    prev_t = time.perf_counter()
    fps_ema = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if is_live_cam:
                    time.sleep(0.01)
                    continue
                break

            res = detect_people(
                model=model,
                frame=frame,
                confidence=conf,
                iou=iou,
                device=device,
                high_density=alta_densidade,
                max_det=500,
                fast_stream_mode=True,
            )

            has_boxes = res.boxes is not None and len(res.boxes) > 0
            boxes = res.boxes.xyxy.cpu().numpy() if has_boxes else np.empty((0, 4), dtype=np.float32)
            scores = res.boxes.conf.cpu().numpy() if has_boxes else np.empty((0,), dtype=np.float32)
            qtd = len(boxes)
            peak_seen = max(peak_seen, qtd)
            status_txt, cor = classificar_cenario(qtd)

            now = time.perf_counter()
            tracker.update(boxes, scores, timestamp=now)
            heatmap_acc.update(frame.shape, boxes)

            if blur_lgpd:
                frame = blur_detected_faces(frame, boxes=boxes if qtd > 0 else None)

            if heatmap_on:
                frame = heatmap_acc.render_overlay(frame)

            for box, sc in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), cor, 2, cv2.LINE_AA)
                lbl = f"pessoa {sc * 100:.0f}%"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
                y_lbl = max(y1, th + 8)
                cv2.rectangle(frame, (x1, y_lbl - th - 6), (x1 + tw + 8, y_lbl), cor, -1)
                cv2.putText(
                    frame,
                    lbl,
                    (x1 + 4, y_lbl - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            inst_fps = 1.0 / max(now - prev_t, 1e-6)
            prev_t = now
            fps_ema = inst_fps if fps_ema == 0.0 else (0.15 * inst_fps + 0.85 * fps_ema)

            h, w = frame.shape[:2]
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 58), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

            modo_str = "ALTA DENSIDADE" if alta_densidade else "TEMPO REAL"
            unicas_val = max(peak_seen, tracker.unique_people_count)
            hud_line = (
                f"Visao Cidada | Agora: {qtd} (Unicas: {unicas_val}) | {status_txt} | "
                f"Conf: {conf:.2f} | {modo_str} ({fps_ema:.1f} FPS)"
            )
            cv2.putText(
                frame,
                hud_line,
                (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 128),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key in (ord("d"), ord("D")):
                alta_densidade = not alta_densidade
            elif key in (ord("b"), ord("B")):
                blur_lgpd = not blur_lgpd
            elif key in (ord("h"), ord("H")):
                heatmap_on = not heatmap_on
            elif key in (ord("s"), ord("S")):
                save_snapshot(frame, prefix="feira")
            elif key in (ord("+"), ord("=")):
                conf = max(0.10, round(conf - 0.03, 2))
            elif key in (ord("-"), ord("_")):
                conf = min(0.85, round(conf + 0.03, 2))
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if peak_seen > 0:
            register_detection(
                source=f"Feira ({source})",
                input_type="feira_aovivo",
                count=peak_seen,
                confidence=conf,
                device=device,
                unique_people=max(peak_seen, tracker.unique_people_count),
                model_name=model_name,
            )


def apply_custom_styles() -> None:
    """Aplica folha de estilos CSS personalizada com proteção contra mutação DOM de tradutores (removeChild fix)."""
    st.markdown(
        """
        <meta name="google" content="notranslate">
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 1.2rem 1.8rem;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            margin-bottom: 1.5rem;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        }
        .app-title {
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        .app-subtitle {
            color: #94A3B8;
            font-size: 0.9rem;
            margin-top: 0.2rem;
        }
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
            letter-spacing: 0.02em;
        }
        .badge-active {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        .badge-device {
            background: rgba(56, 189, 248, 0.15);
            color: #38BDF8;
            border: 1px solid rgba(56, 189, 248, 0.3);
        }
        .badge-privacy {
            background: rgba(168, 85, 247, 0.15);
            color: #C084FC;
            border: 1px solid rgba(168, 85, 247, 0.3);
        }

        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.1rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }
        .metric-val {
            font-size: 1.9rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.1;
        }
        .metric-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            margin-top: 0.35rem;
        }

        .scenario-banner {
            padding: 0.75rem 1.2rem;
            border-radius: 12px;
            font-weight: 700;
            font-size: 1.05rem;
            margin-bottom: 0.8rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(15, 23, 42, 0.85);
            border: 1px solid rgba(255, 255, 255, 0.12);
        }

        /* Painel Lateral Unificado de Stream (Evita mutação de múltiplos nós React DOM) */
        .live-stats-panel {
            display: flex;
            flex-direction: column;
            gap: 10px;
            background: rgba(15, 23, 42, 0.75);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 14px;
        }
        .live-stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 8px 12px;
            background: rgba(30, 41, 59, 0.65);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .live-stat-title {
            font-size: 0.82rem;
            color: #94A3B8;
            font-weight: 600;
        }
        .live-stat-num {
            font-size: 1.35rem;
            color: #F8FAFC;
            font-weight: 800;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(device_str: str, blur_enabled: bool, model_name: str, high_density: bool) -> None:
    """Renderiza cabeçalho com identificadores visuais."""
    if device_str == "intel:gpu":
        device_label = "GPU Intel UHD (OpenVINO)"
    elif device_str == "intel:cpu":
        device_label = "CPU Intel Core i7 (OpenVINO)"
    elif "cuda" in device_str.lower() or device_str == "0":
        device_label = "GPU CUDA (NVIDIA)"
    else:
        device_label = "CPU Multithread (PyTorch)"
    privacy_label = "LGPD: Ativa" if blur_enabled else "LGPD: Desativada"
    mode_label = "Alta Densidade (Tiling)" if high_density else "Tempo Real"

    st.markdown(
        f"""
        <div class="app-header notranslate" translate="no">
            <div>
                <h1 class="app-title">Visão Cidadã YOLO</h1>
                <div class="app-subtitle">Motor Inteligente de Contagem, Rastreamento de Fluxo e Densidade Ocupacional</div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-badge badge-active">● Modelo: {Path(model_name).stem} ({mode_label})</span>
                <span class="status-badge badge-device">⚙ {device_label}</span>
                <span class="status-badge badge-privacy">🛡 {privacy_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_scenario_banner(count: int, unique_count: int | None = None) -> str:
    """Gera HTML de destaque para a classificação de cenário (Individual / Em Grupo / Aglomeração)."""
    _, full_lbl, _, hex_color, _ = classify_scenario(count)
    uniq_str = f" &nbsp;|&nbsp; Fluxo Único: {unique_count}" if unique_count is not None else ""
    return (
        f'<div class="scenario-banner notranslate" translate="no" style="border-left: 6px solid {hex_color};">'
        f'<span style="color: {hex_color};">📌 CENÁRIO: {full_lbl}</span>'
        f'<span style="color: #CBD5E1; font-size: 0.9rem;">Ocupação Instantânea: {count}{uniq_str}</span>'
        f'</div>'
    )


def render_live_stats_html(
    count: int,
    unique_val: int,
    peak_seen: int,
    short_scen: str,
    hex_color: str,
    fps_ema: float,
    lat_ema: float,
    blur_faces: bool,
) -> str:
    """
    Renderiza todas as métricas da transmissão em um único bloco HTML estável (`notranslate`),
    eliminando colisões de `removeChild` no React DOM do Streamlit.
    """
    lgpd_bg = "rgba(16, 185, 129, 0.16)" if blur_faces else "rgba(245, 158, 11, 0.16)"
    lgpd_col = "#34D399" if blur_faces else "#FBBF24"
    lgpd_txt = "🛡️ Privacidade LGPD: Ativa" if blur_faces else "⚠️ Privacidade LGPD: Desativada"

    return f"""
    <div class="live-stats-panel notranslate" translate="no">
        <div class="live-stat-row" style="border-left: 4px solid {hex_color};">
            <span class="live-stat-title">📌 Cenário Atual</span>
            <span class="live-stat-num" style="color: {hex_color}; font-size: 1.1rem;">{short_scen}</span>
        </div>
        <div class="live-stat-row">
            <span class="live-stat-title">👥 Pessoas Agora</span>
            <span class="live-stat-num">{count}</span>
        </div>
        <div class="live-stat-row">
            <span class="live-stat-title">🚶 Pessoas Únicas (Fluxo)</span>
            <span class="live-stat-num">{unique_val}</span>
        </div>
        <div class="live-stat-row">
            <span class="live-stat-title">🏆 Maior Pico</span>
            <span class="live-stat-num">{peak_seen}</span>
        </div>
        <div class="live-stat-row">
            <span class="live-stat-title">⚡ Taxa (FPS)</span>
            <span class="live-stat-num" style="color: #38BDF8;">{fps_ema:.1f}</span>
        </div>
        <div class="live-stat-row">
            <span class="live-stat-title">⏱️ Latência IA</span>
            <span class="live-stat-num" style="font-size: 1.15rem;">{lat_ema:.1f} ms</span>
        </div>
        <div style="padding: 10px 12px; border-radius: 10px; background: {lgpd_bg}; color: {lgpd_col}; font-weight: 700; font-size: 0.85rem; text-align: center;">
            {lgpd_txt}
        </div>
    </div>
    """


def get_camera_capture(index: int) -> ThreadedCamera | None:
    """
    Obtém ou reutiliza o leitor de câmera assíncrono em thread dedicada.
    Usa resolução 960x540 (qHD 16:9) para transmissão fluida sem gargalo de WebSocket no navegador.
    """
    active_cap: ThreadedCamera | None = st.session_state.get("active_webcam_cap")
    active_idx = st.session_state.get("active_webcam_idx")

    if active_cap is not None and active_cap.isOpened() and active_idx == index:
        return active_cap

    release_camera_capture()

    try:
        cam = ThreadedCamera(source=int(index), width=960, height=540).start()
        if cam.isOpened():
            st.session_state["active_webcam_cap"] = cam
            st.session_state["active_webcam_idx"] = index
            return cam
        cam.release()
    except Exception:
        pass

    return None


def release_camera_capture() -> None:
    """Libera a câmera aberta com segurança."""
    active_cap = st.session_state.get("active_webcam_cap")
    if active_cap is not None:
        try:
            active_cap.release()
        except Exception:
            pass
        st.session_state["active_webcam_cap"] = None
        st.session_state["active_webcam_idx"] = None
        time.sleep(0.05)


def main() -> None:
    st.set_page_config(
        page_title="Visão Cidadã - YOLO",
        page_icon="👥",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_custom_styles()
    ensure_project_directories()

    # --- SIDEBAR DE CONTROLE E CONFIGURAÇÃO ---
    st.sidebar.markdown("### ⚙️ Painel de Controle")
    available_models = get_available_models()
    selected_model_name = st.sidebar.selectbox(
        "Pesos do Modelo (YOLO)",
        options=available_models,
        index=0,
        help="Modelo oficial mais treinado (mAP50: 79.2%) com aceleração OpenVINO.",
    )

    confidence = st.sidebar.slider(
        "Confiança Mínima",
        min_value=0.05,
        max_value=0.95,
        value=float(CONFIDENCE),
        step=0.05,
        help="Probabilidade mínima do algoritmo classificar o objeto como 'pessoa'.",
    )

    iou_thresh = st.sidebar.slider(
        "Threshold NMS (IoU)",
        min_value=0.10,
        max_value=0.90,
        value=float(IOU_THRESHOLD),
        step=0.05,
        help="Supressão Não-Máxima para evitar caixas duplicadas para a mesma pessoa.",
    )

    max_det = st.sidebar.slider(
        "Detecções Máximas por Frame",
        min_value=50,
        max_value=1000,
        value=int(MAX_DETECTIONS_DEFAULT),
        step=50,
        help="Número máximo de pessoas detectadas simultaneamente por frame.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🧠 Motores de Inteligência & Privacidade")

    blur_faces = st.sidebar.toggle(
        "🛡️ Desfoque Facial Estrito (LGPD)",
        value=True,
        help="Anonimiza rostos frontais e close-ups de webcam com detecção adaptativa para conformidade total com a LGPD.",
    )

    feira_color_mode = st.sidebar.toggle(
        "🎨 Modo Feira (Cores por Cenário)",
        value=True,
        help="Colore as caixas automaticamente conforme o cenário da cena: Verde (Individual), Amarelo (Em Grupo), Laranja/Vermelho (Aglomeração).",
    )

    high_density_global = st.sidebar.toggle(
        "🔍 Modo Alta Densidade (Grupos/Multidões)",
        value=False,
        help="Ativa inferência em sub-regiões sobrepostas (Tiling + Fusão WBF) para detectar pessoas pequenas ao fundo.",
    )

    heatmap_enabled = st.sidebar.toggle(
        "🔥 Mapa de Calor de Ocupação",
        value=False,
        help="Sobrepõe camada térmica indicando as áreas com maior concentração de pessoas.",
    )

    low_light_prep = st.sidebar.toggle(
        "🌙 Realce Low-Light (CLAHE)",
        value=False,
        help="Aplica equalização adaptativa no espaço LAB para melhorar detecção em ambientes escuros.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ Aceleração de Hardware")
    available_devices = get_available_devices()
    device_ids = [d["id"] for d in available_devices]
    device_names = {d["id"]: d["name"] for d in available_devices}
    default_dev_idx = device_ids.index(DEVICE) if DEVICE in device_ids else 0

    selected_device = st.sidebar.radio(
        "Dispositivo de Execução",
        options=device_ids,
        format_func=lambda d: device_names.get(d, d),
        index=default_dev_idx,
        help="Acelere a inferência usando a GPU Intel UHD integrada (OpenVINO) ou o processador Intel Core i7.",
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Visão Cidadã YOLO v3.2 | Otimizado para Intel Core i7 + UHD Graphics")

    try:
        model = load_model(selected_model_name, device=selected_device)
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo '{selected_model_name}': {e}")
        return

    render_header(selected_device, blur_faces, selected_model_name, high_density_global)

    # --- ABAS PRINCIPAIS ---
    tab_webcam, tab_media, tab_dashboard, tab_settings = st.tabs([
        "📷 Webcam & Modo Feira ao Vivo",
        "📁 Análise de Mídia (Fotos e Vídeos)",
        "📊 Dashboard & Histórico",
        "⚙️ Informações do Sistema & Modelo",
    ])

    # =========================================================================
    # TAB 1: WEBCAM EM TEMPO REAL & MODO FEIRA
    # =========================================================================
    with tab_webcam:
        st.markdown("### 🎥 Monitoramento e Demonstração ao Vivo por Webcam")
        st.markdown(
            "Selecione o modo desejado: **Transmissão Contínua** no navegador, **Janela Desktop Tela Cheia (Modo Feira)** ou **Captura Instantânea**."
        )

        webcam_mode = st.radio(
            "Modo da Câmera",
            ["Transmissão Contínua (Stream ao Vivo)", "Captura Instantânea (Foto pela Câmera)"],
            horizontal=True,
        )

        if webcam_mode == "Transmissão Contínua (Stream ao Vivo)":
            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
            with col_ctrl1:
                camera_index = st.number_input("Índice da Câmera", min_value=0, max_value=5, value=0, step=1)
            with col_ctrl2:
                run_cam = st.toggle("▶ Iniciar Câmera Web", value=False, key="stream_cam_toggle")
            with col_ctrl3:
                if st.button("🖥️ Abrir Janela Nativa Modo Feira (Tela Cheia)"):
                    release_camera_capture()
                    cmd = [
                        sys.executable,
                        str(ROOT_DIR / "app.py"),
                        "--feira",
                        "--model",
                        selected_model_name,
                        "--source",
                        str(int(camera_index)),
                        "--conf",
                        str(confidence),
                        "--iou",
                        str(iou_thresh),
                        "--device",
                        selected_device,
                    ]
                    if high_density_global:
                        cmd.append("--high-density")
                    if blur_faces:
                        cmd.append("--blur")
                    subprocess.Popen(cmd, cwd=str(ROOT_DIR))
                    st.toast("Janela nativa Modo Feira iniciada! Use [D], [B], [H], [S] ou [Q] na janela de vídeo.", icon="🚀")

            if run_cam:
                col_frame, col_stats = st.columns([3, 1])
                with col_frame:
                    frame_placeholder = st.empty()
                with col_stats:
                    stats_placeholder = st.empty()

                cap = get_camera_capture(int(camera_index))

                if cap is None or not cap.isOpened():
                    st.error(f"Não foi possível abrir a câmera no índice {camera_index}. Verifique se outra aplicação está utilizando-a.")
                    release_camera_capture()
                else:
                    if "stream_tracker" not in st.session_state:
                        st.session_state["stream_tracker"] = PersonTracker(min_hits_to_confirm=2)
                    if "stream_heatmap" not in st.session_state:
                        st.session_state["stream_heatmap"] = HeatmapAccumulator()

                    tracker: PersonTracker = st.session_state["stream_tracker"]
                    heatmap_acc: HeatmapAccumulator = st.session_state["stream_heatmap"]
                    peak_seen = st.session_state.get("stream_peak_seen", 0)
                    prev_t = time.perf_counter()
                    fps_ema = 0.0
                    lat_ema = 0.0
                    last_stats_update = 0.0

                    while st.session_state.get("stream_cam_toggle", False):
                        try:
                            ret, raw_frame = cap.read()
                            if not ret or raw_frame is None:
                                time.sleep(0.01)
                                continue

                            curr_t = time.perf_counter()
                            inst_fps = 1.0 / max(curr_t - prev_t, 1e-5)
                            prev_t = curr_t
                            fps_ema = inst_fps if fps_ema == 0.0 else (0.18 * inst_fps + 0.82 * fps_ema)

                            proc_frame = apply_preprocessing(raw_frame) if low_light_prep else raw_frame

                            result = detect_people(
                                model,
                                proc_frame,
                                confidence=confidence,
                                iou=iou_thresh,
                                device=selected_device,
                                max_det=max_det,
                                high_density=high_density_global,
                                fast_stream_mode=True,
                            )
                            result.orig_img = raw_frame
                            count = count_people(result)
                            peak_seen = max(peak_seen, count)
                            st.session_state["stream_peak_seen"] = peak_seen

                            raw_lat = float(getattr(result, "inference_latency_ms", 0.0))
                            lat_ema = raw_lat if lat_ema == 0.0 else (0.25 * raw_lat + 0.75 * lat_ema)

                            has_boxes = getattr(result, "boxes", None) is not None and len(result.boxes) > 0
                            boxes_np = result.boxes.xyxy.cpu().numpy() if has_boxes else np.empty((0, 4), dtype=np.float32)
                            confs_np = result.boxes.conf.cpu().numpy() if has_boxes else np.empty((0,), dtype=np.float32)

                            active_tracks = tracker.update(boxes_np, confs_np, timestamp=curr_t)
                            heatmap_acc.update(raw_frame.shape, boxes_np)

                            annotated = annotate_frame(
                                result,
                                blur_faces=blur_faces,
                                tracks=active_tracks,
                                heatmap=heatmap_acc if heatmap_enabled else None,
                                show_trails=False,
                                color_by_scenario=feira_color_mode,
                            )
                            short_scen, _, _, hex_color, _ = classify_scenario(count)
                            unique_val = max(peak_seen, tracker.unique_people_count)

                            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                            # Atualiza painel lateral de métricas a 4 Hz (evita sobrecarga React DOM / removeChild)
                            if (curr_t - last_stats_update) >= 0.22:
                                last_stats_update = curr_t
                                stats_placeholder.markdown(
                                    render_live_stats_html(
                                        count=count,
                                        unique_val=unique_val,
                                        peak_seen=peak_seen,
                                        short_scen=short_scen,
                                        hex_color=hex_color,
                                        fps_ema=fps_ema,
                                        lat_ema=lat_ema,
                                        blur_faces=blur_faces,
                                    ),
                                    unsafe_allow_html=True,
                                )

                            time.sleep(0.012)
                        except Exception:
                            break

                    if not st.session_state.get("stream_cam_toggle", False):
                        release_camera_capture()
                        if peak_seen > 0:
                            unique_final = max(peak_seen, tracker.unique_people_count)
                            register_detection(
                                f"Webcam (Índice {camera_index})",
                                "webcam",
                                peak_seen,
                                confidence,
                                device=selected_device,
                                unique_people=unique_final,
                                latency_ms=lat_ema,
                                model_name=selected_model_name,
                            )
            else:
                release_camera_capture()
                st.session_state["stream_peak_seen"] = 0
                if "stream_tracker" in st.session_state:
                    st.session_state["stream_tracker"].reset()
                if "stream_heatmap" in st.session_state:
                    st.session_state["stream_heatmap"].reset()
                st.markdown(
                    """
                    <div class="notranslate" translate="no" style="text-align: center; padding: 40px; background: rgba(30, 41, 59, 0.4); border-radius: 12px; border: 2px dashed rgba(255,255,255,0.1);">
                        <p style="font-size: 1.1rem; color: #94A3B8; margin: 0;">A câmera está em espera. Ative <b>'▶ Iniciar Câmera Web'</b> para assistir no navegador ou clique em <b>'🖥️ Abrir Janela Nativa Modo Feira'</b> para demonstração em tela cheia.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            st.markdown("##### Tire uma foto diretamente com a webcam do seu dispositivo:")
            cam_picture = st.camera_input("Capturar foto")

            if cam_picture is not None:
                bytes_data = cam_picture.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                proc_img = apply_preprocessing(cv_img) if low_light_prep else cv_img

                col_res1, col_res2 = st.columns([3, 2])
                with st.spinner("Analisando pessoas na captura..."):
                    res = detect_people(
                        model,
                        proc_img,
                        confidence=confidence,
                        iou=iou_thresh,
                        device=selected_device,
                        max_det=max_det,
                        high_density=high_density_global,
                    )
                    res.orig_img = cv_img
                    metrics = compute_crowd_metrics(res)
                    count = metrics["count"]
                    annotated_shot = annotate_frame(
                        res,
                        blur_faces=blur_faces,
                        heatmap=heatmap_enabled,
                        color_by_scenario=feira_color_mode,
                    )
                    detections = extract_detections_info(res)
                    register_detection(
                        "Webcam Snapshot",
                        "camera_input",
                        count,
                        confidence,
                        device=selected_device,
                        unique_people=count,
                        avg_confidence=metrics["avg_confidence"],
                        latency_ms=metrics["latency_ms"],
                        model_name=selected_model_name,
                    )

                st.markdown(render_scenario_banner(count), unsafe_allow_html=True)

                with col_res1:
                    st.image(
                        cv2.cvtColor(annotated_shot, cv2.COLOR_BGR2RGB),
                        caption=f"Resultado: {count} pessoa(s) | Cenário: {metrics['scenario_full']}",
                        use_container_width=True,
                    )

                with col_res2:
                    st.metric("Total Detectado", count)
                    st.metric("Cenário Classificado", metrics["scenario_short"])
                    st.metric("Densidade Espacial", f"{metrics['area_coverage_pct']:.1f}% da área")
                    if detections:
                        st.markdown("###### Detalhes das Detecções:")
                        st.dataframe(
                            pd.DataFrame(detections)[["id", "confidence", "bbox", "largura_px", "altura_px"]],
                            hide_index=True,
                            use_container_width=True,
                        )
                    else:
                        st.info("Nenhuma pessoa identificada na imagem com o nível de confiança atual.")

    # =========================================================================
    # TAB 2: ANÁLISE DE MÍDIA (FOTOS E VÍDEOS)
    # =========================================================================
    with tab_media:
        st.markdown("### 📁 Análise Inteligente de Imagens e Vídeos")

        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            uploaded_file = st.file_uploader(
                "Arraste ou selecione uma imagem ou vídeo para análise",
                type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"],
                help="Formatos de imagem suportados: JPG, PNG, WEBP. Vídeos: MP4, AVI, MOV, MKV.",
            )
        with col_m2:
            tta_enabled = st.toggle(
                "Modo Precisão Máxima (Tiling + Multi-Escala WBF)",
                value=high_density_global,
                help="Divide a imagem em sub-regiões sobrepostas e funde predições. Ideal para grupos, feiras, salas de aula e multidões.",
            )

        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            suffix = Path(uploaded_file.name).suffix.lower()

            # --- PROCESSAMENTO DE IMAGEM (COM CACHE DE SESSÃO INTELIGENTE) ---
            if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                raw_img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
                if raw_img is None:
                    st.error("Não foi possível decodificar o arquivo de imagem enviado.")
                else:
                    file_hash = hashlib.md5(file_bytes).hexdigest()[:12]
                    cache_key = (
                        file_hash,
                        selected_model_name,
                        selected_device,
                        round(confidence, 2),
                        round(iou_thresh, 2),
                        int(max_det),
                        bool(low_light_prep),
                        bool(tta_enabled),
                    )

                    if st.session_state.get("img_cache_key") != cache_key:
                        with st.spinner("Processando imagem com Inteligência Artificial..."):
                            proc_img = apply_preprocessing(raw_img) if low_light_prep else raw_img
                            res = detect_people(
                                model,
                                proc_img,
                                confidence=confidence,
                                iou=iou_thresh,
                                device=selected_device,
                                augment=tta_enabled,
                                max_det=max_det,
                                high_density=tta_enabled,
                            )
                            res.orig_img = raw_img
                            metrics = compute_crowd_metrics(res)
                            register_detection(
                                uploaded_file.name,
                                "imagem",
                                metrics["count"],
                                confidence,
                                device=selected_device,
                                unique_people=metrics["count"],
                                avg_confidence=metrics["avg_confidence"],
                                latency_ms=metrics["latency_ms"],
                                model_name=selected_model_name,
                            )
                            st.session_state["img_cache_key"] = cache_key
                            st.session_state["img_cache_res"] = res
                            st.session_state["img_cache_metrics"] = metrics

                    res = st.session_state["img_cache_res"]
                    metrics = st.session_state["img_cache_metrics"]
                    count = metrics["count"]
                    annotated_img = annotate_frame(
                        res,
                        blur_faces=blur_faces,
                        heatmap=heatmap_enabled,
                        color_by_scenario=feira_color_mode,
                    )
                    detections = extract_detections_info(res)

                    st.markdown(render_scenario_banner(count), unsafe_allow_html=True)

                    col_img1, col_img2 = st.columns(2)
                    with col_img1:
                        st.markdown("##### Imagem Original")
                        st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    with col_img2:
                        st.markdown(f"##### Imagem Analisada ({count} pessoas — {metrics['scenario_short']})")
                        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        st.metric("👥 Pessoas Detectadas", count)
                    with m2:
                        st.metric("📌 Cenário", metrics["scenario_short"])
                    with m3:
                        st.metric("🎯 Confiança Média", f"{metrics['avg_confidence'] * 100:.1f}%")
                    with m4:
                        st.metric("📐 Ocupação Espacial", f"{metrics['area_coverage_pct']:.1f}%")
                    with m5:
                        st.metric("⏱️ Tempo de Inferência", f"{metrics['latency_ms']:.1f} ms")

                    if detections:
                        with st.expander("📋 Ver coordenadas, dimensões e níveis de certeza individuais", expanded=False):
                            df_dets = pd.DataFrame(detections)
                            st.dataframe(df_dets, use_container_width=True, hide_index=True)

                    _, buffer = cv2.imencode(".jpg", annotated_img)
                    st.download_button(
                        label="⬇ Baixar Imagem Anotada (JPG)",
                        data=buffer.tobytes(),
                        file_name=f"analise_{Path(uploaded_file.name).stem}.jpg",
                        mime="image/jpeg",
                    )

            # --- PROCESSAMENTO DE VÍDEO ---
            else:
                st.markdown("##### Processamento de Vídeo com Rastreamento de Fluxo (MOT)")
                col_vopt1, col_vopt2, col_vopt3 = st.columns([2, 1, 1])
                with col_vopt1:
                    st.info("O vídeo será analisado com rastreamento de pessoas únicas, cálculo de permanência e curva de ocupação.")
                with col_vopt2:
                    skip_frames = st.selectbox(
                        "Amostragem de Frames",
                        [1, 2, 3, 5],
                        index=1,
                        format_func=lambda x: f"Processar cada {x}º frame" if x > 1 else "Todos os frames (1x)",
                    )
                with col_vopt3:
                    export_mp4 = st.checkbox("Gerar Vídeo MP4 para Download", value=True)

                btn_process = st.button("🚀 Iniciar Análise Completa do Vídeo", type="primary")

                if btn_process:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_media:
                        temp_media.write(file_bytes)
                        media_path = temp_media.name

                    export_path = (
                        VIDEOS_EXPORT_DIR / f"processado_{Path(uploaded_file.name).stem}.mp4"
                        if export_mp4
                        else None
                    )

                    try:
                        progress_bar = st.progress(0, text="Preparando vídeo...")
                        v_col_frame, v_col_stats = st.columns([3, 1])
                        with v_col_frame:
                            video_placeholder = st.empty()
                        with v_col_stats:
                            v_stats_placeholder = st.empty()

                        max_in_video = 0
                        unique_in_video = 0
                        history_counts: list[int] = []
                        history_stabilized: list[int] = []
                        last_v_ui_update = 0.0

                        gen = process_video(
                            model=model,
                            video_path=media_path,
                            confidence=confidence,
                            iou=iou_thresh,
                            blur_faces=blur_faces,
                            source_name=uploaded_file.name,
                            device=selected_device,
                            frame_step=skip_frames,
                            low_light_prep=low_light_prep,
                            high_density=tta_enabled,
                            enable_tracking=True,
                            enable_heatmap=heatmap_enabled,
                            color_by_scenario=feira_color_mode,
                            max_det=max_det,
                            export_video_path=export_path,
                        )

                        for ann_frame, count, idx, total, v_fps, extra in gen:
                            max_in_video = max(max_in_video, count)
                            unique_in_video = extra["unique_people"]
                            history_counts.append(count)
                            history_stabilized.append(extra["stabilized_count"])

                            now_v = time.perf_counter()
                            if (now_v - last_v_ui_update) >= 0.20 or idx == total:
                                last_v_ui_update = now_v
                                video_placeholder.image(
                                    cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB),
                                    channels="RGB",
                                    use_container_width=True,
                                )
                                short_scen, _, _, hex_color, _ = classify_scenario(count)
                                v_stats_placeholder.markdown(
                                    render_live_stats_html(
                                        count=count,
                                        unique_val=unique_in_video,
                                        peak_seen=max_in_video,
                                        short_scen=short_scen,
                                        hex_color=hex_color,
                                        fps_ema=extra["processing_fps"],
                                        lat_ema=extra["latency_ms"],
                                        blur_faces=blur_faces,
                                    ),
                                    unsafe_allow_html=True,
                                )

                            prog = min(1.0, float(idx) / max(total, 1))
                            progress_bar.progress(prog, text=f"Processando frame {idx}/{total} ({prog * 100:.0f}%)")

                        progress_bar.progress(1.0, text="Processamento concluído com sucesso!")
                        st.success(
                            f"Análise concluída! Pico simultâneo: **{max_in_video} pessoas** | "
                            f"Total de pessoas únicas rastreadas: **{unique_in_video} pessoas**."
                        )

                        if history_counts:
                            st.markdown("###### Curva Temporal de Ocupação no Vídeo (Bruta vs. Estabilizada):")
                            df_chart = pd.DataFrame({
                                "Contagem Instantânea": history_counts,
                                "Ocupação Estabilizada (Mediana)": history_stabilized,
                            })
                            st.line_chart(df_chart)

                        if export_path is not None and export_path.exists():
                            with export_path.open("rb") as f_vid:
                                st.download_button(
                                    label="⬇ Baixar Vídeo Processado e Anotado (.MP4)",
                                    data=f_vid.read(),
                                    file_name=export_path.name,
                                    mime="video/mp4",
                                )
                    except Exception as ex:
                        st.error(f"Erro durante o processamento do vídeo: {ex}")
                    finally:
                        if os.path.exists(media_path):
                            os.unlink(media_path)

    # =========================================================================
    # TAB 3: DASHBOARD & HISTÓRICO
    # =========================================================================
    with tab_dashboard:
        st.markdown("### 📊 Histórico Analítico e Indicadores de Fluxo (SQLite + CSV)")

        df_records = load_records_df()
        if not df_records.empty:
            total_analyses = len(df_records)
            max_people = int(df_records["quantidade"].max()) if "quantidade" in df_records else 0
            avg_people = float(df_records["quantidade"].mean()) if "quantidade" in df_records else 0.0

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(
                    f'<div class="metric-card notranslate" translate="no"><div class="metric-val">{total_analyses}</div><div class="metric-label">Análises Registradas</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi2:
                st.markdown(
                    f'<div class="metric-card notranslate" translate="no"><div class="metric-val">{max_people}</div><div class="metric-label">Maior Pico Registrado</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi3:
                st.markdown(
                    f'<div class="metric-card notranslate" translate="no"><div class="metric-val">{avg_people:.1f}</div><div class="metric-label">Média de Pessoas</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi4:
                most_common_scen = (
                    df_records["cenario"].mode()[0]
                    if "cenario" in df_records and not df_records["cenario"].empty
                    else "N/A"
                )
                st.markdown(
                    f'<div class="metric-card notranslate" translate="no"><div class="metric-val" style="font-size: 1.35rem;">{most_common_scen}</div><div class="metric-label">Cenário Predominante</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            col_ch1, col_ch2 = st.columns([2, 1])
            with col_ch1:
                st.markdown("##### Evolução das Contagens Registradas")
                chart_cols = ["quantidade"]
                if "pessoas_unicas" in df_records.columns:
                    df_records["pessoas_unicas"] = pd.to_numeric(df_records["pessoas_unicas"], errors="coerce").fillna(df_records["quantidade"])
                    chart_cols.append("pessoas_unicas")
                st.line_chart(df_records[chart_cols], use_container_width=True)
            with col_ch2:
                st.markdown("##### Distribuição por Cenário")
                if "cenario" in df_records.columns:
                    scen_counts = df_records["cenario"].value_counts()
                    st.bar_chart(scen_counts, use_container_width=True)

            st.markdown("##### Tabela Detalhada de Registros")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                types_avail = list(df_records["tipo"].unique())
                filter_type = st.multiselect("Filtrar por Tipo de Entrada", options=types_avail, default=types_avail)
            with col_f2:
                scens_avail = list(df_records["cenario"].unique()) if "cenario" in df_records.columns else []
                filter_scen = st.multiselect("Filtrar por Cenário", options=scens_avail, default=scens_avail)
            with col_f3:
                min_q = st.number_input("Quantidade Mínima de Pessoas", min_value=0, value=0, step=1)

            mask = (df_records["tipo"].isin(filter_type)) & (df_records["quantidade"] >= min_q)
            if filter_scen and "cenario" in df_records.columns:
                mask = mask & (df_records["cenario"].isin(filter_scen))

            filtered_df = df_records[mask].sort_values(by="id", ascending=False) if "id" in df_records.columns else df_records[mask]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            col_act1, col_act2 = st.columns([1, 1])
            with col_act1:
                csv_data = filtered_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Exportar Relatório em CSV",
                    data=csv_data,
                    file_name=f"relatorio_visao_cidada_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                )
            with col_act2:
                if st.button("🗑️ Limpar Histórico de Registros"):
                    clear_detection_records()
                    st.success("Histórico limpo com sucesso!")
                    st.rerun()
        else:
            st.info("Nenhuma análise registrada no histórico ainda. Execute uma detecção por foto, vídeo ou webcam para visualizar os dados.")

    # =========================================================================
    # TAB 4: SISTEMA E MODELO
    # =========================================================================
    with tab_settings:
        st.markdown("### ⚙️ Informações do Modelo, Arquitetura Back-End e Diagnóstico")
        hw = get_system_hardware_info()

        st.markdown("##### 🧠 Modelo Oficial Único e Capacidades")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                - **Modelo Ativo:** `{selected_model_name}` (YOLO11m Especializado — **mAP50: 79.2%**, **Recall: 69.9%**)
                - **Resolução de Inferência:** `{IMAGE_SIZE}x{IMAGE_SIZE} px` (`640x640` via Intel OpenVINO)
                - **Classes Reconhecidas:** `{getattr(model, 'names', {0: 'pessoa'})}`
                - **Motor Alta Densidade:** Tiling Sobreposto + Fusão Ponderada (WBF) + Supressão de Contenção (IoA)
                """
            )
        with col_m2:
            st.markdown(
                """
                - **Rastreamento Multi-Objeto (MOT):** `PersonTracker` (IoU + Velocidade + Pessoas Únicas)
                - **Anonimização LGPD:** Detecção adaptativa para Close-Ups de Webcam e Corpo Inteiro
                - **Persistência de Dados:** Banco relacional `SQLite (WAL)` + sincronização `CSV` com desduplicação
                - **Renderização Web Estável:** Painel DOM consolidado com blindagem `notranslate` contra erros `removeChild`
                """
            )

        st.markdown("---")
        st.markdown("##### 📦 Diagnóstico Real do Notebook e Aceleradores")
        col_env1, col_env2 = st.columns(2)
        with col_env1:
            st.write(f"**Processador (CPU):** {hw['cpu_name']} ({hw['cpu_count']} threads lógicas)")
            st.write(f"**Unidade Gráfica (GPU):** {hw['gpu_name']}")
            st.write(f"**Acelerador Intel OpenVINO:** {'Ativo (' + str(hw['openvino_version']) + ')' if hw['openvino_available'] else 'Indisponível'}")
        with col_env2:
            st.write(f"**Sistema Operacional:** {hw['os_platform']}")
            st.write(f"**PyTorch / OpenCV:** `{hw['torch_version']}` / `{cv2.__version__}`")
            st.write(f"**Dispositivo Selecionado:** `{selected_device}`")

        st.markdown("---")
        st.markdown("##### 🚀 Execução em Janela Desktop / Modo Feira")
        st.markdown(
            """
            Você pode iniciar a janela nativa de alta taxa de quadros tanto pelo botão na aba **Webcam** quanto pelo terminal:
            ```powershell
            python app.py --feira --model best.pt --source 0
            ```
            *(Ou execute `iniciar_webcam_desktop.bat` na raiz do projeto)*

            **Teclas de Atalho na Janela Desktop:**
            - `[D]`: Alterna Modo Alta Densidade (Tiling para grupos/multidões)
            - `[B]`: Alterna Desfoque Facial Estrito (LGPD)
            - `[H]`: Alterna Mapa de Calor Térmico de Ocupação
            - `[+]` / `[-]`: Ajusta sensibilidade (confiança) em tempo real
            - `[S]`: Salva snapshot instantâneo em `logs/snapshots/`
            - `[Q]` ou `[ESC]`: Encerra a demonstração
            """
        )


def _is_running_in_streamlit() -> bool:
    """Detecta se o script está sendo executado dentro do runtime do Streamlit."""
    try:
        from streamlit.runtime import exists

        return bool(exists())
    except Exception:
        return False


if __name__ == "__main__":
    if _is_running_in_streamlit():
        main()
    else:
        parser = argparse.ArgumentParser(description="Visão Cidadã YOLO - Aplicativo Unificado (Web & Feira)")
        parser.add_argument("--feira", action="store_true", help="Executa em janela nativa OpenCV (Modo Feira)")
        parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME, help="Caminho para o modelo (best.pt)")
        parser.add_argument("--source", type=str, default="0", help="0 para webcam local, ou caminho de vídeo/RTSP")
        parser.add_argument("--conf", type=float, default=0.25, help="Limiar de confiança")
        parser.add_argument("--iou", type=float, default=0.55, help="Limiar IoU para NMS em grupos")
        parser.add_argument("--imgsz", type=int, default=960, help="Resolução de referência")
        parser.add_argument("--device", type=str, default=DEVICE, help="Dispositivo (intel:gpu, intel:cpu, cpu)")
        parser.add_argument("--high-density", action="store_true", help="Inicia com Modo Alta Densidade ativo")
        parser.add_argument("--blur", action="store_true", help="Inicia com Desfoque Facial LGPD ativo")
        cli_args, _ = parser.parse_known_args()

        run_feira_desktop_mode(
            model_name=cli_args.model,
            source=cli_args.source,
            conf=cli_args.conf,
            iou=cli_args.iou,
            device=cli_args.device,
            alta_densidade_init=cli_args.high_density,
            blur_init=cli_args.blur,
        )

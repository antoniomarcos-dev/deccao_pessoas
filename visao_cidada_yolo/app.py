"""
Visão Cidadã YOLO - Sistema Profissional de Monitoramento e Contagem de Pessoas.
Interface Web Moderna, Inteligente e Completa com Suporte a Webcam em Tempo Real.
"""

from datetime import datetime
import os
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from config import (
    CONFIDENCE,
    DEFAULT_MODEL_NAME,
    DEVICE,
    IOU_THRESHOLD,
    RECORDS_FILE,
    ROOT_DIR,
    SNAPSHOTS_DIR,
    ensure_project_directories,
    get_available_models,
    IMAGE_SIZE,
)
from inference import (
    annotate_frame,
    count_people,
    detect_people,
    extract_detections_info,
    load_model,
    load_records_df,
    process_video,
    register_detection,
    save_snapshot,
    apply_preprocessing,
)


def apply_custom_styles() -> None:
    """Aplica folha de estilos CSS personalizada para uma interface de nível empresarial."""
    st.markdown(
        """
        <style>
        /* Estilos Globais e Tipografia */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Top Header & Badges */
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

        /* Cards de Métricas */
        .metric-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 1.2rem;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }
        .metric-val {
            font-size: 2.2rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.1;
        }
        .metric-label {
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            margin-top: 0.4rem;
        }

        /* Banner Informativo */
        .privacy-box {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #818CF8;
            padding: 0.9rem 1.2rem;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #CBD5E1;
            margin: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header(device_str: str, blur_enabled: bool, model_name: str) -> None:
    """Renderiza cabeçalho com identificadores visuais."""
    device_label = "GPU CUDA Ativa" if "cuda" in device_str.lower() or device_str == "0" else "CPU Multithread"
    privacy_label = "Privacidade Facial: Ativada" if blur_enabled else "Privacidade: Desativada"
    
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <h1 class="app-title">Visão Cidadã YOLO</h1>
                <div class="app-subtitle">Inteligência Computacional para Contagem e Densidade de Pessoas</div>
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="status-badge badge-active">● Modelo: {Path(model_name).stem}</span>
                <span class="status-badge badge-device">⚙ {device_label}</span>
                <span class="status-badge badge-privacy">🛡 {privacy_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        help="Selecione os pesos pré-treinados ou treinados sob medida em models/.",
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
        value=300,
        step=50,
        help="Número máximo de pessoas detectadas por frame.",
    )

    blur_faces = st.sidebar.toggle(
        "🛡️ Desfoque de Rostos (Privacidade)",
        value=True,
        help="Aplica desfoque em rostos para anonimização ética e conformidade com a LGPD.",
    )

    low_light_prep = st.sidebar.toggle(
        "Pré-processamento (Low Light)",
        value=False,
        help="Aplica algoritmo CLAHE para melhorar a detecção em ambientes com baixa luminosidade.",
    )


    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🖥️ Aceleração de Hardware")
    available_devices = ["cpu"]
    if torch.cuda.is_available():
        available_devices.insert(0, "0")
    selected_device = st.sidebar.radio(
        "Dispositivo de Execução",
        options=available_devices,
        format_func=lambda d: f"NVIDIA GPU (CUDA {d})" if d == "0" else "Processador (CPU)",
        index=0 if DEVICE == "0" and torch.cuda.is_available() else (1 if len(available_devices) > 1 else 0),
    )

    st.sidebar.markdown("---")
    st.sidebar.caption("Visão Cidadã YOLO v2.0 | Processamento local e seguro")

    # Carrega modelo selecionado
    try:
        model = load_model(selected_model_name)
    except Exception as e:
        st.error(f"Erro ao inicializar o modelo '{selected_model_name}': {e}")
        return

    # Renderiza Cabeçalho
    render_header(selected_device, blur_faces, selected_model_name)

    # --- ABAS PRINCIPAIS ---
    tab_webcam, tab_media, tab_dashboard, tab_settings = st.tabs([
        "📷 Webcam em Tempo Real",
        "📁 Análise de Mídia (Fotos e Vídeos)",
        "📊 Dashboard & Histórico",
        "⚙️ Informações do Sistema & Modelo",
    ])

    # =========================================================================
    # TAB 1: WEBCAM EM TEMPO REAL
    # =========================================================================
    with tab_webcam:
        st.markdown("### 🎥 Monitoramento ao Vivo por Webcam")
        st.markdown(
            "Selecione o modo desejado: **Transmissão Contínua** para fluxo em tempo real ou **Captura Instantânea** para tirar uma foto pontual com a câmera."
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
                run_cam = st.toggle("▶ Iniciar Câmera", value=False, key="stream_cam_toggle")
            with col_ctrl3:
                st.info("Para melhor desempenho em tela cheia com atalhos de teclado, você também pode usar `python webcam.py`.")

            if run_cam:
                col_frame, col_stats = st.columns([3, 1])
                with col_stats:
                    metric_people = st.empty()
                    metric_peak = st.empty()
                    metric_fps = st.empty()
                    metric_latency = st.empty()
                    snapshot_btn_area = st.empty()

                with col_frame:
                    frame_placeholder = st.empty()

                cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW if cv2.CAP_DSHOW else 0)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(int(camera_index))

                if not cap.isOpened():
                    st.error(f"Não foi possível abrir a câmera no índice {camera_index}. Verifique se outra aplicação está utilizando-a.")
                else:
                    peak_seen = 0
                    prev_t = time.perf_counter()

                    try:
                        while st.session_state.get("stream_cam_toggle", False):
                            ret, raw_frame = cap.read()
                            if not ret:
                                st.warning("Falha ao capturar imagem da câmera.")
                                break

                            curr_t = time.perf_counter()
                            fps_val = 1.0 / max(curr_t - prev_t, 1e-5)
                            prev_t = curr_t
                            
                            proc_frame = raw_frame
                            if low_light_prep:
                                proc_frame = apply_preprocessing(proc_frame)

                            result = detect_people(
                                model, proc_frame, confidence=confidence, iou=iou_thresh, device=selected_device, max_det=max_det
                            )
                            count = count_people(result)
                            peak_seen = max(peak_seen, count)
                            annotated = annotate_frame(result, blur_faces=blur_faces)
                            latency = getattr(result, "inference_latency_ms", 0.0)

                            # Exibição do Frame
                            frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                            # Atualização de Métricas
                            metric_people.metric("👥 Pessoas Agora", count)
                            metric_peak.metric("🏆 Maior Pico", peak_seen)
                            metric_fps.metric("⚡ Taxa (FPS)", f"{fps_val:.1f}")
                            metric_latency.metric("⏱️ Latência IA", f"{latency:.1f} ms")

                            # Delay cooperativo para UI do Streamlit
                            time.sleep(0.01)

                    finally:
                        cap.release()
                        if peak_seen > 0:
                            register_detection(
                                f"Webcam (Índice {camera_index})", "webcam", peak_seen, confidence, device=selected_device
                            )
            else:
                st.markdown(
                    """
                    <div style="text-align: center; padding: 40px; background: rgba(30, 41, 59, 0.4); border-radius: 12px; border: 2px dashed rgba(255,255,255,0.1);">
                        <p style="font-size: 1.1rem; color: #94A3B8; margin: 0;">A câmera está em espera. Clique no botão <b>'▶ Iniciar Câmera'</b> acima para iniciar o monitoramento contínuo.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        else:
            # Modo Snapshot Nativo
            st.markdown("##### Tire uma foto diretamente com a webcam do seu dispositivo:")
            cam_picture = st.camera_input("Capturar foto")

            if cam_picture is not None:
                bytes_data = cam_picture.getvalue()
                cv_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if low_light_prep:
                    cv_img = apply_preprocessing(cv_img)

                col_res1, col_res2 = st.columns([3, 2])
                with st.spinner("Analisando pessoas na captura..."):
                    res = detect_people(model, cv_img, confidence=confidence, iou=iou_thresh, device=selected_device, max_det=max_det)
                    count = count_people(res)
                    annotated_shot = annotate_frame(res, blur_faces=blur_faces)
                    detections = extract_detections_info(res)
                    register_detection("Webcam Snapshot", "camera_input", count, confidence, device=selected_device)

                with col_res1:
                    st.image(cv2.cvtColor(annotated_shot, cv2.COLOR_BGR2RGB), caption=f"Resultado: {count} pessoa(s) detectada(s)", use_container_width=True)

                with col_res2:
                    st.metric("Total Detectado", count)
                    if detections:
                        st.markdown("###### Detalhes das Detecções:")
                        st.dataframe(pd.DataFrame(detections)[["id", "confidence", "bbox"]], hide_index=True, use_container_width=True)
                    else:
                        st.info("Nenhuma pessoa identificada na imagem com o nível de confiança atual.")

    # =========================================================================
    # TAB 2: ANÁLISE DE MÍDIA (FOTOS E VÍDEOS)
    # =========================================================================
    with tab_media:
        st.markdown("### 📁 Análise Inteligente de Imagens e Vídeos")
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            uploaded_file = st.file_uploader(
                "Arraste ou selecione uma imagem ou vídeo para análise",
                type=["jpg", "jpeg", "png", "webp", "mp4", "avi", "mov", "mkv"],
                help="Formatos de imagem suportados: JPG, PNG, WEBP. Vídeos: MP4, AVI, MOV, MKV.",
            )
        with col_m2:
            st.markdown("<br>", unsafe_allow_html=True)
            tta_enabled = st.toggle("Modo Precisão (TTA - Test-Time Augmentation)", value=False, help="Realiza inferência múltipla redimensionando a imagem. Pode aumentar bastante a precisão de detecção de pessoas distantes às custas de maior tempo de processamento.")

        if uploaded_file is not None:
            suffix = Path(uploaded_file.name).suffix.lower()
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_media:
                temp_media.write(uploaded_file.getvalue())
                media_path = temp_media.name

            try:
                # --- PROCESSAMENTO DE IMAGEM ---
                if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
                    raw_img = cv2.imread(media_path)
                    if raw_img is None:
                        st.error("Não foi possível decodificar o arquivo de imagem enviado.")
                    else:
                        with st.spinner("Processando imagem com Inteligência Artificial..."):
                            proc_img = raw_img
                            if low_light_prep:
                                proc_img = apply_preprocessing(proc_img)
                                
                            res = detect_people(
                                model, proc_img, confidence=confidence, iou=iou_thresh, device=selected_device, augment=tta_enabled, max_det=max_det
                            )
                            count = count_people(res)
                            annotated_img = annotate_frame(res, blur_faces=blur_faces)
                            detections = extract_detections_info(res)
                            register_detection(uploaded_file.name, "imagem", count, confidence, device=selected_device)

                        # Exibição Lado a Lado
                        col_img1, col_img2 = st.columns(2)
                        with col_img1:
                            st.markdown("##### Imagem Original")
                            st.image(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_img2:
                            st.markdown(f"##### Imagem com Detecção ({count} pessoas)")
                            st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)

                        # Métricas e Dados
                        m1, m2, m3 = st.columns(3)
                        with m1:
                            st.metric("👥 Pessoas Detectadas", count)
                        with m2:
                            avg_conf = np.mean([d["confidence"] for d in detections]) if detections else 0.0
                            st.metric("🎯 Confiança Média", f"{avg_conf * 100:.1f}%")
                        with m3:
                            lat = getattr(res, "inference_latency_ms", 0.0)
                            st.metric("⏱️ Tempo de Inferência", f"{lat:.1f} ms")

                        # Tabela Detalhada
                        if detections:
                            with st.expander("📋 Ver coordenadas e níveis de certeza individuais", expanded=False):
                                df_dets = pd.DataFrame(detections)
                                st.dataframe(df_dets, use_container_width=True, hide_index=True)

                        # Botão de Download
                        res_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                        _, buffer = cv2.imencode(".jpg", annotated_img)
                        st.download_button(
                            label="⬇ Baixar Imagem Anotada",
                            data=buffer.tobytes(),
                            file_name=f"analise_{Path(uploaded_file.name).stem}.jpg",
                            mime="image/jpeg",
                        )

                # --- PROCESSAMENTO DE VÍDEO ---
                else:
                    st.markdown("##### Processamento de Vídeo Frame a Frame")
                    col_vopt1, col_vopt2 = st.columns([2, 1])
                    with col_vopt1:
                        st.info("O vídeo será analisado com detecção contínua, gerando estatísticas de pico e curva de fluxo.")
                    with col_vopt2:
                        skip_frames = st.selectbox("Aceleração (Amostragem de Frames)", [1, 2, 3, 5], index=0, format_func=lambda x: f"Processar cada {x}º frame" if x > 1 else "Todos os frames (Máxima precisão)")

                    btn_process = st.button("🚀 Iniciar Análise do Vídeo", type="primary")

                    if btn_process:
                        progress_bar = st.progress(0, text="Preparando vídeo...")
                        video_placeholder = st.empty()
                        stat_col1, stat_col2, stat_col3 = st.columns(3)
                        p_cur = stat_col1.empty()
                        p_max = stat_col2.empty()
                        p_fps = stat_col3.empty()

                        max_in_video = 0
                        history_counts = []

                        gen = process_video(
                            model=model,
                            video_path=media_path,
                            confidence=confidence,
                            iou=iou_thresh,
                            blur_faces=blur_faces,
                            source_name=uploaded_file.name,
                            device=selected_device,
                            frame_step=skip_frames,
                        )

                        for ann_frame, count, idx, total, v_fps in gen:
                            max_in_video = max(max_in_video, count)
                            history_counts.append(count)

                            # Atualiza imagem a cada N frames para fluidez da web
                            if idx % (2 * skip_frames) == 0 or idx == total:
                                video_placeholder.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

                            prog = min(1.0, float(idx) / max(total, 1))
                            progress_bar.progress(prog, text=f"Processando frame {idx}/{total} ({prog * 100:.0f}%)")
                            p_cur.metric("Pessoas no Frame", count)
                            p_max.metric("Pico Máximo Observado", max_in_video)
                            p_fps.metric("FPS do Vídeo", f"{v_fps:.1f}")

                        progress_bar.progress(1.0, text="Processamento concluído com sucesso!")
                        st.success(f"Análise concluída! Maior aglomeração observada no vídeo: {max_in_video} pessoas simultâneas.")

                        if history_counts:
                            st.markdown("###### Curva Temporal de Ocupação no Vídeo:")
                            st.line_chart(pd.DataFrame({"Pessoas por Frame": history_counts}))

            except Exception as ex:
                st.error(f"Erro durante o processamento da mídia: {ex}")
            finally:
                if os.path.exists(media_path):
                    os.unlink(media_path)

    # =========================================================================
    # TAB 3: DASHBOARD & HISTÓRICO
    # =========================================================================
    with tab_dashboard:
        st.markdown("### 📊 Histórico e Indicadores de Fluxo")

        df_records = load_records_df()
        if not df_records.empty:

            # KPIs Principais
            total_analyses = len(df_records)
            max_people = df_records["quantidade"].max() if "quantidade" in df_records else 0
            avg_people = df_records["quantidade"].mean() if "quantidade" in df_records else 0.0

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{total_analyses}</div><div class="metric-label">Análises Realizadas</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi2:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{max_people}</div><div class="metric-label">Maior Pico Registrado</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi3:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val">{avg_people:.1f}</div><div class="metric-label">Média por Análise</div></div>',
                    unsafe_allow_html=True,
                )
            with kpi4:
                most_common_type = df_records["tipo"].mode()[0] if not df_records.empty and "tipo" in df_records else "N/A"
                st.markdown(
                    f'<div class="metric-card"><div class="metric-val" style="font-size: 1.5rem; text-transform: capitalize;">{most_common_type}</div><div class="metric-label">Entrada Mais Comum</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown("---")

            # Gráfico de Tendência
            st.markdown("##### Histórico de Contagens Recentes")
            st.line_chart(df_records["quantidade"], use_container_width=True)

            # Tabela Completa com Filtros
            st.markdown("##### Tabela Detalhada de Registros")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                filter_type = st.multiselect("Filtrar por Tipo", options=df_records["tipo"].unique(), default=df_records["tipo"].unique())
            with col_f2:
                min_q = st.number_input("Quantidade Mínima de Pessoas", min_value=0, value=0, step=1)

            filtered_df = df_records[(df_records["tipo"].isin(filter_type)) & (df_records["quantidade"] >= min_q)]
            st.dataframe(filtered_df, use_container_width=True, hide_index=True)

            # Exportação CSV
            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Exportar Relatório em CSV",
                data=csv_data,
                file_name=f"relatorio_deteccoes_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
            )
        else:
            st.info("Nenhuma análise registrada no histórico ainda. Execute uma detecção por foto, vídeo ou webcam para visualizar os dados.")

    # =========================================================================
    # TAB 4: SISTEMA E MODELO
    # =========================================================================
    with tab_settings:
        st.markdown("### ⚙️ Informações do Modelo e Diagnóstico do Sistema")

        st.markdown("##### 🧠 Modelo Ativo e Capacidades de Detecção")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown(
                f"""
                - **Pesos Carregados:** `{selected_model_name}`
                - **Resolução de Entrada:** `{IMAGE_SIZE}x{IMAGE_SIZE} px`
                - **Classes Reconhecidas:** `{getattr(model, 'names', {0: 'pessoa'})}`
                - **Origem:** Treinamento especializado via Google Colab (`best.pt`)
                """
            )
        with col_m2:
            st.markdown(
                """
                - **Otimização:** Filtro direto na classe `pessoa` (ID 0)
                - **Anonimização:** Desfoque automático de rostos com OpenCV Haar Cascade
                - **Registro:** Salvamento automático de métricas em CSV local
                """
            )

        st.markdown("---")
        st.markdown("##### 📦 Ambiente de Execução e Hardware")
        col_env1, col_env2 = st.columns(2)
        with col_env1:
            st.write(f"**Python Runtime:** {os.sys.version.split()[0]}")
            st.write(f"**PyTorch:** {torch.__version__}")
            st.write(f"**Suporte a CUDA (NVIDIA):** {'Disponível' if torch.cuda.is_available() else 'Não detectado (Modo CPU)'}")
        with col_env2:
            st.write(f"**OpenCV:** {cv2.__version__}")
            st.write(f"**Diretório de Snapshots:** `{SNAPSHOTS_DIR}`")
            st.write(f"**Arquivo de Registros:** `{RECORDS_FILE}`")

        st.markdown("---")
        st.markdown("##### 🚀 Monitoramento Desktop Dedicado (HUD 60+ FPS)")
        st.markdown(
            """
            Para estações fixas ou portarias onde se deseja alta performance e controle rápido por teclado:
            ```powershell
            python webcam.py --confidence 0.35
            ```
            *(Ou execute o arquivo `iniciar_webcam_desktop.bat` na raiz do projeto)*

            **Teclas de Atalho:**
            - `[S]`: Salva snapshot instantâneo com anotações
            - `[B]`: Alterna desfoque facial de privacidade (LGPD)
            - `[C]`: Zera contador de pico de ocupação
            - `[ESPAÇO]`: Congela/retoma o feed da câmera
            - `[Q]` ou `[ESC]`: Encerra a aplicação
            """
        )

        st.markdown("---")
        st.markdown("##### 🛡️ Privacidade e Conformidade com a LGPD")
        st.info(
            "O sistema opera localmente em sua máquina. O modelo não efetua reconhecimento biométrico ou identificação civil. "
            "Todas as mídias temporárias enviadas por upload são limpas automaticamente após o término do processamento."
        )


if __name__ == "__main__":
    main()

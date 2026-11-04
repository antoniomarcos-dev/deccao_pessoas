# Visão Cidadã YOLO v3.0

Sistema profissional de visão computacional para detecção, rastreamento multi-objeto (MOT), análise de densidade ocupacional e contagem em tempo real de pessoas em imagens, vídeos e transmissões de webcam, desenvolvido com foco em ética, privacidade e conformidade estrita com a LGPD.

---

## 🚀 Como Iniciar o Sistema

### 1. Inicialização em 1 Clique (Recomendado)

Na pasta raiz do projeto, basta dar um duplo clique em:

- **`iniciar_sistema.bat`**: Abre a interface web profissional completa no navegador (Streamlit).
- **`iniciar_webcam_desktop.bat`**: Abre o monitoramento dedicado via webcam com HUD de alto desempenho (60+ FPS).

### 2. Pelo Terminal (PowerShell / Prompt)

```powershell
cd visao_cidada_yolo
python -m streamlit run app.py
```

---

## ⚙️ Arquitetura Back-End Aprimorada

1. **Motor de Alta Densidade (Tiling Sobreposto 2x2 + Fusão WBF/NMS + Supressão de Contenção IoA)**:
   - Divide cenas complexas em sub-regiões sobrepostas (compatível inclusive com modelos estáticos `640x640` do Intel OpenVINO), ampliando pessoas distantes/pequenas em grupos ou multidões e fundindo as caixas delimitadoras ponderadas pela confiança sem duplicações nas bordas.
2. **Rastreamento Multi-Objeto (`PersonTracker`)**:
   - Associa detecções entre frames usando IoU + distância de centroide normalizada e predição linear de velocidade.
   - Calcula **Pessoas Únicas (Fluxo Acumulado)**, **Contagem Estabilizada (Mediana anti-oscilação)**, **Tempo Médio de Permanência (Dwell Time)** e desenha trilhas de deslocamento suaves.
3. **Mapa de Calor Térmico (`HeatmapAccumulator`)**:
   - Acumula a ocupação espacial ao longo do tempo (em webcam/vídeos) ou instantaneamente (em fotos), gerando visualização térmica das zonas mais movimentadas.
4. **Anonimização Facial Estrita e Otimizada (LGPD)**:
   - Restringe a busca do Haar Cascade aos 45% superiores de cada pessoa detectada (3x a 4x mais rápido e sem falsos positivos) e aplica **Fallback Anatômico de Cabeça** para anonimizar pessoas de perfil, inclinadas ou distantes, antes de renderizar os rótulos visuais.
5. **Persistência Híbrida Thread-Safe (`SQLite WAL` + `CSV`)**:
   - Armazena o histórico em banco relacional `logs/visao_cidada.db` sincronizado automaticamente com `logs/detections.csv`, com desduplicação inteligente (cooldown) para evitar registros repetidos em re-renderizações de interface.
6. **Captura Assíncrona de Baixa Latência (`ThreadedCamera`)**:
   - Thread daemon dedicada que mantém sempre o quadro mais recente da câmera, eliminando o atraso de buffer do driver OpenCV/DirectShow.

---

## 📷 Modos de Uso da Webcam e Feira

### A. Módulo Desktop de Alta Performance (`webcam.py`)

```powershell
python webcam.py --camera 0 --confidence 0.30
```

**Atalhos do Teclado durante o monitoramento:**

- `[D]`: Ativa / Desativa o **Modo Alta Densidade** (Tiling + Fusão WBF para grupos)
- `[T]`: Ativa / Desativa o **Rastreamento de Pessoas Únicas** e trilhas de movimento
- `[H]`: Ativa / Desativa o **Mapa de Calor Térmico** de ocupação
- `[L]`: Ativa / Desativa o **Realce Noturno (Low-Light CLAHE)**
- `[B]`: Ativa / Desativa o **Desfoque Facial Estrito (LGPD)**
- `[+]` / `[-]`: Aumenta / Diminui a sensibilidade (confiança mínima) em tempo real
- `[S]`: Salva um snapshot anotado em `logs/snapshots/`
- `[C]`: Zera contadores de pico, pessoas únicas e mapa de calor
- `[ESPAÇO]`: Congela ou retoma o feed da câmera
- `[Q]` ou `[ESC]`: Encerra a aplicação

### B. Modo Demonstração para Feira Unificado (`app.py --feira`)

```powershell
python app.py --feira --model best.pt --source 0
```
*(Ou clique no botão **"🖥️ Abrir Janela Nativa Modo Feira"** diretamente dentro da aba Webcam da interface web)*

Inclui classificação automática de cenário (`AMBIENTE LIVRE`, `INDIVIDUAL`, `EM GRUPO`, `GRUPO GRANDE`, `AGLOMERAÇÃO`), contagem de pessoas únicas e todos os atalhos interativos (`[D]`, `[B]`, `[H]`, `[S]`, `[+/-]`).

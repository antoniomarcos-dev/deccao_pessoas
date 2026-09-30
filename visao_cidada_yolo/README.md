# Visão Cidadã YOLO

Sistema profissional de visão computacional para detecção, monitoramento e contagem em tempo real de pessoas em imagens, vídeos e transmissões de webcam, desenvolvido com foco em ética, privacidade e conformidade com a LGPD (anonimização facial por desfoque).

---

## 🚀 Como Iniciar o Sistema

### 1. Inicialização em 1 Clique (Recomendado)

Na pasta raiz do projeto, basta dar um duplo clique em:

- **`iniciar_sistema.bat`**: Abre a interface web profissional completa no navegador.
- **`iniciar_webcam_desktop.bat`**: Abre o monitoramento via webcam com HUD de alto desempenho.

### 2. Pelo Terminal (PowerShell / Prompt)

```powershell
cd visao_cidada_yolo
python -m streamlit run app.py
```

---

## 📷 Modos de Uso da Webcam

### A. Interface Web (Streamlit)

Acesse a aba **"📷 Webcam em Tempo Real"** na aplicação web:

- **Transmissão Contínua**: Exibe o feed da câmera com caixas delimitadoras, contagem ao vivo, pico máximo, taxa de FPS e latência em milissegundos.
- **Captura Instantânea**: Permite tirar fotos pontuais com a câmera do notebook ou webcam para análise imediata.

### B. Módulo Desktop de Alta Performance (`webcam.py`)

Para estações dedicadas que exigem alta taxa de quadros (60+ FPS):

```powershell
python webcam.py --camera 0 --confidence 0.35
```

**Atalhos do Teclado durante o monitoramento:**

- `[S]`: Salva um snapshot anotado em `logs/snapshots/`
- `[B]`: Ativa / Desativa o desfoque facial de privacidade
- `[C]`: Zera o contador de pico máximo de pessoas
- `[ESPAÇO]`: Congela ou retoma o feed da câmera
- `[Q]` ou `[ESC]`: Encerra a aplicação

---

## 🧠 Modelo de Inteligência Artificial (`best.pt`)

O sistema utiliza como modelo base os pesos customizados **`best.pt`**, treinados via Google Colab especificamente para detecção e contagem de alta precisão da classe `pessoa` (ID 0).

- **Pesos Padrão:** `visao_cidada_yolo/best.pt`
- **Treinamento e Validação:** Realizados em nuvem (Google Colab).
- **Repositório Local:** Otimizado exclusivamente para execução em tempo real, alta performance de inferência e monitoramento ético.

---

## 🛡️ Ética e Privacidade (LGPD)

- **Filtro Exclusivo de Classe:** O modelo processa estritamente a classe `pessoa` (ID 0) do YOLO. Não realiza biometria facial, reconhecimento de identidade ou cruzamento de dados pessoais.
- **Anonimização Automática:** Desfoque facial por algoritmo de visão computacional ativado por padrão em fotos, vídeos e webcam.
- **Dados Temporários:** Arquivos de mídia enviados via upload são processados em memória/temporários e descartados ao término da análise.

"""Configuração central do projeto Visão Cidadã YOLO."""

import os
from pathlib import Path
import torch

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
DATASET_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
RECORDS_FILE = LOGS_DIR / "detections.csv"

os.environ.setdefault("YOLO_CONFIG_DIR", str(LOGS_DIR / "ultralytics"))

# Modelo padrão treinado no Google Colab
DEFAULT_MODEL_NAME = "best.pt"
MODEL_NAME = DEFAULT_MODEL_NAME
MODEL_PATH = ROOT_DIR / DEFAULT_MODEL_NAME

# Parâmetros de inferência
IMAGE_SIZE = 800
CONFIDENCE = 0.30
IOU_THRESHOLD = 0.50
PERSON_CLASS_ID = 0


def get_available_devices() -> list[dict[str, str]]:
    """Detecta os aceleradores de hardware disponíveis no sistema (GPU Intel, CUDA, CPU)."""
    devices = []
    
    # 1. GPU Intel Integrada via OpenVINO
    try:
        import openvino as ov
        core = ov.Core()
        ov_devs = core.available_devices
        if "GPU" in ov_devs:
            gpu_name = core.get_property("GPU", "FULL_DEVICE_NAME")
            devices.append({"id": "intel:gpu", "name": f"🚀 GPU Intel Integrada ({gpu_name})", "type": "intel_gpu"})
        if "CPU" in ov_devs:
            cpu_name = core.get_property("CPU", "FULL_DEVICE_NAME")
            devices.append({"id": "intel:cpu", "name": f"⚡ CPU Intel Otimizada ({cpu_name})", "type": "intel_cpu"})
    except Exception:
        pass

    # 2. NVIDIA CUDA
    if torch.cuda.is_available():
        devices.append({"id": "0", "name": f"NVIDIA GPU (CUDA {torch.cuda.get_device_name(0)})", "type": "cuda"})

    # 3. CPU PyTorch padrão
    devices.append({"id": "cpu", "name": "💻 Processador Convencional (CPU Padrão)", "type": "cpu"})
    return devices


def get_default_device() -> str:
    env_device = os.environ.get("YOLO_DEVICE")
    if env_device is not None:
        return env_device
    devs = get_available_devices()
    return devs[0]["id"] if devs else "cpu"


DEVICE = get_default_device()


def ensure_project_directories() -> None:
    for directory in (MODELS_DIR, LOGS_DIR, SNAPSHOTS_DIR, LOGS_DIR / "ultralytics"):
        directory.mkdir(parents=True, exist_ok=True)


def get_available_models() -> list[str]:
    ensure_project_directories()
    models = []

    # Prioriza o best_openvino_model se existir (acelerado para Intel GPU/CPU)
    ov_model = ROOT_DIR / "best_openvino_model"
    if ov_model.exists():
        models.append("best_openvino_model")

    # Prioriza o best.pt na raiz ou na pasta models/
    best_root = ROOT_DIR / "best.pt"
    best_models = MODELS_DIR / "best.pt"

    if best_root.exists():
        models.append("best.pt")
    elif best_models.exists():
        models.append(str(best_models.relative_to(ROOT_DIR)))

    # Descobre outros modelos .pt na pasta models/
    for pt in MODELS_DIR.glob("**/*.pt"):
        rel = str(pt.relative_to(ROOT_DIR))
        if rel not in models:
            models.append(rel)

    # Descobre outros modelos .pt na raiz
    for pt in ROOT_DIR.glob("*.pt"):
        if pt.name not in models:
            models.append(pt.name)

    if not models:
        models.append(DEFAULT_MODEL_NAME)

    return list(dict.fromkeys(models))


ensure_project_directories()


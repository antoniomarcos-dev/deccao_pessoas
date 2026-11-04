"""Configuração central e autodetecção de hardware do projeto Visão Cidadã YOLO."""

import os
import platform
from pathlib import Path
from typing import Any

import torch

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "dataset"
DATASET_YAML = DATASET_DIR / "data.yaml"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
SNAPSHOTS_DIR = LOGS_DIR / "snapshots"
VIDEOS_EXPORT_DIR = LOGS_DIR / "videos_processados"
RECORDS_FILE = LOGS_DIR / "detections.csv"
DB_FILE = LOGS_DIR / "visao_cidada.db"

os.environ.setdefault("YOLO_CONFIG_DIR", str(LOGS_DIR / "ultralytics"))

# Modelo padrão treinado no Google Colab
DEFAULT_MODEL_NAME = "best.pt"
MODEL_NAME = DEFAULT_MODEL_NAME
MODEL_PATH = ROOT_DIR / DEFAULT_MODEL_NAME

# Parâmetros de inferência padrão
IMAGE_SIZE = 800
OPENVINO_IMAGE_SIZE = 640
CONFIDENCE = 0.30
IOU_THRESHOLD = 0.50
PERSON_CLASS_ID = 0
MAX_DETECTIONS_DEFAULT = 300


def get_available_devices() -> list[dict[str, str]]:
    """Detecta os aceleradores de hardware disponíveis no sistema (GPU Intel, CPU OpenVINO, CUDA, CPU PyTorch)."""
    devices: list[dict[str, str]] = []

    # 1. GPU e CPU Intel via OpenVINO
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
        devices.append({"id": "0", "name": f"🔥 NVIDIA GPU (CUDA {torch.cuda.get_device_name(0)})", "type": "cuda"})

    # 3. CPU PyTorch padrão (suporta resolução dinâmica, TTA e Multi-Scale completo)
    devices.append({"id": "cpu", "name": "💻 Processador Multithread (PyTorch Dinâmico)", "type": "cpu"})
    return devices


def get_default_device() -> str:
    env_device = os.environ.get("YOLO_DEVICE")
    if env_device is not None:
        return env_device
    devs = get_available_devices()
    return devs[0]["id"] if devs else "cpu"


DEVICE = get_default_device()


def get_system_hardware_info() -> dict[str, Any]:
    """Retorna diagnóstico detalhado do hardware e aceleradores disponíveis."""
    cpu_label = platform.processor() or "Processador x86_64"
    gpu_label = "Não detectada / Padrão do Sistema"
    openvino_available = False
    openvino_version = "N/A"

    try:
        import openvino as ov

        openvino_available = True
        openvino_version = getattr(ov, "__version__", "Ativo")
        core = ov.Core()
        if "CPU" in core.available_devices:
            cpu_label = core.get_property("CPU", "FULL_DEVICE_NAME")
        if "GPU" in core.available_devices:
            gpu_label = core.get_property("GPU", "FULL_DEVICE_NAME")
    except Exception:
        pass

    if torch.cuda.is_available():
        gpu_label = torch.cuda.get_device_name(0)

    return {
        "cpu_name": cpu_label,
        "gpu_name": gpu_label,
        "cpu_count": os.cpu_count() or 1,
        "openvino_available": openvino_available,
        "openvino_version": openvino_version,
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "os_platform": f"{platform.system()} {platform.release()}",
    }


def ensure_project_directories() -> None:
    for directory in (MODELS_DIR, LOGS_DIR, SNAPSHOTS_DIR, VIDEOS_EXPORT_DIR, LOGS_DIR / "ultralytics"):
        directory.mkdir(parents=True, exist_ok=True)


def get_available_models() -> list[str]:
    """Descobre todos os modelos disponíveis (.pt e diretórios OpenVINO), priorizando `best.pt` e `best_openvino_model`."""
    ensure_project_directories()
    models: list[str] = []

    # 1. Prioriza o best.pt na raiz ou na pasta models/ (flexível para tanto PyTorch quanto redirecionamento OpenVINO)
    best_root = ROOT_DIR / "best.pt"
    best_models = MODELS_DIR / "best.pt"
    if best_root.exists():
        models.append("best.pt")
    elif best_models.exists():
        models.append(str(best_models.relative_to(ROOT_DIR)))

    # 2. Modelo OpenVINO exportado
    ov_model = ROOT_DIR / "best_openvino_model"
    if ov_model.exists():
        models.append("best_openvino_model")

    # 3. Outros modelos .pt na pasta models/
    for pt in sorted(MODELS_DIR.glob("**/*.pt")):
        rel = str(pt.relative_to(ROOT_DIR))
        if rel not in models:
            models.append(rel)

    # 4. Outros modelos .pt na raiz (ex: best (2).pt, best_antigo.pt)
    for pt in sorted(ROOT_DIR.glob("*.pt")):
        if pt.name not in models:
            models.append(pt.name)

    if not models:
        models.append(DEFAULT_MODEL_NAME)

    return list(dict.fromkeys(models))


ensure_project_directories()

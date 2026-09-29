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

DEFAULT_MODEL_NAME = "yolo11m.pt"
MODEL_NAME = DEFAULT_MODEL_NAME
MODEL_PATH = ROOT_DIR / DEFAULT_MODEL_NAME
IMAGE_SIZE = 800
EPOCHS = 250
BATCH_SIZE = -1
WORKERS = 4
CONFIDENCE = 0.30
IOU_THRESHOLD = 0.50

# Training hyperparameters
OPTIMIZER = "AdamW"  # superior to default SGD for fine-tuning
LR0 = 0.001  # initial learning rate
LRF = 0.01  # final learning rate factor 
WARMUP_EPOCHS = 5.0  # warmup for stable convergence
WARMUP_MOMENTUM = 0.8
WEIGHT_DECAY = 0.0005
MOSAIC = 1.0  # mosaic augmentation (critical for crowd detection)
MIXUP = 0.15  # mixup augmentation
COPY_PASTE = 0.1  # copy-paste augmentation for more person instances
DEGREES = 10.0  # rotation augmentation
TRANSLATE = 0.2  # translation augmentation
SCALE = 0.5  # scale augmentation
FLIPUD = 0.01  # vertical flip (slight)
FLIPLR = 0.5  # horizontal flip
HSV_H = 0.015
HSV_S = 0.7
HSV_V = 0.4
PATIENCE = 30  # early stopping patience
CLOSE_MOSAIC = 15  # disable mosaic in last N epochs for fine-tuning
AMP = True  # automatic mixed precision
LABEL_SMOOTHING = 0.01  # slight label smoothing to prevent overconfidence
MULTI_SCALE = True  # multi-scale training
OVERLAP_MASK = True
RECT = False  # rectangular training off for better augmentation
COS_LR = True  # cosine learning rate scheduler

# Augmentation Presets
AUGMENTATION_PRESETS = {
    'light': {
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 0.0, 'translate': 0.1, 'scale': 0.5,
        'flipud': 0.0, 'fliplr': 0.5,
        'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0
    },
    'medium': {
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 10.0, 'translate': 0.2, 'scale': 0.5,
        'flipud': 0.01, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.15, 'copy_paste': 0.1
    },
    'aggressive': {
        'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4,
        'degrees': 45.0, 'translate': 0.3, 'scale': 0.9,
        'flipud': 0.1, 'fliplr': 0.5,
        'mosaic': 1.0, 'mixup': 0.3, 'copy_paste': 0.3
    }
}

def get_default_device() -> str:
    env_device = os.environ.get("YOLO_DEVICE")
    if env_device is not None:
        return env_device
    if torch.cuda.is_available():
        return "0"
    return "cpu"

DEVICE = get_default_device()
PERSON_CLASS_ID = 0

def ensure_project_directories() -> None:
    for directory in (DATASET_DIR, MODELS_DIR, LOGS_DIR, SNAPSHOTS_DIR, LOGS_DIR / "ultralytics"):
        directory.mkdir(parents=True, exist_ok=True)

def get_available_models() -> list[str]:
    ensure_project_directories()
    models = []
    if MODEL_PATH.exists():
        models.append(str(MODEL_PATH.name))
    for pt in MODELS_DIR.glob("**/*.pt"):
        rel = pt.relative_to(ROOT_DIR)
        models.append(str(rel))
    if not models:
        models.append(DEFAULT_MODEL_NAME)
    return list(dict.fromkeys(models))

def require_dataset_yaml() -> Path:
    if not DATASET_YAML.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATASET_YAML}. "
            "Execute check_dataset.py para validar o ambiente."
        )
    return DATASET_YAML

ensure_project_directories()

"""Valida um peso treinado contra o dataset configurado."""

import argparse
from pathlib import Path

from config import DATASET_YAML, DEVICE, IMAGE_SIZE, require_dataset_yaml
from ultralytics import YOLO
from utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Caminho para o arquivo .pt")
    args = parser.parse_args()

    logger = get_logger("validate")
    dataset_yaml = require_dataset_yaml()
    weights = Path(args.weights)
    if not weights.exists():
        parser.error(f"Pesos nao encontrados: {weights}")

    metrics = YOLO(str(weights)).val(
        data=str(dataset_yaml), imgsz=IMAGE_SIZE, device=DEVICE
    )
    logger.info("mAP50-95: %.4f", metrics.box.map)
    logger.info("mAP50: %.4f", metrics.box.map50)


if __name__ == "__main__":
    main()

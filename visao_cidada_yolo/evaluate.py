"""Calcula metricas YOLO e salva exemplos visuais da validacao."""

import argparse
from pathlib import Path

from config import DATASET_YAML, DEVICE, IMAGE_SIZE, LOGS_DIR, require_dataset_yaml
from ultralytics import YOLO
from utils.logger import get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Caminho para os pesos treinados")
    args = parser.parse_args()
    logger = get_logger("evaluate")
    require_dataset_yaml()
    output = LOGS_DIR / "evaluation"
    metrics = YOLO(args.weights).val(data=str(DATASET_YAML), imgsz=IMAGE_SIZE, device=DEVICE,
                                    project=str(output), name="validation", plots=True, exist_ok=True)
    logger.info("Precisao: %.4f", metrics.box.mp)
    logger.info("Recall: %.4f", metrics.box.mr)
    logger.info("mAP50: %.4f", metrics.box.map50)
    logger.info("mAP50-95: %.4f", metrics.box.map)
    logger.info("Graficos e exemplos salvos em %s", Path(metrics.save_dir))


if __name__ == "__main__":
    main()

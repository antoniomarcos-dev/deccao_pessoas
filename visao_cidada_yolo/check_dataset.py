"""Verifica a estrutura e integridade do dataset YOLO."""

from pathlib import Path
import yaml

from config import DATASET_DIR, DATASET_YAML
from utils.logger import get_logger


def find_split_dirs(dataset_root: Path, split_path_str: str) -> tuple[Path, Path]:
    """Resolve os diretórios de imagens e labels para um split."""
    candidate = dataset_root / split_path_str
    if candidate.is_dir():
        images = candidate
        # Procura pasta labels correspondente (ex: train/labels)
        if candidate.name == "images":
            labels = candidate.parent / "labels"
        else:
            labels = candidate.parent / (candidate.name + "_labels")
        if not labels.is_dir():
            # tenta mesmo nível
            labels = dataset_root / candidate.parent.name / "labels"
        return images, labels

    # Tenta padrão Roboflow: split/images e split/labels
    split_name = split_path_str.replace("images/", "").replace("/images", "").strip("/")
    roboflow_images = dataset_root / split_name / "images"
    roboflow_labels = dataset_root / split_name / "labels"
    if roboflow_images.is_dir():
        return roboflow_images, roboflow_labels

    # Tenta padrão YOLO tradicional: images/split e labels/split
    yolo_images = dataset_root / "images" / split_name
    yolo_labels = dataset_root / "labels" / split_name
    return yolo_images, yolo_labels


def check_split(dataset_root: Path, split_name: str, split_path: str, logger) -> bool:
    images, labels = find_split_dirs(dataset_root, split_path)
    if not images.is_dir():
        logger.error("[%s] Pasta de imagens nao encontrada: %s", split_name, images)
        return False
    if not labels.is_dir():
        logger.warning("[%s] Pasta de labels nao encontrada: %s", split_name, labels)

    image_files = {p.stem for p in images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    label_files = {p.stem for p in labels.glob("*.txt")} if labels.is_dir() else set()

    missing_labels = image_files - label_files
    extra_labels = label_files - image_files

    if missing_labels:
        logger.warning("[%s] %d imagem(ns) sem label correspondente", split_name, len(missing_labels))
    if extra_labels:
        logger.warning("[%s] %d label(s) sem imagem correspondente", split_name, len(extra_labels))

    logger.info("[%s] OK: %d imagens, %d labels identificados em '%s'", split_name, len(image_files), len(label_files), images.parent.name)
    return len(image_files) > 0


def main() -> int:
    logger = get_logger("check_dataset")
    logger.info("Verificando integridade do dataset...")
    if not DATASET_YAML.exists():
        logger.error("Arquivo de configuracao ausente: %s", DATASET_YAML)
        return 1

    with DATASET_YAML.open(encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    names = data.get("names")
    if not names:
        logger.error("O data.yaml precisa definir 'names'")
        return 1

    dataset_root = Path(data.get("path", "."))
    if not dataset_root.is_absolute():
        dataset_root = (DATASET_YAML.parent / dataset_root).resolve()

    splits_to_check = []
    for key in ("train", "val", "test"):
        if key in data:
            splits_to_check.append((key, data[key]))

    if not splits_to_check:
        logger.error("Nenhum split (train/val/test) configurado no data.yaml")
        return 1

    results = [check_split(dataset_root, name, path_str, logger) for name, path_str in splits_to_check]
    valid = all(results)
    logger.info("Classes configuradas: %s", names)
    if valid:
        logger.info("==> Dataset valido e pronto para treinamento e avaliacao!")
    else:
        logger.error("==> Problemas encontrados na estrutura do dataset.")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
